"""Train with reviewed benign images; select on validation before held-out evaluation."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from violence_app.model import FeatureExtractor,LinearHead,save_head,PREPROCESS_VERSION
from scripts.train import metrics
from scripts.people_training import group_splits,candidate_allowed,validate_review
from scripts.view_augmentation import validate_groups

DATA=ROOT/'data/people-v3'
MODELS=ROOT/'models/people-v3'
REPORTS=ROOT/'reports/people-v3'


def negative_stats(p):
    p=np.asarray(p)
    return {'count':len(p),'false_positives':int(np.sum(p>=.5)),
            'false_positive_rate':float(np.mean(p>=.5)),
            'mean_probability':float(np.mean(p)),
            'p95_probability':float(np.percentile(p,95))}


def main():
    MODELS.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True)
    reviewed=json.loads((ROOT/'docs/people-review.json').read_text())
    candidates=json.loads((DATA/'candidates.json').read_text())
    validate_review(reviewed,candidates)
    excluded={int(k) for k in reviewed['excluded_ids']}
    items=[im for im in candidates if im['id'] not in excluded]
    # Group exact and near duplicates before splitting. Identity/photographer metadata is unavailable.
    hashes=[];parents=list(range(len(items)))
    def root(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    for i,im in enumerate(items):
        payload=(ROOT/im['path']).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=im['sha256']:
            raise ValueError('Image changed since candidate collection')
        with Image.open(ROOT/im['path']) as image:
            gray=np.asarray(image.convert('L').resize((9,8),Image.Resampling.LANCZOS))
        bits=(gray[:,1:]>gray[:,:-1]).ravel()
        dhash=sum(int(bit)<<k for k,bit in enumerate(bits))
        for j,old in enumerate(hashes):
            if (dhash^old).bit_count()<=5 or im['sha256']==items[j]['sha256']:
                parents[root(i)]=root(j)
        hashes.append(dhash)
    for i,im in enumerate(items):im['group_id']='coco-group-'+str(items[root(i)]['id'])
    split=group_splits(items)
    baseline_dir=ROOT/'models/baseline-v2'
    baseline_dir.mkdir(exist_ok=True)
    if not (baseline_dir/'violence_head.json').exists():
        for src in ['models/violence_head.json','reports/training.json','reports/predictions.csv','reports/benchmark.json']:
            shutil.copy2(ROOT/src,baseline_dir/Path(src).name)
    baseline=LinearHead(baseline_dir/'violence_head.json')
    base_manifest=ROOT/'data/views-v2/manifest.csv'
    rows=list(csv.DictReader(base_manifest.open()))
    with np.load(ROOT/'models/views-v2/embeddings.npz',allow_pickle=False) as saved:
        if str(saved['manifest_sha256'])!=hashlib.sha256(base_manifest.read_bytes()).hexdigest() or str(saved['preprocess'])!=PREPROCESS_VERSION:
            raise ValueError('Base feature cache does not match manifest/preprocessing')
        old_features=saved['features']
    for r in rows:r.update(data_source='AIRTLab',category='original-assault-dataset',group_id=r['scene_id'])
    for im in items:
        rows.append(dict(path=im['path'],label='0',label_name='non-violent',scene_id=im['group_id'],
            camera='photo',video_path='',frame_index='',timestamp_seconds='',split=split[im['id']],
            actions=im['category'],width=im['width'],height=im['height'],sha256=im['sha256'],
            pixel_sha256='',source_url=im['source_url'],source_git_sha='',
            license=im['license_info']['url'],label_basis='captions and full contact-sheet visual review; benign visible scene',
            parent_path=im['path'],augmentation='original',data_source='COCO2017',category=im['category'],group_id=im['group_id'],
            flickr_photo_page=im['flickr_photo_page'],flickr_url=im['flickr_url'],person_count=im['person_count']))
    validate_groups(rows)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    manifest=DATA/'manifest.csv'
    with manifest.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    fingerprint=hashlib.sha256(manifest.read_bytes()).hexdigest()
    cache=MODELS/'embeddings.npz'
    if cache.exists():
        with np.load(cache,allow_pickle=False) as saved:
            if str(saved['manifest_sha256'])!=fingerprint or str(saved['preprocess'])!=PREPROCESS_VERSION:
                raise ValueError('Feature cache does not match manifest/preprocessing')
            features=saved['features']
    else:
        extractor=FeatureExtractor();batches=[]
        for start in range(0,len(items),16):
            images=[]
            for im in items[start:start+16]:
                with Image.open(ROOT/im['path']) as image:images.append(image.convert('RGB'))
            batches.append(extractor.features(images))
        features=np.concatenate([old_features,*batches])
        np.savez_compressed(cache,features=features,manifest_sha256=fingerprint,preprocess=PREPROCESS_VERSION)
    features=features.astype(np.float64)
    y=np.array([int(r['label']) for r in rows]);source=np.array([r['data_source'] for r in rows])
    subsets={s:np.array([r['split']==s for r in rows]) for s in ['train','validation','test']}
    old_val=subsets['validation']&(source=='AIRTLab');new_val=subsets['validation']&(source=='COCO2017')
    training=subsets['train']
    # Each AIRTLab original and its five derivatives have total weight one.
    weights=np.array([1/6 if r['data_source']=='AIRTLab' else 1. for r in rows])
    scaler=StandardScaler().fit(features[training],sample_weight=weights[training])
    x=scaler.transform(features)
    def validation(p):
        old=metrics(y[old_val],p[old_val]);neg=negative_stats(p[new_val])
        return {'recall':old['recall'],'fpr':float(np.mean(p[old_val&(y==0)]>=.5)),
                'new_fpr':neg['false_positive_rate'],'original':old,'new_negatives':neg,
                'score':float(.5*log_loss(y[old_val],p[old_val])+.5*np.mean(-np.log(np.clip(1-p[new_val],1e-12,1))))}
    baseline_p=baseline.predict(features)
    reference=validation(baseline_p)
    print('Baseline validation:',json.dumps(reference),flush=True)
    choices=[];best=None
    for negative_weight in [1.,2.,4.]:
        sample_weight=weights.copy();sample_weight[source=='COCO2017']*=negative_weight
        for c in [.0001,.001,.01,.1,1.]:
            model=LogisticRegression(C=c,max_iter=3000,random_state=42)
            model.fit(x[training],y[training],sample_weight=sample_weight[training])
            p=model.predict_proba(x)[:,1];score=validation(p)
            allowed=candidate_allowed(reference,score)
            choice={'C':c,'new_negative_weight':negative_weight,'allowed':allowed,**score}
            choices.append(choice)
            if allowed and (best is None or score['score']<best[0]):best=(score['score'],model,choice)
    if best is None:
        (REPORTS/'validation.json').write_text(json.dumps({'baseline':reference,'choices':choices,'promote':False},indent=2))
        raise SystemExit('No candidate met validation recall/FPR guards; current model preserved.')
    _,model,selected=best
    metadata={**baseline.metadata,'manifest_path':str(manifest.relative_to(ROOT)),
        'manifest_sha256':fingerprint,'training_images':int(training.sum()),'dataset_images':len(rows),
        'original_images':sum(r['augmentation']=='original' for r in rows),
        'dataset':'AIRTLab research/education + reviewed COCO2017 CC BY / BY-SA benign photos',
        'C':selected['C'],'new_negative_weight':selected['new_negative_weight'],
        'version':'people-v3','scope':'Assault classifier with reviewed ordinary crowd/close-person negatives; no self-striking training.'}
    head_path=MODELS/'violence_head.json';save_head(head_path,scaler,model,metadata)
    decision={'promote':True,'baseline':reference,'selected':selected,'choices':choices,
        'selection_basis':'validation only; original recall decrease <= .05, original FPR increase <= .02, new negative FPR must decrease',
        'head_sha256':hashlib.sha256(head_path.read_bytes()).hexdigest(),
        'baseline_sha256':hashlib.sha256((baseline_dir/'violence_head.json').read_bytes()).hexdigest(),
        'manifest_sha256':fingerprint}
    (REPORTS/'validation.json').write_text(json.dumps(decision,indent=2))
    # Candidate is selected and saved before any test metrics are computed below.
    p=LinearHead(head_path).predict(features)
    np.testing.assert_allclose(p,model.predict_proba(x)[:,1],atol=1e-7)
    test=subsets['test'];old_test=test&(source=='AIRTLab');new_test=test&(source=='COCO2017')
    comparison={}
    for name,probs in [('baseline_v2',baseline_p),('candidate_v3',p)]:
        clips={}
        for i,r in enumerate(rows):
            if old_test[i]:clips.setdefault(r['video_path'],{'y':y[i],'p':[]})['p'].append(probs[i])
        comparison[name]={'original_frames':metrics(y[old_test],probs[old_test]),
            'original_five_frame_clip_mean':metrics(np.array([v['y'] for v in clips.values()]),np.array([np.mean(v['p']) for v in clips.values()])),
            'new_negative_photos':negative_stats(probs[new_test]),
            'negative_by_category':{cat:negative_stats(probs[new_test&np.array([r['category']==cat for r in rows])]) for cat in sorted({im['category'] for im in items})}}
    report={'metadata':metadata,'validation':decision,'test_comparison':comparison,
        'test_frames':comparison['candidate_v3']['original_frames'],
        'test_five_frame_clip_mean':comparison['candidate_v3']['original_five_frame_clip_mean'],
        'new_image_count':len(items),'excluded_image_count':len(excluded),
        'new_split_counts':dict(collections.Counter(split.values())),
        'new_group_count':len({im['group_id'] for im in items}),
        'limits':['New photos have negative labels only; this is not unseen-domain assault recall validation.',
            'COCO categories/count/box fraction select candidates, not violence labels. Captions and visual review provided labels.',
            'No user webcam images collected; ordinary webcam performance remains unmeasured.',
            'No identity/photographer split guarantee; exact/near-duplicate image grouping only.',
            'Old AIRTLab test set has been used in previous iterations; it is a regression set, not a fresh final test.',
            'Five-frame mean is unchanged and does not model motion; no self-striking training.']}
    (REPORTS/'training.json').write_text(json.dumps(report,indent=2))
    with (REPORTS/'predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','split','label','data_source','category','baseline_probability','probability']);writer.writeheader()
        for r,a,b in zip(rows,baseline_p,p):writer.writerow({**{k:r[k] for k in ['path','split','label','data_source','category']},'baseline_probability':a,'probability':b})
    print(json.dumps({'selected':selected,'test':comparison,'new_split_counts':report['new_split_counts']},indent=2),flush=True)


if __name__=='__main__':main()

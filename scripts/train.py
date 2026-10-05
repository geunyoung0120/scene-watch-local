import csv
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss, log_loss, confusion_matrix
from sklearn.preprocessing import StandardScaler
from violence_app.model import FeatureExtractor, LinearHead, save_head, PREPROCESS_VERSION


def metrics(labels, probabilities):
    pred = probabilities >= .5
    return { 'accuracy':float(accuracy_score(labels,pred)),
        'precision':float(precision_score(labels,pred,zero_division=0)),
        'recall':float(recall_score(labels,pred,zero_division=0)),
        'f1':float(f1_score(labels,pred,zero_division=0)),
        'roc_auc':float(roc_auc_score(labels, probabilities)),
        'brier_score':float(brier_score_loss(labels, probabilities)),
        'log_loss':float(log_loss(labels, probabilities)),
        'confusion_matrix':confusion_matrix(labels,pred,labels=[0,1]).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=ROOT/'data/manifest.csv')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    model_dir = args.output_dir.resolve() if args.output_dir else ROOT/'models'
    report_dir = model_dir if args.output_dir else ROOT/'reports'
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(manifest.open()))
    from scripts.view_augmentation import validate_groups
    validate_groups(rows)
    fingerprint = hashlib.sha256(manifest.read_bytes()).hexdigest()
    cache = model_dir / 'embeddings.npz'
    started = time.perf_counter()
    if cache.exists():
        with np.load(cache,allow_pickle=False) as saved:
            if str(saved['manifest_sha256']) != fingerprint or str(saved['preprocess']) != PREPROCESS_VERSION:
                raise ValueError('Stale embeddings: remove models/embeddings.npz and rerun')
            features = saved['features']
    else:
        extractor = FeatureExtractor()
        print('Feature device:',extractor.device,flush=True)
        batches = []
        for start in range(0,len(rows),16):
            images = []
            for row in rows[start:start+16]:
                with Image.open(ROOT / row['path']) as im:
                    images.append(im.convert('RGB'))
            batches.append(extractor.features(images))
            if start % 160 == 0:
                print(f'Extracted {min(start+16,len(rows))}/{len(rows)}',flush=True)
        features = np.concatenate(batches)
        np.savez_compressed(cache, features=features, manifest_sha256=fingerprint, preprocess=PREPROCESS_VERSION)
    y = np.array([int(row['label']) for row in rows])
    partitions = {s:np.array([r['split']==s for r in rows]) for s in ('train','validation','test')}
    for first, second in [('train','validation'),('train','test'),('validation','test')]:
        assert not {r['scene_id'] for r in rows if r['split']==first} & {r['scene_id'] for r in rows if r['split']==second}
    # Match the exported head's float64 arithmetic, rather than rounding scaler
    # output to the extractor's float32 dtype before logistic regression.
    features = features.astype(np.float64)
    scaler = StandardScaler().fit(features[partitions['train']])
    x = scaler.transform(features)
    candidates = []
    best = None
    for c in [.0001,.001,.01,.1,1,10]:
        lr = LogisticRegression(C=c,max_iter=3000,random_state=42)
        lr.fit(x[partitions['train']],y[partitions['train']])
        probabilities = lr.predict_proba(x[partitions['validation']])[:,1]
        score = metrics(y[partitions['validation']],probabilities)
        candidates.append({'C':c,**score})
        if best is None or score['log_loss'] < best[0]:
            best = (score['log_loss'],lr,c)
    classifier = best[1]
    metadata = {'model':'facebook/dinov2-small','feature_dim':384,'feature':'last_hidden_state[:,0,:]',
        'preprocess_version':PREPROCESS_VERSION,'manifest_sha256':fingerprint,'C':best[2],
        'label_0':'non-violent','label_1':'physical violence','backbone_frozen':True,
        'training_images':int(partitions['train'].sum()),'threshold':.5,
        'dataset':'AIRTLab; research/educational use',
        'scope':'Staged physical assault vs friendly interactions; experimental classification probability.'}
    metadata['manifest_path'] = str(manifest.relative_to(ROOT))
    metadata['augmentation'] = sorted({r.get('augmentation','original') for r in rows})
    metadata['dataset_images'] = len(rows)
    metadata['original_images'] = sum(r.get('augmentation','original')=='original' for r in rows)
    path = model_dir / 'violence_head.json'
    save_head(path,scaler,classifier,metadata)
    probabilities = classifier.predict_proba(x)[:,1]
    exported = LinearHead(path).predict(features)
    np.testing.assert_allclose(exported,probabilities,atol=1e-7)
    test = partitions['test']
    # Each held-out clip contains exactly five selected frames; average probabilities.
    clips = {}
    for row,p in zip(rows,probabilities):
        if row['split']=='test':
            clips.setdefault(row['video_path'], {'label':int(row['label']),'p':[]})['p'].append(float(p))
    clip_y = np.array([c['label'] for c in clips.values()])
    clip_p = np.array([np.mean(c['p']) for c in clips.values()])
    report = {'metadata':metadata,'validation_candidates':candidates,
        'test_frames':metrics(y[test], probabilities[test]),
        'test_five_frame_clip_mean':metrics(clip_y,clip_p),
        'test_frame_count':int(test.sum()),'test_clip_count':len(clips),
        'test_independent_scene_count':len({r['scene_id'] for r in rows if r['split']=='test'}),
        'elapsed_seconds':time.perf_counter()-started,
        'limits':['Source labels apply to clips, not exact action boundaries.',
            'Same room and actors appear in all splits; this is not an unseen-environment evaluation.',
            'Five-frame mean smooths scores and does not model movement direction.',
            '50/50 source prevalence differs from a normal webcam; scores are not real-world incident probabilities.']}
    (report_dir/'training.json').write_text(json.dumps(report,indent=2))
    with (report_dir/'predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','scene_id','camera','split','label','probability'])
        writer.writeheader()
        for row,p in zip(rows,probabilities):
            writer.writerow({**{k:row[k] for k in ['path','scene_id','camera','split','label']},'probability':float(p)})
    print(json.dumps({k:v for k,v in report.items() if k!='validation_candidates'},indent=2),flush=True)


if __name__=='__main__':
    main()

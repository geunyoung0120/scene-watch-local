"""Freeze deployment decision on validation; report original and 2D stress tests."""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scripts.train import metrics
from scripts.view_augmentation import variants
from violence_app.model import FeatureExtractor, LinearHead


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',type=Path,default=ROOT/'models/baseline-v1/violence_head.json')
    args=parser.parse_args()
    baseline=args.baseline.resolve()
    candidate=ROOT/'models/views-v2/violence_head.json'
    if not baseline.is_file():
        raise SystemExit('Save the original model in models/baseline-v1/violence_head.json before comparing.')
    manifest=ROOT/'data/views-v2/manifest.csv'
    rows=list(csv.DictReader(manifest.open()))
    with np.load(ROOT/'models/views-v2/embeddings.npz',allow_pickle=False) as saved:
        assert str(saved['manifest_sha256'])==hashlib.sha256(manifest.read_bytes()).hexdigest()
        features=saved['features']
    heads={'baseline':LinearHead(baseline),'candidate':LinearHead(candidate)}
    extractor=FeatureExtractor()
    report={'limits':['Synthetic transformations of existing views; not unseen-camera or unseen-person performance.',
        'Correlated variants do not increase the independent test scene count (20).',
        'No self-directed striking examples were added.'], 'validation':{},'test':{}}
    report['head_sha256']={name:hashlib.sha256(path.read_bytes()).hexdigest()
                          for name,path in [('baseline',baseline),('candidate',candidate)]}
    report['manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
    for split in ['validation','test']:
        subset=[(i,r) for i,r in enumerate(rows) if r['split']==split]
        originals=features[[i for i,r in subset]]
        labels=np.array([int(r['label']) for i,r in subset])
        transformed=[]; transformed_labels=[]; kinds=[]; batch=[]
        for i,row in subset:
            with Image.open(ROOT/row['path']) as im:
                for name,view in variants(im):
                    batch.append(view);transformed_labels.append(int(row['label']));kinds.append(name)
                    if len(batch)==16:
                        transformed.append(extractor.features(batch));batch=[]
        if batch:
            transformed.append(extractor.features(batch))
        transformed=np.concatenate(transformed)
        y=np.array(transformed_labels)
        for name,head in heads.items():
            p=head.predict(originals);q=head.predict(transformed)
            cameras={}
            for cam in ['cam1','cam2']:
                select=np.array([r['camera']==cam for i,r in subset])
                cameras[cam]=metrics(labels[select],p[select])
            report[split][name]={'original':metrics(labels,p),
                'synthetic_views':metrics(y,q),
                'combined':metrics(np.concatenate([labels,y]),np.concatenate([p,q])),
                'per_camera_original':cameras,
                'per_transform':{k:metrics(y[np.array(kinds)==k],q[np.array(kinds)==k]) for k in sorted(set(kinds))}}
        if split=='validation':
            base,new=report[split]['baseline'],report[split]['candidate']
            report['promotion_rule']='combined validation log loss <= baseline; original validation accuracy drop <= 0.05'
            report['promote']=bool(new['combined']['log_loss']<=base['combined']['log_loss'] and
                new['original']['accuracy']>=base['original']['accuracy']-.05-1e-12)
            report['decision_basis']='promotion uses validation metrics only; training also reports test metrics'
            (ROOT/'reports/views-v2/validation_decision.json').write_text(json.dumps(report,indent=2))
        print(split, {name:{k:round(v['accuracy'],4) for k,v in data.items() if k in ['original','synthetic_views']}
                      for name,data in report[split].items()},flush=True)
    (ROOT/'reports/views-v2/comparison.json').write_text(json.dumps(report,indent=2))
    print('Promote:',report['promote'])


if __name__=='__main__':
    main()

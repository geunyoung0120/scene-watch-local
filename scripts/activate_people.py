"""Activate the validation-selected people-v3 model after checking its artifacts."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    decision=json.loads((ROOT/'reports/people-v3/validation.json').read_text())
    if not decision['promote']:
        raise SystemExit('Validation gate did not pass; current model preserved.')
    checks=[('models/people-v3/violence_head.json','head_sha256'),
            ('models/baseline-v2/violence_head.json','baseline_sha256'),
            ('data/people-v3/manifest.csv','manifest_sha256')]
    for path,key in checks:
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=decision[key]:
            raise SystemExit(f'{path} changed since validation; retrain/evaluate again.')
    report=json.loads((ROOT/'reports/people-v3/training.json').read_text())
    if report['validation']!=decision:
        raise SystemExit('Training report and validation decision do not match.')
    copies=[('models/people-v3/violence_head.json','models/violence_head.json'),
            ('reports/people-v3/training.json','reports/training.json'),
            ('reports/people-v3/predictions.csv','reports/predictions.csv')]
    for source,target in copies:
        if not (ROOT/source).is_file():raise SystemExit(f'Missing {source}')
    for source,target in copies:
        dest=ROOT/target;tmp=dest.with_suffix(dest.suffix+'.pending')
        shutil.copyfile(ROOT/source,tmp);tmp.replace(dest)
    print('Activated people-v3. Restart server to load it.')


if __name__=='__main__':main()

"""Activate the evaluated candidate only when validation passed and hashes match.

Stop/restart the server separately to load the activated head.
"""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    report=json.loads((ROOT/'reports/views-v2/comparison.json').read_text())
    if not report['promote']:
        raise SystemExit('Validation gate did not pass; the deployed model was preserved.')
    for name,path in [('baseline',ROOT/'models/baseline-v1/violence_head.json'),
                      ('candidate',ROOT/'models/views-v2/violence_head.json')]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=report['head_sha256'][name]:
            raise SystemExit(f'{name} changed after evaluation; evaluate again.')
    manifest=ROOT/'data/views-v2/manifest.csv'
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=report['manifest_sha256']:
        raise SystemExit('Manifest changed after evaluation; evaluate again.')
    copies=[('models/views-v2/violence_head.json','models/violence_head.json'),
            ('models/views-v2/training.json','reports/training.json'),
            ('models/views-v2/predictions.csv','reports/predictions.csv')]
    for source,target in copies:
        if not (ROOT/source).is_file():
            raise SystemExit(f'Missing {source}')
    for source,target in copies:
        destination=ROOT/target
        temporary=destination.with_suffix(destination.suffix+'.pending')
        shutil.copyfile(ROOT/source,temporary)
        temporary.replace(destination)
    print('Activated views-v2. Restart the local server to load it.')


if __name__=='__main__':
    main()

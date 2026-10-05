"""Download a pinned safetensors checkpoint; no remote Python code is executed."""
from pathlib import Path
import hashlib
import json
import requests

ROOT = Path(__file__).resolve().parents[1]
REV = 'ed25f3a31f01632728cabb09d1542f84ab7b0056'
DEST = ROOT / 'models/dinov2-small'


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for name in ['config.json', 'preprocessor_config.json', 'README.md', 'model.safetensors']:
        target = DEST / name
        if target.exists():
            continue
        url = f'https://huggingface.co/facebook/dinov2-small/resolve/{REV}/{name}'
        with requests.get(url, stream=True, timeout=(20, 120)) as r:
            r.raise_for_status()
            temp = target.with_suffix(target.suffix + '.part')
            with temp.open('wb') as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
            temp.replace(target)
        print('Downloaded', name, target.stat().st_size, flush=True)
    meta = {'model':'facebook/dinov2-small','revision':REV,'license':'Apache-2.0',
        'weights_sha256':hashlib.sha256((DEST/'model.safetensors').read_bytes()).hexdigest()}
    (DEST/'provenance.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta),flush=True)


if __name__ == '__main__':
    main()

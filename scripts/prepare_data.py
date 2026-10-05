"""Fetch official AIRTLab clips and extract a reproducible, scene-disjoint subset."""
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from violence_app.core import make_scene_splits

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import requests

REPO = 'airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos'
REVISION = '1f7747e104301ccaa82ef5a2f6804b51ced1c398'
BASE = f'https://raw.githubusercontent.com/{REPO}/{REVISION}/'
DATA = ROOT / 'data'


def download(url, target, git_sha=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        content = target.read_bytes()
        if git_sha is None or hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest() == git_sha:
            return content
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=(15, 90))
            r.raise_for_status()
            content = r.content
            if git_sha and hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest() != git_sha:
                raise ValueError(f'Source checksum mismatch: {target.name}')
            tmp = target.with_suffix(target.suffix + '.part')
            tmp.write_bytes(content)
            tmp.replace(target)
            return content
        except (requests.RequestException, ValueError):
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


def main():
    (ROOT / 'reports').mkdir(parents=True, exist_ok=True)
    tree = json.loads(download(f'https://api.github.com/repos/{REPO}/git/trees/{REVISION}?recursive=1', DATA / 'references/source-tree.json'))
    blobs = {x['path']: x for x in tree['tree'] if x['type'] == 'blob'}
    for path in ['readme.md', 'violence-detection-dataset/action-class-occurrences.csv',
                 'violence-detection-dataset/violent-action-classes.csv',
                 'violence-detection-dataset/nonviolent-action-classes.csv']:
        download(BASE + path, DATA / 'references' / Path(path).name, blobs[path]['sha'])
    scenes, actions = {}, {}
    rng = random.Random(42)
    for label, filename in [(0, 'nonviolent-action-classes.csv'), (1, 'violent-action-classes.csv')]:
        candidates = []
        with (DATA / 'references' / filename).open() as f:
            for row in csv.DictReader(f, delimiter=';'):
                act = row[' ACTION CLASSES'].split(',')
                scene = int(Path(row['FILE']).stem)
                if label == 1 and set(act) & {'gunshot', 'stab', 'club'}:
                    continue
                # Friendly punch has deliberately ambiguous still-image ground truth.
                if label == 0 and 'friendly punch' in act:
                    continue
                candidates.append(scene)
                actions[label, scene] = act
        if len(candidates) < 50:
            raise ValueError(f'Only {len(candidates)} eligible scenes in class {label}')
        scenes[label] = sorted(rng.sample(candidates, 50))
    splits = make_scene_splits(scenes)
    jobs = []
    for label in (0, 1):
        category = 'violent' if label else 'non-violent'
        for scene in scenes[label]:
            for camera in ('cam1', 'cam2'):
                source_path = f'violence-detection-dataset/{category}/{camera}/{scene}.mp4'
                jobs.append((label, category, scene, camera, source_path))

    def fetch_extract(job):
        label, category, scene, camera, source_path = job
        video = DATA / 'videos' / category / camera / f'{scene}.mp4'
        download(BASE + source_path, video, blobs[source_path]['sha'])
        cap = cv2.VideoCapture(str(video))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if total < 10 or fps <= 0:
                raise ValueError(f'Cannot decode {video}')
            rows = []
            # Avoid lead-in/out frames, preserve temporal order within a short clip.
            indices = np.linspace(total * .2, total * .8, 5).astype(int)
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError(f'Unreadable frame {video}:{index}')
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(rgb)
                im.thumbnail((960, 960), Image.Resampling.LANCZOS)
                name = f'{category}_{scene:03d}_{camera}_f{index:04d}.jpg'
                target = DATA / 'images' / category / name
                target.parent.mkdir(parents=True, exist_ok=True)
                im.save(target, quality=93)
                rows.append(dict(path=str(target.relative_to(ROOT)), label=label,
                    label_name=category, scene_id=f'{category}_{scene:03d}', camera=camera,
                    video_path=str(video.relative_to(ROOT)), frame_index=int(index),
                    timestamp_seconds=round(int(index) / fps, 4), split=splits[label, scene],
                    actions=','.join(actions[label, scene]), width=im.width, height=im.height,
                    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    pixel_sha256=hashlib.sha256(im.tobytes()).hexdigest(),
                    source_url=BASE + source_path, source_git_sha=blobs[source_path]['sha'],
                    license='AIRTLab: free for research and educational purposes',
                    label_basis='source clip label; frame not individually annotated'))
            return rows
        finally:
            cap.release()

    rows = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, result in enumerate(pool.map(fetch_extract, jobs), 1):
            rows.extend(result)
            if i % 10 == 0:
                print(f'Prepared {i}/{len(jobs)} clips, {len(rows)} frames', flush=True)
    rows.sort(key=lambda r: (r['label'], r['scene_id'], r['camera'], r['frame_index']))
    assert len(rows) == 1000
    assert all(sum(r['label'] == y for r in rows) == 500 for y in (0, 1))
    assert len({r['sha256'] for r in rows}) == len(rows), 'Exact file duplicate detected'
    assert len({r['pixel_sha256'] for r in rows}) == len(rows), 'Exact pixel duplicate detected'
    groups = {}
    for row in rows:
        groups.setdefault(row['scene_id'], set()).add(row['split'])
    assert all(len(s) == 1 for s in groups.values()), 'Scene leakage'
    with (DATA / 'manifest.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'total_images': len(rows), 'per_class': {'non-violent':500,'violent':500},
        'source_revision':REVISION, 'distinct_scenes':len(groups), 'clips':len(jobs),
        'split_counts':{s:sum(r['split']==s for r in rows) for s in ('train','validation','test')},
        'exact_duplicates':0, 'scene_split_overlap':0,
        'sampling':'five frames at 20,35,50,65,80 percent of each clip',
        'limits':'One staged room and repeated actors; source clip labels are weak frame labels.'}
    (ROOT / 'reports/data_summary.json').write_text(json.dumps(summary, indent=2))
    # Compact contact sheets: one row per video, all five selected frames.
    preview = ROOT / 'reports/contact_sheets'
    preview.mkdir(exist_ok=True)
    for label in (0, 1):
        subset = [r for r in rows if r['label'] == label]
        for start in range(0, len(subset), 50):
            sheet = Image.new('RGB', (1000, 1320), '#111820')
            draw = ImageDraw.Draw(sheet)
            for j, row in enumerate(subset[start:start+50]):
                with Image.open(ROOT / row['path']) as im:
                    thumb = ImageOps.pad(im, (196, 110), color='#111820')
                x,y=(j%5)*200,(j//5)*132
                sheet.paste(thumb,(x,y))
                draw.text((x+2,y+112), f"{row['scene_id']} {row['camera']} {row['frame_index']}",fill='white')
            sheet.save(preview / f'class{label}_{start//50+1:02d}.jpg',quality=90)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

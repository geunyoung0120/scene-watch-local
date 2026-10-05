"""Add unused AIRTLab scenes and train-only view augmentations, preserving v1."""
import csv
import hashlib
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scripts.prepare_data import BASE, REVISION, download
from scripts.view_augmentation import variants, should_augment, validate_groups


def main():
    output = ROOT/'data/views-v2'
    output.mkdir(exist_ok=True)
    reports = ROOT/'reports/views-v2'
    reports.mkdir(exist_ok=True)
    rows = list(csv.DictReader((ROOT/'data/manifest.csv').open()))
    for row in rows:
        row.update(parent_path=row['path'], augmentation='original')
    known = {r['scene_id'] for r in rows}
    tree = json.loads((ROOT/'data/references/source-tree.json').read_text())
    blobs = {x['path']:x['sha'] for x in tree['tree'] if x['type']=='blob'}
    jobs = []
    for label, filename in [(0,'nonviolent-action-classes.csv'),(1,'violent-action-classes.csv')]:
        category = 'violent' if label else 'non-violent'
        eligible = []
        for record in csv.DictReader((ROOT/'data/references'/filename).open(),delimiter=';'):
            actions = record[' ACTION CLASSES'].split(',')
            scene = int(Path(record['FILE']).stem)
            if (label and set(actions)&{'gunshot','stab','club'}) or (not label and 'friendly punch' in actions):
                continue
            if f'{category}_{scene:03d}' not in known:
                eligible.append((scene,actions))
        if len(eligible)<5:
            raise ValueError('Need five unused eligible scenes per class')
        for scene, actions in sorted(eligible)[:5]:
            for camera in ('cam1','cam2'):
                jobs.append((label,category,scene,actions,camera))

    def extract(job):
        label,category,scene,actions,camera = job
        source = f'violence-detection-dataset/{category}/{camera}/{scene}.mp4'
        video = ROOT/'data/videos'/category/camera/f'{scene}.mp4'
        download(BASE+source,video,blobs[source])
        cap = cv2.VideoCapture(str(video))
        result = []
        try:
            total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
            if total<10 or fps<=0:
                raise ValueError(f'Invalid video {video}')
            for index in np.linspace(total*.2,total*.8,5).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(index))
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f'Invalid frame {video}:{index}')
                im = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                im.thumbnail((960,960),Image.Resampling.LANCZOS)
                target = output/'originals'/f'{category}_{scene:03d}_{camera}_f{index:04d}.jpg'
                target.parent.mkdir(exist_ok=True)
                im.save(target,quality=93)
                path = str(target.relative_to(ROOT))
                result.append(dict(path=path,label=str(label),label_name=category,
                    scene_id=f'{category}_{scene:03d}',camera=camera,video_path=str(video.relative_to(ROOT)),
                    frame_index=str(index),timestamp_seconds=str(round(int(index)/fps,4)),split='train',
                    actions=','.join(actions),width=str(im.width),height=str(im.height),
                    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    pixel_sha256=hashlib.sha256(im.tobytes()).hexdigest(),source_url=BASE+source,
                    source_git_sha=blobs[source],license='AIRTLab: free for research and educational purposes',
                    label_basis='source clip label; frame not individually annotated',
                    parent_path=path,augmentation='original'))
            return result
        finally:
            cap.release()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,result in enumerate(pool.map(extract,jobs),1):
            rows.extend(result)
            print(f'Additional clips {i}/{len(jobs)}',flush=True)
    originals = list(rows)
    for i,row in enumerate(originals):
        if not should_augment(row):
            continue
        with Image.open(ROOT/row['path']) as im:
            for name, augmented in variants(im):
                target = output/'augmented'/f'{Path(row["path"]).stem}_{name}.jpg'
                target.parent.mkdir(exist_ok=True)
                augmented.save(target,quality=93)
                rows.append({**row,'path':str(target.relative_to(ROOT)),'augmentation':name,
                    'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                    'pixel_sha256':hashlib.sha256(augmented.tobytes()).hexdigest()})
        if i%100==0:
            print(f'Augmented originals {i}/{len(originals)}',flush=True)
    validate_groups(rows)
    if len({r['sha256'] for r in rows})!=len(rows):
        raise ValueError('Exact file duplicates')
    assert len(rows)==5100
    with (output/'manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # One original and its five transformations in each row; both real cameras/classes.
    sample = [next(r for r in originals if r['label']==str(y) and r['camera']==cam and r['path'].startswith('data/views-v2'))
              for y in (0,1) for cam in ('cam1','cam2')]
    sheet=Image.new('RGB',(1200,560),'#17202a');draw=ImageDraw.Draw(sheet)
    for j,row in enumerate(sample):
        with Image.open(ROOT/row['path']) as im:
            for k,(name,view) in enumerate([('original',im.copy()),*variants(im)]):
                sheet.paste(ImageOps.pad(view,(196,110)),(k*200,j*140))
                draw.text((k*200+2,j*140+112),name,fill='white')
                draw.text((k*200+2,j*140+124),row['scene_id']+' '+row['camera'],fill='white')
    sheet.save(reports/'augmentation-preview.jpg',quality=90)
    summary=dict(source_revision=REVISION,total_images=len(rows),original_images=len(originals),
        added_original_images=100,added_scenes=10,real_camera_count=2,synthetic_training_images=4000,
        train_images=4800,validation_original_images=100,test_original_images=200,
        per_class_total=2550,scene_split_overlap=0,parent_split_overlap=0,
        limitation='2D augmentation only; no new real camera viewpoints, environments, actors or self-striking examples.')
    (reports/'data_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()

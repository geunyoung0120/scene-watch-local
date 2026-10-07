"""Download candidate benign people images for human review before assigning labels."""
import collections
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import requests
from PIL import Image,ImageDraw,ImageOps
from scripts.prepare_data import download

DATA=ROOT/'data/people-v3'
REPORTS=ROOT/'reports/people-v3'
ARCHIVE_SHA='113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268'
S3='https://s3.amazonaws.com/images.cocodataset.org/'


def main():
    DATA.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True)
    archive=DATA/'annotations_trainval2017.zip'
    if not archive.exists():
        with requests.get(S3+'annotations/annotations_trainval2017.zip',stream=True,timeout=(20,90)) as r:
            r.raise_for_status()
            temporary=archive.with_suffix('.part')
            with temporary.open('wb') as f:
                for chunk in r.iter_content(1024*1024):f.write(chunk)
            temporary.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=ARCHIVE_SHA:
        raise ValueError('Annotation checksum mismatch')
    with zipfile.ZipFile(archive) as z:
        for name in ['instances_val2017.json','captions_val2017.json']:
            (DATA/name).write_bytes(z.read('annotations/'+name))
    annotations=json.loads((DATA/'instances_val2017.json').read_text())
    people=collections.defaultdict(list);captions=collections.defaultdict(list)
    for ann in annotations['annotations']:
        if ann['category_id']==1:people[ann['image_id']].append(ann)
    for ann in json.loads((DATA/'captions_val2017.json').read_text())['annotations']:
        captions[ann['image_id']].append(ann['caption'])
    licenses={l['id']:l for l in annotations['licenses']}
    groups=collections.defaultdict(list)
    for im in annotations['images']:
        if im['license'] not in [4,5]:continue  # CC BY / CC BY-SA 2.0, per image
        text=' '.join(captions[im['id']])
        if re.search(r'\b(fight\w*|punch\w*|wrestl\w*|boxer\w*|boxing|blood\w*|injur\w*|attack\w*|weapon\w*|riot\w*|gun\w*)\b',text,re.I):continue
        ps=people[im['id']]
        large=max((a['bbox'][2]*a['bbox'][3]/(im['width']*im['height']) for a in ps),default=0)
        category='crowd' if len(ps)>=5 else 'close' if large>=.2 else 'people' if ps else 'background'
        groups[category].append({**im,'category':category,'person_count':len(ps),
            'largest_person_box_fraction':large,'captions':captions[im['id']],
            'license_info':licenses[im['license']],
            'flickr_photo_page':'https://www.flickr.com/photo.gne?id='+im['flickr_url'].split('/')[-1].split('_')[0]})
    selected=[]
    rng=random.Random(20261008)
    for category,limit in [('crowd',160),('close',200),('people',120),('background',80)]:
        values=sorted(groups[category],key=lambda x:x['id']);rng.shuffle(values)
        selected.extend(values[:limit])
        print(category,len(values),'selected',min(limit,len(values)),flush=True)
    def fetch(im):
        target=DATA/'images'/im['file_name']
        download(S3+'val2017/'+im['file_name'],target)
        with Image.open(target) as image:
            image.verify()
        return {**im,'path':str(target.relative_to(ROOT)),
                'source_url':S3+'val2017/'+im['file_name'],
                'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    with ThreadPoolExecutor(max_workers=8) as pool:
        selected=list(pool.map(fetch,selected))
    (DATA/'candidates.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2))
    for start in range(0,len(selected),40):
        sheet=Image.new('RGB',(1280,1000),'#15202b');draw=ImageDraw.Draw(sheet)
        for k,item in enumerate(selected[start:start+40]):
            x,y=(k%5)*256,(k//5)*125
            with Image.open(ROOT/item['path']) as im:
                sheet.paste(ImageOps.contain(im.convert('RGB'),(252,101)),(x,y))
            draw.text((x+2,y+104),f"{start+k:03d} {item['id']} {item['category']}",fill='white')
        sheet.save(REPORTS/f'review-{start//40+1:02d}.jpg',quality=92)
    print('Candidates and contact sheets ready:',len(selected),flush=True)


if __name__=='__main__':main()

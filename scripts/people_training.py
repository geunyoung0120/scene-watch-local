"""Split and selection guards for the benign-person data expansion."""
import random


def validate_review(review,items):
    if set(review['reviewed_ids'])!={im['id'] for im in items}:
        raise ValueError('Every candidate must be reviewed')
    if review['image_sha256']!={str(im['id']):im['sha256'] for im in items}:
        raise ValueError('Candidate images changed since visual review')


def group_splits(items,seed=20261008):
    groups={}
    for item in items:groups.setdefault(item['group_id'],[]).append(item)
    by_category={}
    for key,values in groups.items():
        category=sorted(v['category'] for v in values)[0]
        by_category.setdefault(category,[]).append(key)
    assigned={};rng=random.Random(seed)
    for keys in by_category.values():
        keys.sort();rng.shuffle(keys)
        n=len(keys);train=int(n*.65);val=int(n*.15)
        for i,key in enumerate(keys):
            split='train' if i<train else 'validation' if i<train+val else 'test'
            for item in groups[key]:assigned[item['id']]=split
    return assigned


def candidate_allowed(baseline,candidate):
    return (candidate['recall']>=baseline['recall']-.05-1e-12 and
            candidate['fpr']<=baseline['fpr']+.02+1e-12 and
            candidate['new_fpr']<baseline['new_fpr'])

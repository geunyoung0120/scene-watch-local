"""Deterministic image-plane augmentation; these are not new 3D camera views."""
import cv2
import numpy as np
from PIL import Image, ImageOps


def should_augment(row):
    return row['split'] == 'train' and row.get('augmentation', 'original') == 'original'


def validate_groups(rows):
    for field in ('scene_id', 'parent_path'):
        groups = {}
        for row in rows:
            key = row.get(field) or row['path']
            groups.setdefault(key, set()).add(row['split'])
        if any(len(splits) != 1 for splits in groups.values()):
            raise ValueError(f'{field} crosses dataset splits')
    if len({r['path'] for r in rows}) != len(rows):
        raise ValueError('Duplicate image paths')


def variants(image):
    image = image.convert('RGB')
    yield 'mirror', ImageOps.mirror(image)
    pixels = np.asarray(image)
    h, w = pixels.shape[:2]
    for name, angle in [('roll_left', 8), ('roll_right', -8)]:
        matrix = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), angle, 1.0)
        warped = cv2.warpAffine(pixels, matrix, (w,h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
        yield name, Image.fromarray(warped)
    source = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    for name, destination in [
        ('perspective_left', [[0,.06*h],[w-1,0],[w-1,h-1],[0,.94*h-1]]),
        ('perspective_right', [[0,0],[w-1,.06*h],[w-1,.94*h-1],[0,h-1]])]:
        matrix = cv2.getPerspectiveTransform(source, np.float32(destination))
        warped = cv2.warpPerspective(pixels, matrix, (w,h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT_101)
        yield name, Image.fromarray(warped)

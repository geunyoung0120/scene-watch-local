import unittest
import numpy as np
from PIL import Image
from scripts import view_augmentation as views


class ViewTests(unittest.TestCase):
    def test_rejects_parent_or_scene_crossing_split(self):
        original = dict(path='a.jpg', parent_path='a.jpg', scene_id='scene-a', split='train')
        for other in [dict(path='b.jpg', parent_path='a.jpg', scene_id='scene-b', split='test'),
                      dict(path='b.jpg', parent_path='b.jpg', scene_id='scene-a', split='validation')]:
            with self.assertRaises(ValueError):
                views.validate_groups([original, other])

    def test_transformations_keep_dimensions_and_mirror_correctly(self):
        pixels = np.zeros((48, 64, 3), dtype=np.uint8)
        pixels[:, :20] = [255, 50, 10]
        im = Image.fromarray(pixels)
        variants = dict(views.variants(im))
        self.assertEqual(len(variants), 5)
        for value in variants.values():
            self.assertEqual(value.size, im.size)
        np.testing.assert_array_equal(np.asarray(variants['mirror']), pixels[:, ::-1])
        for name, value in variants.items():
            self.assertFalse(np.array_equal(np.asarray(value), pixels), name)

    def test_only_training_rows_are_augmented(self):
        rows = [dict(split=s, augmentation=a) for s in ['train', 'validation', 'test']
                for a in ['original', 'mirror']]
        selected = [r for r in rows if views.should_augment(r)]
        self.assertEqual(selected, [dict(split='train', augmentation='original')])

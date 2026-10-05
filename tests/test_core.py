import math
import unittest

from violence_app.core import ProbabilityWindow, make_scene_splits, benchmark_interval


class WindowTests(unittest.TestCase):
    def test_waits_for_five_then_discards_oldest(self):
        w = ProbabilityWindow()
        for i, p in enumerate([0.1, 0.2, 0.3, 0.4]):
            self.assertIsNone(w.add(p, float(i)))
        self.assertAlmostEqual(w.add(0.5, 4.0), 0.3)
        self.assertAlmostEqual(w.add(1.0, 5.0), 0.48)

    def test_gap_resets_history_and_reset_clears_time(self):
        w = ProbabilityWindow(max_gap=2)
        for i in range(5):
            w.add(1, i / 10)
        self.assertIsNone(w.add(0, 5))
        self.assertEqual(w.count, 1)
        w.reset()
        self.assertIsNone(w.add(0, 0))

    def test_invalid_probabilities_and_out_of_order_do_not_change_window(self):
        w = ProbabilityWindow()
        w.add(0.4, 10)
        for value in [math.nan, math.inf, -0.1, 1.1]:
            with self.assertRaises(ValueError):
                w.add(value, 11)
        for time in [math.nan, math.inf, 9, 10]:
            with self.assertRaises(ValueError):
                w.add(0.4, time)
        self.assertEqual(w.count, 1)


class DatasetTests(unittest.TestCase):
    def test_balanced_scenes_are_disjoint_and_reproducible(self):
        scenes = {0: list(range(1, 51)), 1: list(range(1, 51))}
        split = make_scene_splits(scenes, seed=42)
        self.assertEqual(split, make_scene_splits(scenes, seed=42))
        for label in [0, 1]:
            self.assertEqual(sum(v == 'train' for (y, s), v in split.items() if y == label), 35)
            self.assertEqual(sum(v == 'validation' for (y, s), v in split.items() if y == label), 5)
            self.assertEqual(sum(v == 'test' for (y, s), v in split.items() if y == label), 10)
        self.assertEqual(len(split), 100)

    def test_interval_leaves_headroom_and_rounds_up(self):
        self.assertEqual(benchmark_interval(82), 110)
        self.assertEqual(benchmark_interval(221), 280)
        self.assertEqual(benchmark_interval(12), 100)
        with self.assertRaises(ValueError):
            benchmark_interval(float('nan'))


if __name__ == '__main__':
    unittest.main()

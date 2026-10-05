from collections import deque
import math
import random


class ProbabilityWindow:
    def __init__(self, size=5, max_gap=5.0):
        if size < 1 or max_gap <= 0:
            raise ValueError('Invalid window configuration')
        self.values = deque(maxlen=size)
        self.max_gap = max_gap
        self.last_time = None

    @property
    def count(self):
        return len(self.values)

    def reset(self):
        self.values.clear()
        self.last_time = None

    def add(self, probability, timestamp):
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Probability must be finite and in [0, 1]')
        if not math.isfinite(timestamp):
            raise ValueError('Timestamp must be finite')
        if self.last_time is not None:
            if timestamp <= self.last_time:
                raise ValueError('Frames must be strictly ordered')
            if timestamp - self.last_time > self.max_gap:
                self.reset()
        self.values.append(float(probability))
        self.last_time = timestamp
        return sum(self.values) / len(self.values) if len(self.values) == self.values.maxlen else None


def make_scene_splits(scenes, seed=42):
    rng = random.Random(seed)
    result = {}
    for label in sorted(scenes):
        ids = sorted(set(scenes[label]))
        if len(ids) != 50:
            raise ValueError('Expected 50 distinct scenes per class')
        rng.shuffle(ids)
        for i, scene in enumerate(ids):
            result[label, scene] = 'train' if i < 35 else 'validation' if i < 40 else 'test'
    return result


def benchmark_interval(p95_ms):
    if not math.isfinite(p95_ms) or p95_ms <= 0:
        raise ValueError('Latency must be positive and finite')
    # 25% headroom; cap analysis frequency at 10 Hz for a small webcam prototype.
    return max(100, math.ceil(p95_ms * 1.25 / 10) * 10)

"""Shared training/live preprocessing and safe, portable logistic head."""
import io
import json
import os
from pathlib import Path
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
from PIL import Image, ImageOps
import torch

ROOT = Path(__file__).resolve().parents[1]
PREPROCESS_VERSION = 'rgb-letterbox-224-imagenet-v1'
MEAN = np.array([.485, .456, .406], dtype=np.float32)
STD = np.array([.229, .224, .225], dtype=np.float32)


def preprocess(image):
    image = ImageOps.exif_transpose(image).convert('RGB')
    fitted = ImageOps.contain(image, (224, 224), Image.Resampling.BICUBIC)
    canvas = Image.new('RGB', (224, 224), tuple(int(round(x * 255)) for x in MEAN))
    canvas.paste(fitted, ((224 - fitted.width) // 2, (224 - fitted.height) // 2))
    pixels = (np.asarray(canvas, dtype=np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy())


class FeatureExtractor:
    def __init__(self, device='auto'):
        from transformers import Dinov2Model
        torch.set_num_threads(4)
        if device == 'auto':
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = device
        self.model = Dinov2Model.from_pretrained(
            ROOT / 'models/dinov2-small', local_files_only=True,
            use_safetensors=True, attn_implementation='sdpa').eval().to(device)
        self.model.requires_grad_(False)

    def sync(self):
        if self.device == 'mps':
            torch.mps.synchronize()

    def features(self, images):
        batch = torch.stack([preprocess(im) for im in images]).to(self.device)
        with torch.inference_mode():
            cls = self.model(pixel_values=batch).last_hidden_state[:, 0, :]
        return cls.cpu().numpy()


def save_head(path, scaler, classifier, metadata):
    if list(classifier.classes_) != [0, 1]:
        raise ValueError('Expected non-violent=0, violent=1')
    data = dict(metadata=metadata, classes=[0, 1], mean=scaler.mean_.tolist(),
                scale=scaler.scale_.tolist(), coef=classifier.coef_[0].tolist(),
                intercept=float(classifier.intercept_[0]))
    path.write_text(json.dumps(data, indent=2))


class LinearHead:
    def __init__(self, path):
        data = json.loads(Path(path).read_text())
        self.metadata = data['metadata']
        if data['classes'] != [0, 1]:
            raise ValueError('Classifier labels do not match this application')
        self.mean, self.scale, self.coef = [np.array(data[k], dtype=np.float64) for k in ('mean','scale','coef')]
        self.intercept = data['intercept']

    def predict(self, features):
        z = ((np.asarray(features, dtype=np.float64) - self.mean) / self.scale) @ self.coef + self.intercept
        # Stable sigmoid, including extreme logits.
        return np.exp(-np.logaddexp(0, -z))


class InferenceEngine:
    def __init__(self, device='auto'):
        started = time.perf_counter()
        self.extractor = FeatureExtractor(device)
        self.head = LinearHead(ROOT / 'models/violence_head.json')
        if self.head.metadata['preprocess_version'] != PREPROCESS_VERSION:
            raise ValueError('Model preprocessing version mismatch; retrain classifier')
        self.device = self.extractor.device
        self.startup_ms = (time.perf_counter() - started) * 1000

    def predict_jpeg(self, payload):
        started = time.perf_counter()
        with Image.open(io.BytesIO(payload)) as image:
            if image.width * image.height > 4096 * 4096:
                raise ValueError('Image is too large')
            image.load()
            decoded = time.perf_counter()
            feature = self.extractor.features([image])
        embedded = time.perf_counter()
        probability = float(self.head.predict(feature)[0])
        ended = time.perf_counter()
        return probability, {'decode_ms':(decoded-started)*1000,
            'embedding_ms':(embedded-decoded)*1000, 'classifier_ms':(ended-embedded)*1000,
            'inference_ms':(ended-started)*1000}

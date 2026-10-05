import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from violence_app.model import LinearHead, preprocess, save_head


class ModelTests(unittest.TestCase):
    def test_export_matches_sklearn_and_positive_class(self):
        x = np.random.default_rng(5).normal(size=(80, 12))
        y = (x[:, 0] + x[:, 1] > 0).astype(int)
        scaler = StandardScaler().fit(x)
        lr = LogisticRegression(C=.1).fit(scaler.transform(x), y)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'head.json'
            save_head(path, scaler, lr, {'feature_dim':12})
            head = LinearHead(path)
            np.testing.assert_allclose(head.predict(x), lr.predict_proba(scaler.transform(x))[:, 1], atol=1e-8)

    def test_preprocess_preserves_edges_and_converts_rgb(self):
        im = Image.new('RGB', (640, 360), 'red')
        tensor = preprocess(im)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertTrue(np.isfinite(tensor.numpy()).all())
        # Both horizontal edges survive full-frame letterboxing.
        self.assertGreater(tensor[0, 112, 0], 1)
        self.assertGreater(tensor[0, 112, -1], 1)
        self.assertEqual(tuple(preprocess(Image.new('L', (100,100))).shape), (3,224,224))


if __name__ == '__main__':
    unittest.main()

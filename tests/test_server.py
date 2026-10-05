import http.client
import json
import threading
import unittest
from violence_app.server import create_server


class FakeEngine:
    device = 'test'
    startup_ms = 0
    def predict_jpeg(self, data):
        if data != b'jpeg-test':
            raise ValueError('Invalid image')
        return .8, {'inference_ms':1.0}


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server(FakeEngine(), port=0, token='test-token', interval_ms=100)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, path, body=b'', headers=None, method='POST'):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        h = {'X-App-Token':'test-token', **(headers or {})}
        connection.request(method, path, body, h)
        response = connection.getresponse()
        code, payload = response.status, json.loads(response.read())
        connection.close()
        return code, payload

    def test_foreign_origin_and_missing_token_rejected(self):
        self.assertEqual(self.request('/api/start', headers={'X-App-Token':''})[0],403)
        self.assertEqual(self.request('/api/start', headers={'Origin':'https://example.com'})[0],403)

    def test_five_frames_reset_and_old_session_rejection(self):
        _, start = self.request('/api/start')
        session = start['session_id']
        headers = {'X-Session-Id':session,'Content-Type':'image/jpeg'}
        for i in range(1,6):
            code,result = self.request('/api/predict',b'jpeg-test',{**headers,'X-Frame-Id':str(i)})
            self.assertEqual(code,200)
            if i < 5:
                self.assertIsNone(result['mean_probability'])
            else:
                self.assertAlmostEqual(result['mean_probability'],.8)
        self.assertEqual(self.request('/api/predict',b'jpeg-test',{**headers,'X-Frame-Id':'5'})[0],409)
        self.assertEqual(self.request('/api/stop',headers=headers)[0],200)
        self.assertEqual(self.request('/api/predict',b'jpeg-test',{**headers,'X-Frame-Id':'6'})[0],409)
        _, next_session = self.request('/api/start')
        code,result=self.request('/api/predict',b'jpeg-test',{**headers,'X-Session-Id':next_session['session_id'],'X-Frame-Id':'1'})
        self.assertEqual(result['window_count'],1)
        self.assertIsNone(result['mean_probability'])

    def test_corrupt_frame_and_oversized_body_rejected(self):
        _, start=self.request('/api/start')
        headers={'X-Session-Id':start['session_id'],'X-Frame-Id':'1','Content-Type':'image/jpeg'}
        self.assertEqual(self.request('/api/predict',b'bad',headers)[0],400)
        self.assertEqual(self.request('/api/predict',b'',{**headers,'Content-Length':'5000000'})[0],413)


if __name__ == '__main__':
    unittest.main()

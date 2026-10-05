"""Loopback-only web server. Camera frames are processed in memory, never saved."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlsplit

from .core import ProbabilityWindow

ROOT = Path(__file__).resolve().parents[1]


class AppState:
    def __init__(self, engine, token, interval_ms):
        self.engine, self.token, self.interval_ms = engine, token, interval_ms
        self.lock = threading.Lock()
        self.session = None
        self.sequence = 0
        self.window = ProbabilityWindow(max_gap=max(5.0, interval_ms / 1000 * 4))
        self.timestamps = []
        self.request_count = 0
        self.latency_total = 0

    def reset(self):
        self.window.reset()
        self.sequence = 0
        self.timestamps = []
        self.request_count = 0
        self.latency_total = 0


def create_server(engine, port=8876, token=None, interval_ms=100):
    state = AppState(engine, token or secrets.token_urlsafe(32), interval_ms)

    class Handler(BaseHTTPRequestHandler):
        server_version = 'LocalVision/1.0'

        def log_message(self, message, *args):
            # No image bodies, tokens, or frame content in logs.
            if args and '/api/predict' in str(args[0]):
                return
            super().log_message(message, *args)

        def allowed_host(self):
            port = self.server.server_address[1]
            return self.headers.get('Host') in {f'127.0.0.1:{port}', f'localhost:{port}'}

        def send_bytes(self, code, body, content_type):
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Referrer-Policy','no-referrer')
            self.send_header('Permissions-Policy','camera=(self), microphone=()')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def json(self, code, body):
            self.send_bytes(code,json.dumps(body,allow_nan=False).encode(),'application/json; charset=utf-8')

        def do_GET(self):
            if not self.allowed_host():
                return self.json(403,{'error':'허용되지 않은 호스트입니다.'})
            path=urlsplit(self.path).path
            if path=='/api/status':
                benchmark_path=ROOT/'reports/benchmark.json'
                benchmark=json.loads(benchmark_path.read_text()) if benchmark_path.exists() else {}
                training_path=ROOT/'reports/training.json'
                training=json.loads(training_path.read_text()) if training_path.exists() else {}
                return self.json(200,{'app_id':'scene-watch-local-v1','ready':True,'device':state.engine.device,'interval_ms':state.interval_ms,
                    'window_size':5,'model':'DINOv2 Small + Logistic Regression',
                    'benchmark':{k:v for k,v in benchmark.items() if k!='samples'},
                    'evaluation':training.get('test_five_frame_clip_mean',{}),
                    'dataset_images':training.get('metadata',{}).get('dataset_images',1000),
                    'scope':'폭행·비폭행 분류 실험'})
            routes={'/':('index.html','text/html; charset=utf-8'),
                    '/app.js':('app.js','text/javascript; charset=utf-8'),
                    '/style.css':('style.css','text/css; charset=utf-8')}
            if path not in routes:
                return self.json(404,{'error':'페이지를 찾을 수 없습니다.'})
            filename,content_type=routes[path]
            data=(ROOT/'web'/filename).read_bytes()
            if path=='/':
                data=data.replace(b'__APP_TOKEN__',state.token.encode())
            self.send_bytes(200,data,content_type)

        def authorized(self):
            if not self.allowed_host():
                return False
            if not secrets.compare_digest(self.headers.get('X-App-Token',''),state.token):
                return False
            origin=self.headers.get('Origin')
            port=self.server.server_address[1]
            return origin is None or origin in {f'http://127.0.0.1:{port}',f'http://localhost:{port}'}

        def do_POST(self):
            if not self.authorized():
                return self.json(403,{'error':'이 페이지에서 보낸 요청만 허용됩니다. 새로고침해주세요.'})
            try:
                length=int(self.headers.get('Content-Length','0'))
            except ValueError:
                return self.json(400,{'error':'잘못된 요청 길이입니다.'})
            if length < 0 or length > 4_000_000:
                self.close_connection=True
                return self.json(413,{'error':'사진은 4MB 이하여야 합니다.'})
            self.connection.settimeout(15)
            try:
                payload=self.rfile.read(length)
            except TimeoutError:
                return self.json(408,{'error':'사진 수신 시간이 초과되었습니다.'})
            path=urlsplit(self.path).path
            with state.lock:
                if path=='/api/start':
                    state.reset()
                    state.session=secrets.token_urlsafe(16)
                    return self.json(200,{'session_id':state.session,'interval_ms':state.interval_ms})
                session=self.headers.get('X-Session-Id')
                if state.session is None or session != state.session:
                    return self.json(409,{'error':'종료된 분석 세션입니다. 다시 시작해주세요.'})
                if path=='/api/stop':
                    count=state.request_count
                    mean_ms=state.latency_total/count if count else None
                    state.session=None
                    state.reset()
                    return self.json(200,{'stopped':True,'analyses':count,'mean_inference_ms':mean_ms})
                if path!='/api/predict':
                    return self.json(404,{'error':'요청 경로를 찾을 수 없습니다.'})
                if self.headers.get('Content-Type','').split(';')[0]!='image/jpeg':
                    return self.json(415,{'error':'JPEG 사진을 보내주세요.'})
                try:
                    sequence=int(self.headers.get('X-Frame-Id','0'))
                except ValueError:
                    return self.json(400,{'error':'잘못된 사진 번호입니다.'})
                if sequence <= state.sequence:
                    return self.json(409,{'error':'이미 처리한 사진입니다.'})
                try:
                    probability,timing=state.engine.predict_jpeg(payload)
                    now=time.perf_counter()
                    mean=state.window.add(probability,now)
                except (ValueError,OSError) as error:
                    return self.json(400,{'error':'사진을 읽을 수 없습니다. 카메라를 다시 시작해주세요.'})
                except Exception:
                    # Do not turn failed inference into a zero-risk result.
                    state.reset()
                    return self.json(500,{'error':'분석 오류가 발생했습니다. 잠시 후 다시 시작해주세요.'})
                state.sequence=sequence
                state.timestamps.append(now)
                state.timestamps=state.timestamps[-state.window.count:]
                state.request_count+=1
                state.latency_total+=timing['inference_ms']
                return self.json(200,{'probability':probability,'mean_probability':mean,
                    'window_count':state.window.count,'window_size':5,'frame_id':sequence,
                    'window_span_ms':(state.timestamps[-1]-state.timestamps[0])*1000,
                    'timing':timing,'interval_ms':state.interval_ms})

    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.daemon_threads=True
    server.app_state=state
    return server


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8876)
    args=parser.parse_args()
    from .model import InferenceEngine
    benchmark=json.loads((ROOT/'reports/benchmark.json').read_text())
    engine=InferenceEngine()
    # Warm the model without using a camera, so the first live sample is responsive.
    sample=next((ROOT/'data/images/non-violent').glob('*.jpg'))
    for _ in range(3):
        engine.predict_jpeg(sample.read_bytes())
    server=create_server(engine,args.port,interval_ms=benchmark['recommended_interval_ms'])
    print(f'Ready: http://127.0.0.1:{args.port} | {engine.device} | interval={server.app_state.interval_ms}ms',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__=='__main__':
    main()

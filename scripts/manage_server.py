"""Start/stop only this project's local server, with a persistent PID record."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
PID=ROOT/'reports/server.pid'
URL='http://127.0.0.1:8876'


def ready():
    try:
        with urlopen(URL+'/api/status',timeout=1) as r:
            return json.load(r).get('app_id')=='scene-watch-local-v1'
    except Exception:
        return False


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['start','stop','status'])
    parser.add_argument('--open',action='store_true')
    args=parser.parse_args()
    if args.action=='status':
        print(URL if ready() else '서버가 실행 중이 아닙니다.'); return
    if args.action=='stop':
        if not PID.exists():
            print('이 프로젝트에서 실행한 서버 PID가 없습니다.'); return
        pid=int(PID.read_text().strip())
        result=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
        if str(ROOT/'scripts/serve.py') not in result.stdout:
            print('기록된 서버 프로세스가 없습니다.'); PID.unlink(); return
        os.kill(pid,signal.SIGTERM)
        PID.unlink()
        print('로컬 서버를 종료했습니다.'); return
    if not ready():
        log=(ROOT/'reports/server.log').open('ab')
        process=subprocess.Popen([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/serve.py')],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
            start_new_session=True,close_fds=True)
        log.close()
        PID.write_text(str(process.pid))
        for _ in range(100):
            if ready(): break
            if process.poll() is not None:
                raise SystemExit('서버 시작 실패. reports/server.log를 확인해주세요.')
            time.sleep(.2)
        else:
            raise SystemExit('모델 준비에 시간이 걸립니다. reports/server.log를 확인해주세요.')
    print('서버 실행 중:',URL)
    if args.open:
        subprocess.run(['open',URL],check=True)


if __name__=='__main__':
    main()

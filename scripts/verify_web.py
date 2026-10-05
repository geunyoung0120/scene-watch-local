"""Browser end-to-end verification with a prerecorded, synthetic camera device.

Never opens the laptop's physical camera. Screenshots contain AIRTLab sample data.
"""
import csv
import io
import json
from pathlib import Path
import re
import tempfile
import time

import cv2
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright, expect
import requests

ROOT=Path(__file__).resolve().parents[1]
URL='http://127.0.0.1:8876'


def main():
    rows=list(csv.DictReader((ROOT/'data/manifest.csv').open()))
    sample=next(r for r in rows if r['split']=='test' and r['label']=='0')
    shots=ROOT/'reports/web_checks'
    shots.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='scene-watch-web-') as tmp:
        fake_video=Path(tmp)/'camera.y4m'
        cap=cv2.VideoCapture(str(ROOT/sample['video_path']))
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with fake_video.open('wb') as f:
            f.write(b'YUV4MPEG2 W640 H360 F10:1 Ip A1:1 C420jpeg\n')
            for index in np.linspace(0,frames-1,60).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(index))
                ok,frame=cap.read()
                assert ok
                frame=cv2.resize(frame,(640,360))
                f.write(b'FRAME\n')
                f.write(cv2.cvtColor(frame,cv2.COLOR_BGR2YUV_I420).tobytes())
        cap.release()
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True,args=[
                '--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream',
                f'--use-file-for-fake-video-capture={fake_video}'])
            context=browser.new_context(viewport={'width':1360,'height':1080},permissions=['camera'])
            page=context.new_page()
            errors=[]
            page.on('pageerror',lambda error:errors.append(str(error)))
            results=[]
            def response_received(response):
                if response.url.endswith('/api/predict') and response.status==200:
                    results.append(response.json())
            page.on('response',response_received)
            page.goto(URL)
            page.get_by_text('모델 준비 완료',exact=True).wait_for()
            assert page.get_by_role('button',name='카메라 시작').is_enabled()
            assert page.locator('#placeholder').is_visible()
            assert page.locator('#video').evaluate('(v)=>v.srcObject===null')
            page.screenshot(path=str(shots/'desktop-ready.png'),full_page=True)
            page.get_by_role('button',name='카메라 시작').click()
            expect(page.locator('#risk-value')).not_to_have_text('—')
            expect(page.locator('#history-status')).to_have_text(re.compile(r'(1[2-9]|[2-9]\d|\d{3,})장 분석 완료'))
            assert page.locator('#video').is_visible()
            assert page.get_by_role('button',name='중지',exact=True).is_enabled()
            assert len(results)>=12
            for i,result in enumerate(results):
                assert result['window_count']==min(i+1,5)
                if i<4:
                    assert result['mean_probability'] is None
                else:
                    expected=sum(r['probability'] for r in results[i-4:i+1])/5
                    assert abs(expected-result['mean_probability'])<1e-9
            page.screenshot(path=str(shots/'desktop-file-camera.png'),full_page=True)
            observed_interval=page.locator('#interval').inner_text()
            page.get_by_role('button',name='중지',exact=True).click()
            assert page.locator('#placeholder').is_visible()
            assert page.locator('#risk-value').inner_text()=='—'
            assert page.locator('#video').evaluate('(v)=>v.srcObject===null')
            first_run_count=len(results)
            page.get_by_role('button',name='카메라 시작').click()
            expect(page.locator('#risk-value')).not_to_have_text('—')
            assert results[first_run_count]['window_count']==1
            page.get_by_role('button',name='중지',exact=True).click()
            page.set_viewport_size({'width':390,'height':844})
            page.screenshot(path=str(shots/'mobile-ready.png'),full_page=True)
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            # Error handling simulation in the test page only.
            page.evaluate("() => { navigator.mediaDevices.getUserMedia=async()=>{throw new DOMException('test permission denial','NotAllowedError')}; }")
            page.get_by_role('button',name='카메라 시작').click()
            page.get_by_role('alert').filter(has_text='카메라 권한이 필요합니다').wait_for()
            assert page.get_by_role('button',name='카메라 시작').is_enabled()
            assert not errors, errors
            context.close()
            browser.close()

    # Real HTTP inference timing, including local request/response transport.
    html=requests.get(URL,timeout=10).text
    token=re.search(r'name="app-token" content="([^"]+)"',html).group(1)
    headers={'X-App-Token':token}
    session=requests.post(URL+'/api/start',headers=headers,timeout=10).json()['session_id']
    headers.update({'X-Session-Id':session,'Content-Type':'image/jpeg'})
    timings=[]
    for index,row in enumerate([r for r in rows if r['split']=='test'][::5][:30],1):
        with Image.open(ROOT/row['path']) as im:
            im.thumbnail((640,480))
            out=io.BytesIO();im.save(out,format='JPEG',quality=85)
        start=time.perf_counter()
        result=requests.post(URL+'/api/predict',data=out.getvalue(),headers={**headers,'X-Frame-Id':str(index)},timeout=10)
        result.raise_for_status()
        timings.append((time.perf_counter()-start)*1000)
    requests.post(URL+'/api/stop',headers=headers,timeout=10).raise_for_status()
    report={'browser':'Chromium headless; prerecorded AIRTLab file as synthetic camera',
        'physical_camera_used':False,'passed':['ready screen','camera preview','five-frame exact arithmetic',
        'start/stop','restart resets history','permission error','mobile layout','no browser exceptions'],
        'browser_first_run_analyses':first_run_count,'observed_browser_interval':observed_interval,
        'http_samples':len(timings),'http_p50_ms':float(np.percentile(timings,50)),
        'http_p95_ms':float(np.percentile(timings,95)),
        'http_timings_ms':timings,'physical_camera_permission':'left for user on Start'}
    (ROOT/'reports/web_verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in report.items() if k!='http_timings_ms'},indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()

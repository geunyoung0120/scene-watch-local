'use strict';
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="app-token"]').content;
const video = $('video');
const capture = $('capture');
const ctx = capture.getContext('2d');
const historyCanvas = $('history');
let stream = null, session = null, running = false, starting = false;
let generation = 0, sequence = 0, intervalMs = 100, actualInterval = 100;
let timeout = null, clock = null, controller = null, startedAt = 0;
let history = [], lastCapture = null, status = null;

function message(text) { $('message').textContent = text; $('message').hidden = !text; }
function stat(id, value, unit) {
  $(id).replaceChildren(document.createTextNode(value));
  const small = document.createElement('small'); small.textContent = unit; $(id).append(small);
}
async function api(path, options={}) {
  const response = await fetch(path, {...options, headers: {'X-App-Token': token, ...(options.headers || {})}});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `서버 오류 (${response.status})`);
  return data;
}
function drawHistory() {
  const bounds = historyCanvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  historyCanvas.width = Math.round(bounds.width * ratio); historyCanvas.height = Math.round(bounds.height * ratio);
  const c = historyCanvas.getContext('2d'); c.scale(ratio, ratio);
  const w = bounds.width, h = bounds.height;
  c.strokeStyle = '#edf0f2'; c.lineWidth = 1; c.setLineDash([3,4]);
  for (const y of [1,h/2,h-1]) { c.beginPath(); c.moveTo(0,y); c.lineTo(w,y); c.stroke(); }
  c.setLineDash([]);
  if (history.length < 2) return;
  const points = history.map((p,i) => [i/(history.length-1)*w, 3+(1-p)*(h-6)]);
  const fill = c.createLinearGradient(0,0,0,h); fill.addColorStop(0,'#2a99872c'); fill.addColorStop(1,'#2a998702');
  c.beginPath(); c.moveTo(points[0][0],h); points.forEach(([x,y])=>c.lineTo(x,y)); c.lineTo(w,h); c.closePath(); c.fillStyle=fill; c.fill();
  c.beginPath(); points.forEach(([x,y],i)=>i ? c.lineTo(x,y):c.moveTo(x,y)); c.strokeStyle='#3d9885'; c.lineWidth=2; c.stroke();
}
function resetDisplay() {
  $('risk-value').textContent='—'; $('risk-state').textContent='분석 대기'; $('risk-state').className='risk-state';
  $('risk-description').textContent='카메라를 시작하면 분석 결과가 표시됩니다.';
  $('risk-meter-fill').style.width='0%'; document.documentElement.style.setProperty('--score','#147d70');
  $('window-count').textContent='0 / 5'; $('window-dots').setAttribute('aria-label','0 / 5장');
  [...$('window-dots').querySelectorAll('i')].forEach(dot=>dot.classList.remove('filled'));
  stat('instant','—','%'); stat('latency','—','ms');
  $('window-span').textContent='5장의 분석이 완료되면 평균을 표시합니다';
  $('history-status').textContent='결과를 기다리는 중';
  history=[]; lastCapture=null; drawHistory();
}
function showResult(data, roundTrip) {
  stat('instant',(data.probability*100).toFixed(1),'%');
  stat('latency',Math.round(data.timing.inference_ms),'ms');
  stat('interval',Math.round(actualInterval),'ms');
  $('latency').title=`사진 캡처부터 응답까지 ${Math.round(roundTrip)}ms`;
  $('window-count').textContent=`${data.window_count} / 5`;
  $('window-dots').setAttribute('aria-label',`${data.window_count} / 5장`);
  [...$('window-dots').querySelectorAll('i')].forEach((dot,i)=>dot.classList.toggle('filled',i<data.window_count));
  if (data.mean_probability === null) {
    $('risk-value').textContent='—'; $('risk-state').textContent='사진 수집 중';
    $('risk-state').className='risk-state'; $('risk-meter-fill').style.width='0%';
    $('risk-description').textContent=`평균을 위해 ${5-data.window_count}장을 더 분석합니다.`;
    return;
  }
  const p=data.mean_probability;
  $('risk-value').textContent=(p*100).toFixed(1);
  $('risk-state').textContent=p>=.5 ? '폭행 의심 점수 높음' : '폭행 의심 점수 낮음';
  $('risk-state').className='risk-state ready';
  $('risk-state').classList.toggle('elevated',p>=.5);
  document.documentElement.style.setProperty('--score',p>=.5?'#bd6c44':'#147d70');
  $('risk-description').textContent='최근 분석한 5장의 분류 확률을 평균한 값입니다.';
  $('risk-meter-fill').style.width=`${p*100}%`;
  $('window-span').textContent=`현재 5장이 포괄하는 시간 ${(data.window_span_ms/1000).toFixed(2)}초`;
  $('history-status').textContent=`${sequence}장 분석 완료`;
  history.push(p); if (history.length>80) history.shift(); drawHistory();
}
async function analyse(myGeneration) {
  if (!running || generation!==myGeneration) return;
  const begin=performance.now();
  try {
    if (video.readyState<2 || !video.videoWidth) throw new Error('카메라 영상을 읽을 수 없습니다. 다시 시작해주세요.');
    const scale=Math.min(640/video.videoWidth,480/video.videoHeight,1);
    capture.width=Math.round(video.videoWidth*scale); capture.height=Math.round(video.videoHeight*scale);
    ctx.drawImage(video,0,0,capture.width,capture.height);
    const blob=await new Promise(resolve=>capture.toBlob(resolve,'image/jpeg',.85));
    if (!running || generation!==myGeneration) return;
    if (!blob) throw new Error('카메라 사진 변환에 실패했습니다.');
    if (lastCapture!==null) actualInterval=begin-lastCapture;
    lastCapture=begin;
    controller=new AbortController();
    const abortTimer=setTimeout(()=>controller?.abort(),15000);
    let data;
    try {
      data=await api('/api/predict',{method:'POST',body:blob,signal:controller.signal,
        headers:{'Content-Type':'image/jpeg','X-Session-Id':session,'X-Frame-Id':String(++sequence)}});
    } finally {clearTimeout(abortTimer);}
    if (!running || generation!==myGeneration) return;
    showResult(data,performance.now()-begin);
    const wait=Math.max(0,intervalMs-(performance.now()-begin));
    timeout=setTimeout(()=>analyse(myGeneration),wait);
  } catch(error) {
    if (generation!==myGeneration) return;
    await stopCamera();
    message(error.name==='AbortError'?'분석 응답이 지연되어 중지했습니다. 다시 시작해주세요.':error.message);
  }
}
function cameraError(error) {
  if (error.name==='NotAllowedError') return '카메라 권한이 필요합니다. 주소창의 카메라 설정과 macOS 시스템 설정 → 개인정보 보호 및 보안 → 카메라에서 이 브라우저를 허용해주세요.';
  if (error.name==='NotFoundError') return '연결된 카메라를 찾을 수 없습니다. 카메라 연결을 확인해주세요.';
  if (error.name==='NotReadableError') return '카메라를 사용할 수 없습니다. 다른 영상 통화 앱을 종료한 뒤 다시 시도해주세요.';
  return error.message || '카메라를 시작하지 못했습니다.';
}
async function startCamera() {
  if (running || starting) return;
  starting=true; const myGeneration=++generation;
  $('start').disabled=true; $('stop').disabled=false; message(''); resetDisplay();
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('이 브라우저에서 카메라를 지원하지 않습니다. localhost 주소를 Chrome 또는 Safari로 열어주세요.');
    const acquired=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720},facingMode:'user'},audio:false});
    if (myGeneration!==generation) {acquired.getTracks().forEach(t=>t.stop()); return;}
    stream=acquired; video.srcObject=stream;
    await video.play();
    if(myGeneration!==generation) return;
    const result=await api('/api/start',{method:'POST'});
    if (myGeneration!==generation) {
      await api('/api/stop',{method:'POST',headers:{'X-Session-Id':result.session_id}}).catch(()=>{}); return;
    }
    session=result.session_id; intervalMs=result.interval_ms; actualInterval=intervalMs; sequence=0;
    running=true; startedAt=performance.now();
    stream.getVideoTracks()[0].addEventListener('ended',()=>{if(running){stopCamera();message('카메라 연결이 종료되었습니다. 다시 시작해주세요.');}});
    video.classList.add('active'); video.classList.toggle('mirrored',$('mirror').checked);
    $('placeholder').hidden=true; $('video-top').hidden=false; $('video-bottom').hidden=false;
    $('source-status').textContent='실시간 분석 중'; $('source-status').className='source-status live';
    $('resolution').textContent=`${video.videoWidth} × ${video.videoHeight}`;
    $('camera-name').textContent=stream.getVideoTracks()[0].label || '노트북 카메라';
    $('session-clock').textContent='00:00';
    clock=setInterval(()=>{const s=Math.floor((performance.now()-startedAt)/1000);$('session-clock').textContent=`${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;},1000);
    analyse(myGeneration);
  } catch(error) {
    if(myGeneration===generation) {await stopCamera();message(cameraError(error));}
  } finally {
    // A stopped startup must settle (including stale-session cleanup) before
    // another start is allowed to reach the server.
    starting=false; $('start').disabled=running || !status; $('stop').disabled=!running;
  }
}
async function stopCamera() {
  generation++; running=false;
  clearTimeout(timeout); clearInterval(clock); controller?.abort(); controller=null;
  if(stream) stream.getTracks().forEach(t=>t.stop()); stream=null;
  video.srcObject=null; video.classList.remove('active');
  $('placeholder').hidden=false; $('video-top').hidden=true; $('video-bottom').hidden=true;
  $('source-status').textContent='대기 중'; $('source-status').className='source-status';
  $('start').disabled=starting || !status; $('stop').disabled=true; resetDisplay();
  const old=session; session=null;
  if(old) await api('/api/stop',{method:'POST',headers:{'X-Session-Id':old}}).catch(()=>{});
}
$('start').addEventListener('click',startCamera);
$('stop').addEventListener('click',()=>stopCamera());
$('mirror').addEventListener('change',()=>video.classList.toggle('mirrored',$('mirror').checked));
window.addEventListener('resize',drawHistory);
window.addEventListener('pagehide',()=>{stream?.getTracks().forEach(t=>t.stop());if(session)fetch('/api/stop',{method:'POST',keepalive:true,headers:{'X-App-Token':token,'X-Session-Id':session}}).catch(()=>{});});
document.addEventListener('visibilitychange',()=>{if(document.hidden && (running || starting)){stopCamera();message('페이지를 벗어나 분석을 중지했습니다. 카메라를 다시 시작해주세요.');}});
async function initialize() {
  try {
    status=await api('/api/status'); intervalMs=status.interval_ms; actualInterval=intervalMs;
    $('connection').textContent='모델 준비 완료'; $('connection-dot').classList.add('connected');
    $('start').disabled=false; stat('interval',intervalMs,'ms');
    const t=status.benchmark.timing_ms?.pipeline_ms;
    $('bench-p50').textContent=t?`${t.p50.toFixed(1)} ms`:'—';
    $('bench-p95').textContent=t?`${t.p95.toFixed(1)} ms`:'—';
    $('device').textContent=status.device==='mps'?'Apple GPU':'CPU';
  } catch(error) {message('모델 서버에 연결할 수 없습니다. 서버를 실행한 뒤 새로고침해주세요.');$('connection').textContent='서버 연결 필요';}
  drawHistory();
}
initialize();

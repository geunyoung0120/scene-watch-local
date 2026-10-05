"""Camera-free timing of JPEG decode -> DINOv2 -> logistic -> rolling mean."""
import csv
import io
import json
from pathlib import Path
import platform
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from violence_app.model import InferenceEngine
from violence_app.core import ProbabilityWindow, benchmark_interval


def main():
    rows=[r for r in csv.DictReader((ROOT/'data/manifest.csv').open()) if r['split']=='test']
    random.Random(12).shuffle(rows)
    payloads=[]
    for row in rows[:60]:
        with Image.open(ROOT/row['path']) as im:
            im.thumbnail((640,480))
            stream=io.BytesIO()
            im.save(stream,format='JPEG',quality=85)
            payloads.append(stream.getvalue())
    engine=InferenceEngine()
    cold_started=time.perf_counter()
    engine.predict_jpeg(payloads[0])
    first_ms=(time.perf_counter()-cold_started)*1000
    for payload in payloads[:8]:
        engine.predict_jpeg(payload)
    window=ProbabilityWindow()
    samples=[]
    for payload in payloads:
        started=time.perf_counter()
        p,stages=engine.predict_jpeg(payload)
        window.add(p,time.perf_counter())
        stages['pipeline_ms']=(time.perf_counter()-started)*1000
        samples.append(stages)
    stats={k:{'p50':float(np.percentile([r[k] for r in samples],50)),
              'p95':float(np.percentile([r[k] for r in samples],95)),
              'mean':float(np.mean([r[k] for r in samples]))} for k in samples[0]}
    interval=benchmark_interval(stats['pipeline_ms']['p95'])
    report={'device':engine.device,'platform':platform.platform(),'model':'DINOv2 Small CLS + logistic',
        'sample_count':len(samples),'warmup_count':9,'model_load_ms':engine.startup_ms,
        'first_inference_ms':first_ms,'timing_ms':stats,'recommended_interval_ms':interval,
        'interval_rule':'ceil(p95 * 1.25 / 10) * 10 ms, minimum 100 ms',
        'five_sample_span_ms':4*interval,
        'excludes':'Camera capture, browser JPEG encoding, HTTP transport and browser rendering',
        'samples':samples}
    (ROOT/'reports/benchmark.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2),flush=True)


if __name__=='__main__':
    main()

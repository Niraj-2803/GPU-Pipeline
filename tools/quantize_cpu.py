#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib
import nncf
import openvino as ov
from model_tools import tensor_at
from camex_pipeline.common import load_sources

p=argparse.ArgumentParser()
p.add_argument('--model',type=pathlib.Path,required=True)
p.add_argument('--sources',type=pathlib.Path,required=True)
p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--views',type=int,default=20)
p.add_argument('--quantize-head',action='store_true')
a=p.parse_args()
sources=load_sources(a.sources)[:a.views]
samples=[tensor_at(s,t) for s in sources for t in (0,10,20,40,50)]
core=ov.Core();model=core.read_model(a.model);model.reshape({model.input(0):[1,3,640,640]})
# Keep the YOLO detection head in floating point; quantify its effect separately.
options={} if a.quantize_head else {'ignored_scope':nncf.IgnoredScope(patterns=['.*model.22.*'])}
quantized=nncf.quantize(model,nncf.Dataset(samples),subset_size=len(samples),
    preset=nncf.QuantizationPreset.MIXED,target_device=nncf.TargetDevice.CPU,**options)
a.output.parent.mkdir(parents=True,exist_ok=True)
ov.save_model(quantized,a.output,compress_to_fp16=False)
a.output.with_suffix('.provenance.json').write_text(json.dumps({
    'original_model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),
    'precision':'INT8 including supported head operations' if a.quantize_head else 'INT8 backbone with floating-point detection head','calibration_count':len(samples),
    'calibration':[{'camera_id':s['camera_id'],'seconds':[0,10,20,40,50]} for s in sources],
    'note':'Calibration data is not labeled ground truth. Separate agreement evaluation is required.'},indent=2))
print(a.output)

#!/usr/bin/env python3
"""Build an engine on the actual deployment GPU; never assume portability."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from camex_pipeline.common import cuda_ordinal

p=argparse.ArgumentParser()
p.add_argument('--model',type=pathlib.Path,required=True)
p.add_argument('--output-dir',type=pathlib.Path,required=True)
p.add_argument('--precision',choices=['fp32','fp16','int8'],default='fp32')
p.add_argument('--calibration-dir',type=pathlib.Path)
p.add_argument('--opt-batch',type=int,choices=[1,2,4,8,16],default=4)
p.add_argument('--gpu-id',type=int,default=0)
p.add_argument('--image',default='camex-cmp-deepstream:7.0-review')
a=p.parse_args()
a.output_dir.mkdir(parents=True,exist_ok=True)
target=a.output_dir/('yolov8s-int8-head-b16.engine' if a.precision=='int8' else f'yolov8s-{a.precision}-b16.engine')
if target.exists(): raise SystemExit('Engine already exists; select a new output directory for a new build')
cmd=['docker','run','--rm','--gpus','all','-e','CUDA_DEVICE_ORDER=PCI_BUS_ID',
     '--mount',f'type=bind,src={a.model.resolve()},dst=/model.onnx,readonly',
     '--mount',f'type=bind,src={a.output_dir.resolve()},dst=/models',a.image,
     '/usr/src/tensorrt/bin/trtexec','--onnx=/model.onnx',
     f'--device={cuda_ordinal(a.gpu_id)}',
     '--minShapes=input:1x3x640x640',f'--optShapes=input:{a.opt_batch}x3x640x640','--maxShapes=input:16x3x640x640',
     f'--saveEngine=/models/{target.name}','--memPoolSize=workspace:2048','--skipInference']
calibration_manifest=None
if a.precision=='fp16': cmd.append('--fp16')
if a.precision=='int8':
    if not a.calibration_dir or not (a.calibration_dir/'manifest.json').is_file():
        raise SystemExit('INT8 requires --calibration-dir prepared by tools/prepare_calibration.py')
    calibration_manifest=json.loads((a.calibration_dir/'manifest.json').read_text())
    if calibration_manifest.get('frames_per_view')!=10 or calibration_manifest.get('size')!=640:
        raise SystemExit('Unexpected calibration format; expected ten RGB 640x640 frames per view')
    for i in range(1,calibration_manifest['views']+1):
        file=a.calibration_dir/f'C{i:02d}.rgb'
        if not file.is_file() or file.stat().st_size!=10*640*640*3:
            raise SystemExit(f'Incomplete calibration file: {file}')
    cmd=['docker','run','--rm','--gpus','all','-e','CUDA_DEVICE_ORDER=PCI_BUS_ID',
         '--mount',f'type=bind,src={a.model.resolve()},dst=/model.onnx,readonly',
         '--mount',f'type=bind,src={a.calibration_dir.resolve()},dst=/calibration,readonly',
         '--mount',f'type=bind,src={a.output_dir.resolve()},dst=/models',
         '--entrypoint','/opt/camex/bin/build_int8',a.image,'/model.onnx','/calibration',
         f'/models/{target.name}',str(cuda_ordinal(a.gpu_id)),str(calibration_manifest['views'])]
with (a.output_dir/f'build-{a.precision}.log').open('w') as log:
    r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
if r.returncode: raise SystemExit(f'Engine build failed, see build-{a.precision}.log')
info={'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),
      'engine_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'precision':a.precision,
      'gpu':subprocess.check_output(['nvidia-smi','-i',str(a.gpu_id),'--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip(),
      'image':a.image}
info['optimization_batch']=8 if a.precision=='int8' else a.opt_batch
info['calibration_manifest']=calibration_manifest
info['strategy']='INT8 backbone with FP16 detection head' if a.precision=='int8' else a.precision
target.with_suffix('.json').write_text(json.dumps(info,indent=2))
print(target)

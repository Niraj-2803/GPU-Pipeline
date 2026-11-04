#!/usr/bin/env python3
import argparse
import json
import pathlib
import subprocess
import numpy as np
import openvino as ov
from model_tools import tensor_at,match_boxes
from camex_pipeline.postprocess import decode_person_output
from camex_pipeline.common import cuda_ordinal, load_sources

p=argparse.ArgumentParser()
p.add_argument('--reference',type=pathlib.Path,required=True)
p.add_argument('--candidate',type=pathlib.Path,required=True)
p.add_argument('--sources',type=pathlib.Path,required=True)
p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--gpu-id',type=int,default=1)
p.add_argument('--views',type=int,default=20)
p.add_argument('--image',default='camex-cmp-deepstream:7.0-review')
p.add_argument('--candidate-size',type=int,default=640)
p.add_argument('--candidate-threshold',type=float,default=.25)
p.add_argument('--times',default='5,25,45')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
sources=load_sources(a.sources)[:a.views]
cases=[(s,float(t)) for s in sources for t in a.times.split(',')]
tensors=[tensor_at(s,t) for s,t in cases]
core=ov.Core();model=core.read_model(a.reference);model.reshape({model.input(0):[1,3,640,640]})
ref=core.compile_model(model,'CPU',{'INFERENCE_PRECISION_HINT':'f32','INFERENCE_NUM_THREADS':8,'NUM_STREAMS':1})
if a.candidate.suffix=='.engine':
    tensor_file=a.output/'inputs.float32';np.concatenate(tensors).tofile(tensor_file)
    subprocess.run(['docker','run','--rm','--gpus','all','-e','CUDA_DEVICE_ORDER=PCI_BUS_ID','-e',f'CAMEX_GPU_ID={cuda_ordinal(a.gpu_id)}',
      '--mount',f'type=bind,src={a.candidate.resolve()},dst=/candidate.engine,readonly',
      '--mount',f'type=bind,src={a.output.resolve()},dst=/out',a.image,
      '/opt/camex/bin/evaluate_tensor','/candidate.engine','/out/inputs.float32','/out/candidate.float32'],check=True)
    candidate=np.fromfile(a.output/'candidate.float32',np.float32).reshape(len(cases),1,8400,6)
else:
    model=core.read_model(a.candidate);model.reshape({model.input(0):[1,3,a.candidate_size,a.candidate_size]})
    compiled=core.compile_model(model,'CPU',{'INFERENCE_PRECISION_HINT':'f32','INFERENCE_NUM_THREADS':8,'NUM_STREAMS':1})
    candidate=[compiled(tensor_at(s,t,a.candidate_size))[compiled.output(0)].copy() for s,t in cases]
rows=[];ref_count=cand_count=matches=0
curve={v:{'candidate':0,'matched':0} for v in [.1,.15,.25]}
for (s,t),tensor,out in zip(cases,tensors,candidate):
    baseline=ref(tensor)[ref.output(0)]
    rb,_=decode_person_output(baseline,s['width'],s['height'])
    cb,_=decode_person_output(out,s['width'],s['height'],confidence=a.candidate_threshold,size=a.candidate_size)
    m=match_boxes(rb,cb,.5);ref_count+=len(rb);cand_count+=len(cb);matches+=m
    rows.append({'camera_id':s['camera_id'],'second':t,'reference_count':len(rb),'candidate_count':len(cb),'matched_iou_050':m})
    for threshold,data in curve.items():
        extra,_=decode_person_output(out,s['width'],s['height'],confidence=threshold,size=a.candidate_size)
        data['candidate']+=len(extra);data['matched']+=match_boxes(rb,extra,.5)
report={'frames':len(cases),'reference_detections':ref_count,'candidate_detections':cand_count,'matched':matches,
        'precision_against_reference':matches/max(1,cand_count),'recall_against_reference':matches/max(1,ref_count),
        'note':'Agreement with the FP32 model at confidence 0.25 and IoU 0.5, not labeled real-world accuracy.', 'rows':rows}
report['agreement_gate_pass']=min(report['precision_against_reference'],report['recall_against_reference'])>=.95
report['candidate_threshold']=a.candidate_threshold
report['threshold_sweep']={str(v):dict(data,precision=data['matched']/max(1,data['candidate']),recall=data['matched']/max(1,ref_count)) for v,data in curve.items()}
(a.output/'agreement.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))

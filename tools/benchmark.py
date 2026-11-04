#!/usr/bin/env python3
"""Run sequential, traceable comparisons. Never overlap timed trials."""
import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--backend',choices=['openvino','deepstream'],required=True)
p.add_argument('--sources',type=pathlib.Path,required=True)
p.add_argument('--model',type=pathlib.Path,required=True)
p.add_argument('--engine',type=pathlib.Path)
p.add_argument('--precision',default='fp32')
p.add_argument('--counts',default='1,2,5,10,20')
p.add_argument('--seconds',type=int,default=30)
p.add_argument('--warmup',type=int,default=8)
p.add_argument('--repeat',type=int,default=1)
p.add_argument('--gpu-id',type=int,default=0)
p.add_argument('--threads',type=int,default=8)
p.add_argument('--cpu-streams',type=int,default=1)
p.add_argument('--hyperthreads',action='store_true')
p.add_argument('--analysis-fps',type=int,default=10)
p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--record',action='store_true')
p.add_argument('--no-track',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
root=pathlib.Path(__file__).resolve().parents[1]
rows=[]
for n in [int(n) for n in a.counts.split(',')]:
    for repeat in range(a.repeat):
        name=f'{a.backend}_{a.precision}_n{n}_r{repeat+1}_{time.time_ns()}'
        folder=a.output/name
        cmd=[sys.executable,'-m','camex_pipeline.cli','--backend',a.backend,'--sources',str(a.sources.resolve()),
             '-n',str(n),'--model',str(a.model.resolve()),'--precision',a.precision,'--gpu-id',str(a.gpu_id),
             '--threads',str(a.threads),'--cpu-streams',str(a.cpu_streams),'--analysis-fps',str(a.analysis_fps),'--seconds',str(a.seconds),'--warmup',str(a.warmup),
             '--run-dir',str(folder.resolve()),'-out','yes' if a.record else 'no','-track','no' if a.no_track else 'yes']
        if a.engine:cmd+=['--engine',str(a.engine.resolve())]
        if a.hyperthreads:cmd+=['--hyperthreads']
        print('TRIAL',name,flush=True)
        with (a.output/f'{name}.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in proc.stdout:
                log.write(line);log.flush()
                if line.startswith(('FINAL','ERROR','RESULTS','VIDEO')):print(line.rstrip(),flush=True)
            code=proc.wait()
        row={'name':name,'count':n,'repeat':repeat+1,'exit_code':code,'folder':str(folder),'command':cmd}
        if (folder/'summary.json').exists():row['summary']=json.loads((folder/'summary.json').read_text())
        rows.append(row);(a.output/'trials.json').write_text(json.dumps(rows,indent=2))
print('COMPLETE',a.output/'trials.json',flush=True)

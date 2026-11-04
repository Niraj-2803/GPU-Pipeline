#!/usr/bin/env python3
"""Serve synthetic clips as distinct localhost-only RTSP feeds."""
import argparse
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from camex_pipeline.common import load_sources

p=argparse.ArgumentParser()
p.add_argument('--sources',type=pathlib.Path,required=True)
p.add_argument('-n',type=int,default=5)
p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--port',type=int,default=29554)
p.add_argument('--seconds',type=int,default=3600)
p.add_argument('--fault-camera')
p.add_argument('--fault-after',type=float,default=35)
p.add_argument('--fault-seconds',type=float,default=8)
a=p.parse_args()
sources=load_sources(a.sources,a.n)
if any(s['uri'].startswith(('rtsp://','rtsps://')) for s in sources):
    raise SystemExit('Replay input must be local files')
a.output.mkdir(parents=True,exist_ok=True)
name=f'camex-cmp-replay-{a.port}-{os.getpid()}'
config=a.output/'mediamtx.yml'
config.write_text(f'''logLevel: warn
api: no
metrics: no
rtsp: yes
rtspAddress: 127.0.0.1:{a.port}
rtspTransports: [tcp]
rtmp: no
hls: no
webrtc: no
srt: no
paths:
  all_others:
    source: publisher
''')
processes=[];commands=[];logs=[];running=True;relay_started=False
def stop(*_):
    global running
    running=False
for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
try:
    with socket.socket() as check:
        check.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        check.bind(('127.0.0.1',a.port))
    subprocess.run(['docker','run','-d','--rm','--name',name,'--network','host',
       '--mount',f'type=bind,src={config.resolve()},dst=/mediamtx.yml,readonly',
       'bluenviron/mediamtx:1.21.0'],check=True,stdout=subprocess.DEVNULL)
    relay_started=True
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        try:
            with socket.create_connection(('127.0.0.1',a.port),timeout=.5):break
        except OSError:time.sleep(.1)
    else:raise RuntimeError('Local RTSP relay did not start')
    published=[]
    for s in sources:
        url=f"rtsp://127.0.0.1:{a.port}/{s['camera_id']}"
        log=(a.output/f"{s['camera_id']}.log").open('w')
        cmd=['ffmpeg','-nostdin','-hide_banner','-loglevel','warning','-re','-stream_loop','-1',
             '-i',s['uri'],'-an','-c:v','copy','-rtsp_transport','tcp','-f','rtsp',url]
        process=subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=log)
        processes.append(process)
        commands.append(cmd);logs.append(log)
        published.append(dict(s,uri=url))
    time.sleep(2)
    if any(proc.poll() is not None for proc in processes):raise RuntimeError('A replay publisher failed; inspect replay logs')
    (a.output/'sources.json').write_text(json.dumps({'schema_version':'1.0','note':'Local synthetic-file RTSP replay, not physical cameras','sources':published},indent=2))
    print('RTSP READY',a.output/'sources.json',flush=True)
    start=time.monotonic();end=start+a.seconds
    fault_index=next((i for i,s in enumerate(sources) if s['camera_id']==a.fault_camera),None)
    injected=False;resume_at=None
    while running and time.monotonic()<end:
        now=time.monotonic()
        if fault_index is not None and not injected and now>=start+a.fault_after:
            processes[fault_index].terminate();processes[fault_index].wait(timeout=8)
            processes[fault_index]=None;injected=True;resume_at=now+a.fault_seconds
            with (a.output/'faults.jsonl').open('a') as f:f.write(json.dumps({'event':'disconnect','camera_id':a.fault_camera,'monotonic_s':now})+'\n')
        if resume_at is not None and now>=resume_at:
            processes[fault_index]=subprocess.Popen(commands[fault_index],stdout=subprocess.DEVNULL,stderr=logs[fault_index])
            with (a.output/'faults.jsonl').open('a') as f:f.write(json.dumps({'event':'reconnect','camera_id':a.fault_camera,'monotonic_s':now})+'\n')
            resume_at=None
        if any(proc is not None and proc.poll() is not None for proc in processes):raise RuntimeError('A replay publisher stopped')
        time.sleep(.5)
finally:
    for proc in processes:
        if proc is not None and proc.poll() is None:proc.terminate()
    for proc in processes:
        if proc is None:continue
        try:proc.wait(timeout=8)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
    if relay_started:subprocess.run(['docker','stop','-t','5',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    for log in logs:log.close()

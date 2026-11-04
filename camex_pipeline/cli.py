import argparse
import datetime
import json
import math
import os
import pathlib
import re
import signal
import subprocess
import threading
import time
from urllib.parse import quote

from .common import ResourceMonitor, cuda_ordinal, load_sources, percentile, redact_uri, sha256, validate_frame

IMAGE = 'camex-cmp-deepstream:7.0-review'

def arguments():
    p=argparse.ArgumentParser(description='Portable CAMEX detector/tracker comparison')
    p.add_argument('--backend',choices=['deepstream','openvino'],default=os.getenv('CAMEX_DEFAULT_BACKEND','deepstream'))
    p.add_argument('--sources',type=pathlib.Path,required=True)
    p.add_argument('-n',type=int)
    p.add_argument('-res',choices=['FHD','2K'])
    p.add_argument('--source-fps',type=float)
    p.add_argument('--analysis-fps',type=int,choices=range(1,31),default=10)
    p.add_argument('-track',choices=['yes','no'],default='yes')
    p.add_argument('-out',choices=['yes','no'],default='no')
    p.add_argument('--mosaic-fps',type=int,choices=[5,10],default=5)
    p.add_argument('--seconds',type=float,default=30)
    p.add_argument('--warmup',type=float,default=8)
    p.add_argument('--model',type=pathlib.Path,required=True)
    p.add_argument('--engine',type=pathlib.Path)
    p.add_argument('--precision',choices=['fp32','fp16','int8'],default='fp32')
    p.add_argument('--batch',type=int,choices=[1,2,4,8,16],default=4)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--requests',type=int,default=2)
    p.add_argument('--cpu-streams',choices=['1','2','4','8','AUTO'],default='1')
    p.add_argument('--cpu-batch',type=int,choices=[1,2,4,8],default=1)
    p.add_argument('--performance-hint',choices=['LATENCY','THROUGHPUT'],default='LATENCY')
    p.add_argument('--analysis-size',type=int,choices=[320,384,416,512,640],default=640)
    p.add_argument('--detector',choices=['yolo','ssd'],default='yolo')
    p.add_argument('--ssd-class-id',type=int,default=0)
    p.add_argument('--mux-size',choices=['native','1280x720','960x540','640x360'],default='native')
    p.add_argument('--mux-timeout-us',type=int,default=10000)
    p.add_argument('--decode-surfaces',type=int,default=2)
    p.add_argument('--profile-stages',action='store_true')
    p.add_argument('--vram-reserve-mib',type=int,default=768)
    p.add_argument('--startup-timeout',type=int,default=45)
    p.add_argument('--reconnect-seconds',type=int,default=15)
    p.add_argument('--hyperthreads',action='store_true')
    p.add_argument('--threshold',type=float,default=.25)
    p.add_argument('--nms',type=float,default=.45)
    p.add_argument('--gpu-id',type=int,default=0)
    p.add_argument('--image',default=IMAGE)
    p.add_argument('--run-dir',type=pathlib.Path)
    p.add_argument('--runs-root',type=pathlib.Path,default=pathlib.Path('runs'))
    a=p.parse_args()
    if not 1<=a.seconds<=86400 or not 0<=a.warmup<=90:
        p.error('seconds must be 1..86400 and warmup 0..90')
    if not 0<a.threshold<1 or not 0<a.nms<1 or a.threads<1 or a.requests<1:
        p.error('Invalid thresholds or CPU parallelism')
    a.model=a.model.resolve()
    if not a.model.is_file():
        p.error('The model file is missing; supply an ONNX or OpenVINO XML with --model')
    if a.backend=='openvino' and a.precision=='fp16':
        p.error('CPU supports the FP32 baseline or a separately evaluated INT8 model')
    if a.backend=='deepstream' and a.analysis_size!=640:
        p.error('The GPU reference engine requires analysis-size 640')
    if a.detector=='ssd' and (a.backend!='openvino' or a.cpu_batch!=1):
        p.error('The SSD CPU profile requires OpenVINO with cpu-batch 1')
    if not 1000<=a.mux_timeout_us<=100000 or not 0<=a.decode_surfaces<=16:
        p.error('Invalid GPU buffering parameters')
    if not 1<=a.startup_timeout<=600 or not 1<=a.reconnect_seconds<=300 or a.vram_reserve_mib<0:
        p.error('Invalid startup/reconnect timeout or GPU memory reserve')
    return a

def run_gpu(a,sources,folder,run_id,monitor):
    if not a.engine or not a.engine.is_file():
        raise ValueError('Build a matching TensorRT engine first and supply --engine')
    widths={(s['width'],s['height']) for s in sources}
    if len(widths)!=1:
        raise ValueError('This GPU baseline requires one native resolution per run')
    w,h=next(iter(widths))
    if (w,h) not in [(1920,1080),(2560,1440)]:
        raise ValueError('GPU baseline supports FHD and 2560x1440')
    all_files=all(not s['uri'].startswith(('rtsp://','rtsps://')) for s in sources)
    if not all(s['fps'] in (10,30) for s in sources) or len({s['fps'] for s in sources})!=1:
        raise ValueError('GPU reference requires uniform encoded 10 or 30 FPS sources')
    name='camex-cmp-'+run_id.lower()
    device=cuda_ordinal(a.gpu_id)
    mw,mh=(w,h) if a.mux_size=='native' else tuple(map(int,a.mux_size.split('x')))
    cmd=['docker','run','--rm','--name',name,'--gpus','all',
         '--network','host','--ulimit','nofile=8192:8192','--shm-size','1g',
         '-e','NVIDIA_DRIVER_CAPABILITIES=compute,utility,video',
         '-e','CUDA_DEVICE_ORDER=PCI_BUS_ID',
         '-e',f'CAMEX_RUN_ID={run_id}','-e',f'CAMEX_SYNC_START={int(all_files)}',
         '-e',f'CAMEX_GPU_ID={device}',
         '-e',f'CAMEX_INFER_BATCH={a.batch}','-e',f'CAMEX_ANALYSIS_FPS={a.analysis_fps}',
         '-e',f'CAMEX_SOURCE_FPS={sources[0]["fps"]}',
         '-e',f'CAMEX_STARTUP_TIMEOUT={a.startup_timeout}',
         '-e',f'CAMEX_RECONNECT_SECONDS={a.reconnect_seconds}',
         '-e',f'CAMEX_MOSAIC_FPS={a.mosaic_fps}',
         '-e',f'CAMEX_MUX_W={mw}','-e',f'CAMEX_MUX_H={mh}',
         '-e',f'CAMEX_MUX_TIMEOUT={a.mux_timeout_us}',
         '-e',f'CAMEX_DECODE_SURFACES={a.decode_surfaces}',
         '-e',f'CAMEX_PROFILE_STAGES={int(a.profile_stages)}',
         '--mount',f'type=bind,src={folder},dst=/out',
         '--mount',f'type=bind,src={a.engine.resolve()},dst=/models/detector.engine,readonly']
    lines=[]
    for i,s in enumerate(sources):
        uri=s['uri']
        if not uri.startswith(('rtsp://','rtsps://')):
            dst=f'/sources/{i:04d}{pathlib.Path(uri).suffix}'
            cmd.extend(['--mount',f'type=bind,src={uri},dst={dst},readonly'])
            uri='file://'+quote(dst,safe='/')
        lines.append(f"{uri}\t{s['camera_id']}\n")
    secret_manifest=folder/'sources.private.tsv'
    secret_manifest.write_text(''.join(lines));secret_manifest.chmod(0o600)
    config=f'''[property]
gpu-id={device}
net-scale-factor=0.00392156862745098
model-color-format=0
model-engine-file=/models/detector.engine
batch-size={a.batch}
network-mode={dict(fp32=0,fp16=2,int8=1)[a.precision]}
num-detected-classes=1
interval=0
gie-unique-id=1
process-mode=1
network-type=0
cluster-mode=2
maintain-aspect-ratio=1
symmetric-padding=1
parse-bbox-func-name=NvDsInferParseYoloPerson
custom-lib-path=/opt/camex/lib/libperson_parser.so
[class-attrs-all]
nms-iou-threshold={a.nms}
pre-cluster-threshold={a.threshold}
topk=300
'''
    (folder/'infer.ini').write_text(config)
    cmd += [a.image,'/opt/camex/bin/deepstream_runner',str(len(sources)), '/out/infer.ini','/out',
            lines[0].split('\t')[0], str(max(1,int(sources[0]['fps']//a.analysis_fps))),
            '1' if a.out=='yes' else '0',str(a.seconds),str(w),str(h),
            '1' if a.track=='yes' else '0','/out/sources.private.tsv',str(a.warmup)]
    process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    abort=threading.Event()
    def cancel(*_):
        abort.set()
        subprocess.run(['docker','stop','-t','8',name],capture_output=True,timeout=20)
    previous={s:signal.signal(s,cancel) for s in [signal.SIGINT,signal.SIGTERM]}
    def watchdog():
        deadline=time.monotonic()+a.seconds+a.warmup+a.startup_timeout+30
        while not abort.wait(1):
            if monitor.guard_reason or time.monotonic()>deadline:
                cancel()
                return
    watchdog_thread=threading.Thread(target=watchdog,daemon=True)
    watchdog_thread.start()
    try:
        with (folder/'native.log').open('w') as log:
            for line in process.stdout:
                log.write(line);log.flush()
                if line.startswith(('FPS ','READY','START','FINISH','RESULT','ERROR','WARMUP','SOURCE')):
                    # Native debug logs can include RTSP credentials; they stay private.
                    shown=re.sub(r'rtsp[s]?://\S+','<redacted-rtsp>',line).rstrip()
                    if line.startswith('FPS ') and monitor.samples:
                        r=monitor.samples[-1]
                        shown+=f" | CPU={r['cpu_percent']:.1f}% GPU={r.get('gpu_percent',0):.0f}% decode={r.get('decoder_percent',0):.0f}% VRAM={r.get('vram_mib',0):.0f}MiB"
                    print(shown,flush=True)
        code=process.wait()
    finally:
        abort.set()
        if process.poll() is None:
            cancel();process.wait(timeout=30)
        for sig,handler in previous.items():
            signal.signal(sig,handler)
        secret_manifest.unlink(missing_ok=True)
    if not (folder/'result.json').exists():
        raise RuntimeError(f'GPU runner failed before results (exit {code}); see native.log')
    native=json.loads((folder/'result.json').read_text())
    if monitor.guard_reason: native['error']=monitor.guard_reason
    native['native_exit_code']=code
    (folder/'native_result.json').write_text(json.dumps(native,indent=2))
    return native

def summarize(a,sources,folder,run_id,result,monitor):
    counts={s['camera_id']:0 for s in sources}
    latency={k:[] for k in counts}
    tracks={k:set() for k in counts}
    previous={}
    first_timestamp={}
    validation_errors=[]
    with (folder/'metadata.jsonl').open() as f:
        for line in f:
            event=json.loads(line)
            try:
                validate_frame(event)
                cid=event['camera_id'];counts[cid]+=1
                epoch=event.get('tracking_epoch',0)
                if cid in previous and epoch==previous[cid][0] and event['source_pts_ns']<=previous[cid][1]:
                    validation_errors.append(f'Non-monotonic source timestamp: {cid}')
                previous[cid]=(epoch,event['source_pts_ns'])
                first_timestamp.setdefault(cid,event['source_pts_ns'])
                if event.get('processing_ms',-1)>=0:
                    latency[cid].append(event['processing_ms'])
                tracks[cid].update((epoch,o['track_id']) for o in event['objects'] if o['track_id'] is not None)
            except (KeyError,ValueError) as error:
                validation_errors.append(str(error))
    seconds=result.get('seconds',a.seconds)
    cameras=[{'camera_id':cid,'frames':n,'fps':n/max(seconds,.001),
              'p50_processing_ms':percentile(latency[cid],50),'p95_processing_ms':percentile(latency[cid],95),
              'unique_track_ids':len(tracks[cid])} for cid,n in counts.items()]
    for source in sources:
        if not source['uri'].startswith(('rtsp://','rtsps://')) and counts[source['camera_id']]>math.ceil(source['fps']*seconds)+2:
            validation_errors.append(f"File playback exceeded encoded source rate: {source['camera_id']}")
        cid=source['camera_id']
        if not source['uri'].startswith(('rtsp://','rtsps://')) and cid in previous and (previous[cid][1]-first_timestamp[cid])/1e9>seconds+.5:
            validation_errors.append(f"File timestamp span exceeded wall time: {cid}")
    start=result.get('measurement_start_monotonic') or 0
    status=not(result.get('error') or result.get('interrupted') or result.get('native_exit_code',0) or validation_errors or result.get('metadata_dropped',0))
    status=status and all(c['fps']>=a.analysis_fps*.95 for c in cameras)
    status=status and seconds>=a.seconds-.05
    flat_latency=[v for values in latency.values() for v in values]
    summary={'schema_version':'1.0','run_id':run_id,'backend':a.backend,
        'tracker':('nvdcf' if a.backend=='deepstream' else 'bytetrack') if a.track=='yes' else 'none',
        'precision':a.precision,'model_sha256':sha256(a.model),'engine_sha256':sha256(a.engine) if a.engine else None,
        'source_count':len(sources),'source_kind':'rtsp' if all(s['uri'].startswith(('rtsp://','rtsps://')) for s in sources) else 'file-or-mixed',
        'source_width':sources[0]['width'],'source_height':sources[0]['height'],
        'encoded_source_fps':sources[0]['fps'],'source_is_upscaled':any(s.get('is_upscaled',False) for s in sources),
        'unique_source_views':len({s['original_camera_id'] for s in sources}),
        'seconds':seconds,'warmup_seconds':a.warmup,'analysis_target_fps':a.analysis_fps,
        'recording':a.out=='yes','mosaic_target_fps':a.mosaic_fps if a.out=='yes' else None,
        'per_camera':cameras,'total_fps':sum(c['fps'] for c in cameras),'minimum_fps':min(c['fps'] for c in cameras),
        'p50_processing_ms':percentile(flat_latency,50),'p95_processing_ms':percentile(flat_latency,95),
        'throughput_check':'PASS' if status else 'BELOW_TARGET_OR_ERROR',
        'error':result.get('error'),'validation_errors':validation_errors[:10],
        'metadata_dropped':result.get('metadata_dropped',0),'resources':monitor.summary(start,start+seconds),
        'note':'Processing latency is paced decoded-frame arrival to metadata. It is not physical camera-to-alert latency. Camera-local IDs are not unique-person counts.',
        'native':result}
    if 'model_input_shape' in result:
        summary['model_input_shape']=result['model_input_shape']
        summary['preprocessing']=result['preprocessing']
    provenance=a.model.with_suffix('.provenance.json')
    if provenance.exists():summary['model_provenance']=json.loads(provenance.read_text())
    if a.model.suffix=='.xml':summary['model_weights_sha256']=sha256(a.model.with_suffix('.bin'))
    (folder/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    import csv
    with (folder/'per_camera.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=cameras[0].keys());writer.writeheader();writer.writerows(cameras)
    return summary

def main():
    a=arguments()
    sources=load_sources(a.sources,a.n)
    for s in sources:
        expected={'FHD':(1920,1080),'2K':(2560,1440)}.get(a.res)
        if expected and (s['width'],s['height'])!=expected:
            raise ValueError('The selected source file does not match -res')
        if a.source_fps and abs(s['fps']-a.source_fps)>.01:
            raise ValueError('Source FPS mismatch; changing a flag does not change encoded FPS')
    checked=set()
    for s in sources:
        if s['uri'].startswith(('rtsp://','rtsps://')) or s['uri'] in checked:
            continue
        checked.add(s['uri'])
        probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
            'stream=width,height,avg_frame_rate:format=duration','-of','json',s['uri']],capture_output=True,text=True,timeout=15)
        if probe.returncode: raise ValueError(f"Cannot probe {s['camera_id']}")
        information=json.loads(probe.stdout);stream=information['streams'][0]
        from fractions import Fraction
        if (stream['width'],stream['height'])!=(s['width'],s['height']) or abs(float(Fraction(stream['avg_frame_rate']))-s['fps'])>.05:
            raise ValueError(f"Source dimensions/FPS do not match configuration: {s['camera_id']}")
        duration=float(information.get('format',{}).get('duration',0))
        if duration and duration<a.seconds+a.warmup+1:
            raise ValueError('Source file is shorter than the run plus warmup; use tools/replay_rtsp.py for looping tests')
    run_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder=(a.run_dir or a.runs_root/f'{run_id}_{a.backend}_n{len(sources)}').resolve()
    folder.mkdir(parents=True,exist_ok=False)
    settings={k:str(v) if isinstance(v,pathlib.Path) else v for k,v in vars(a).items()}
    settings['sources']=[dict(s,uri=redact_uri(s['uri'])) for s in sources]
    here=pathlib.Path(__file__).parent
    settings['source_file_hashes']={p.name:sha256(p) for p in here.glob('*.py')}
    if a.backend=='deepstream':
        image_info=subprocess.run(['docker','image','inspect','--format','{{.Id}}',a.image],capture_output=True,text=True)
        settings['image_id']=image_info.stdout.strip()
    (folder/'settings.json').write_text(json.dumps(settings,indent=2))
    print(f'RUN FOLDER: {folder}',flush=True)
    with ResourceMonitor(folder,a.gpu_id,a.vram_reserve_mib if a.backend=='deepstream' else 0) as monitor:
        try:
            if a.backend=='deepstream':
                result=run_gpu(a,sources,folder,run_id,monitor)
            else:
                from .cpu import CpuPipeline
                pipeline=CpuPipeline(a,sources,folder,run_id)
                pipeline.resource_monitor=monitor
                previous={s:signal.signal(s,lambda *_:pipeline.stop.set()) for s in [signal.SIGINT,signal.SIGTERM]}
                try:
                    result=pipeline.run()
                finally:
                    for sig,handler in previous.items(): signal.signal(sig,handler)
                (folder/'result.json').write_text(json.dumps(result,indent=2))
        except Exception as error:
            message=re.sub(r'rtsp[s]?://\S+','<redacted-rtsp>',str(error))
            print('ERROR '+message,flush=True)
            result={'error':message,'seconds':0,'metadata_dropped':0}
            (folder/'result.json').write_text(json.dumps(result,indent=2))
            (folder/'metadata.jsonl').touch(exist_ok=True)
    result=summarize(a,sources,folder,run_id,result,monitor)
    print(f"FINAL: {result['total_fps']:.2f} FPS total; slowest {result['minimum_fps']:.2f} FPS; processing P95 {result['p95_processing_ms']} ms; {result['throughput_check']}")
    print(f'RESULTS FOLDER: {folder}')
    if a.out=='yes': print(f'VIDEO: {folder / "mosaic.mp4"}')
    native=result['native']
    return 0 if not result['error'] and not result['validation_errors'] and not native.get('native_exit_code') and not native.get('interrupted') else 1

if __name__=='__main__':
    raise SystemExit(main())

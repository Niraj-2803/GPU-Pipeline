import argparse
import csv
import datetime
import fcntl
import hashlib
from fractions import Fraction
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import quote

ROOT=pathlib.Path(__file__).resolve().parents[1]
CONTAINER='camex-tune150200-20261015'

def arguments(argv=None):
    p=argparse.ArgumentParser(description='Isolated capacity sweep: independent 2K H.265 inputs, mixed INT8 and NvDCF.')
    p.add_argument('-n',type=int,required=True,help='Number of independently decoded streams, 1 to 400; resource guards apply')
    p.add_argument('-out',choices=['yes','no'],default='no',help='Save an overlaid all-stream mosaic (adds GPU and encoder load)')
    p.add_argument('-track',choices=['yes','no'],default='yes',help='Person tracking with NvDCF (default yes); no for detection-only comparison')
    p.add_argument('-set','--dataset',dest='dataset',choices=['airport50'],default='airport50',help='Isolated experiment using C01-C50 in order')
    p.add_argument('--precision',choices=['fp16','int8-head'],default='int8-head')
    p.add_argument('--tracker-preset',choices=['perf','max_perf'],default='perf')
    p.add_argument('--mux-size',choices=['native','960','640'],default='native')
    p.add_argument('--batch',type=int,choices=[4,8,16],default=8)
    p.add_argument('--tracker-groups',type=int,choices=[1,2,4],default=1)
    p.add_argument('--decode-surfaces',type=int,choices=[2,8,16],default=2)
    p.add_argument('--codec',choices=['h264','h265','h264-lowlatency'],default='h265',help='h264 is the unchanged baseline; other profiles use GOP20 and no B frames')
    p.add_argument('--preroll',action='store_true',help='Load the pipeline in PAUSED before starting the playback clock')
    p.add_argument('--profile-stages',action='store_true',help='Measure mux age, inference transit and tracker transit')
    p.add_argument('--sync-start',action='store_true',help='Start file playback on a common epoch after all decoders initialize')
    p.add_argument('--mosaic-fps',type=int,choices=[5,10],default=10,help='Recording frame rate, independent of the 10-FPS AI rate')
    p.add_argument('--mux-timeout',type=int,default=10000,help='Batch formation timeout in microseconds')
    p.add_argument('--extended-source',action='store_true',help='Use a lossless 180-second repetition for longer steady-state tests')
    p.add_argument('--analysis-fps',type=int,choices=[9,10],default=10,help='AI frame-rate target; 9 drops one of ten decoded input frames')
    p.add_argument('-res','--resolution',dest='resolution',type=str.upper,choices=['FHD','2K'],default='2K',help='Input resolution: FHD=1920x1080; 2K=2560x1440 upscaled workload sample')
    p.add_argument('--source-fps',type=int,choices=[30,10],default=10,help='Encoded input FPS; default 10 for the verified H.265 profile')
    p.add_argument('--seconds',type=int,default=10,help='Measured seconds after warm-up; 1 to 120')
    p.add_argument('--warmup',type=int,default=8,help='Warm-up after all sources are ready, 0 to 90 seconds; default 8')
    args=p.parse_args(argv)
    if not 0<=args.mux_timeout<=100000:p.error('--mux-timeout must be 0 to 100000 microseconds')
    if args.sync_start and args.preroll:p.error('Use --sync-start without --preroll for these live source bins')
    if not 1<=args.n<=400:p.error('-n must be between 1 and 400')
    if not 1<=args.seconds<=120:p.error('--seconds must be between 1 and 120')
    if not 0<=args.warmup<=90:p.error('--warmup must be between 0 and 90')
    if args.seconds+args.warmup>45 and not args.extended_source:p.error('Use --extended-source for a longer test')
    if args.analysis_fps==9 and args.source_fps!=10:p.error('9-FPS analysis requires the 10-FPS input profile')
    return args

def command(*args,**kwargs):
    return subprocess.run(list(args),check=True,text=True,**kwargs)

def source_profile(args):
    if args.resolution=='2K':
        return ROOT/f'C18_2K_{args.source_fps}FPS_upscaled.mp4',2560,1440,True
    return ROOT/('C18.mp4' if args.source_fps==30 else 'C18_10FPS_diagnostic.mp4'),1920,1080,False

def validate_source(source,width,height,fps):
    probe=command('ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate,codec_name','-of','json',str(source),capture_output=True,timeout=15)
    streams=json.loads(probe.stdout)['streams']
    if len(streams)!=1:raise ValueError('Expected one video stream in the benchmark input')
    stream=streams[0]
    actual_fps=float(Fraction(stream['avg_frame_rate']))
    if (stream['width'],stream['height'])!=(width,height) or abs(actual_fps-fps)>.01:
        raise ValueError(f'Input does not match selected profile: {stream["width"]}x{stream["height"]} at {actual_fps:g} FPS')
    return stream

def resolve_sources(args):
    if args.dataset=='c18':
        source,width,height,upscaled=source_profile(args)
        entries=[{'camera_id':'C18','area':'DOCUMENTS','description':'Single repeated C18 clip','path':str(source.relative_to(ROOT))}]
    else:
        manifest_path=ROOT/'datasets/airport50/manifest.json' if args.codec=='h264' else ROOT/'source_profiles'/args.codec/'manifest.json'
        if args.extended_source:manifest_path=ROOT/'long_sources'/args.codec/'manifest.json'
        if not manifest_path.exists():raise ValueError('airport50 assets are not prepared yet')
        manifest=json.loads(manifest_path.read_text())
        if args.codec=='h264' and not {'FHD_30','FHD_10','2K_30','2K_10'}.issubset(manifest['profiles']):
            raise ValueError('airport50 is still preparing its profiles; wait until asset preparation finishes')
        key=f'{args.resolution}_{args.source_fps}'
        if key not in manifest['profiles']:raise ValueError(f'airport50 profile {key} is not prepared yet')
        profile=manifest['profiles'][key]
        width,height,upscaled=profile['width'],profile['height'],profile['upscaled']
        expected=(2560,1440) if args.resolution=='2K' else (1920,1080)
        if (width,height)!=expected or profile['fps']!=args.source_fps or bool(upscaled)!=(args.resolution=='2K'):
            raise ValueError('Dataset profile does not match the requested resolution/FPS')
        entries=sorted(profile['sources'],key=lambda x:x['camera_id'])
        if [e['camera_id'] for e in entries]!=[f'C{i:02d}' for i in range(1,51)]:raise ValueError('airport50 must contain exactly C01-C50')
        if len({e['path'] for e in entries})!=50 or len({e['sha256'] for e in entries})!=50:raise ValueError('airport50 requires fifty distinct verified video files')
    assignments=[];verified={}
    for i in range(args.n):
        entry=entries[i%len(entries)];path=(ROOT/entry['path']).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not re.fullmatch(r'C\d{2}',entry['camera_id']):raise ValueError('Invalid dataset path or camera label')
        if not path.exists():raise ValueError(f'Missing dataset video: {path}')
        if path not in verified:
            if 'bytes' in entry and path.stat().st_size!=entry['bytes']:raise ValueError(f'Dataset file size changed: {path}')
            verified[path]=validate_source(path,width,height,args.source_fps)
            expected_codec='hevc' if args.codec=='h265' else 'h264'
            if verified[path].get('codec_name',expected_codec)!=expected_codec:raise ValueError('Encoded codec does not match requested profile')
        assignments.append({'stream_id':i+1,'camera_id':entry['camera_id'],'replica':i//len(entries)+1,'area':entry.get('area',''),'description':entry.get('description',''),'path':str(path),
                            'uri':'file:///work/'+quote(str(path.relative_to(ROOT.resolve())),safe='/'),'expected_sha256':entry.get('sha256'),'probe':verified[path]})
    return assignments,width,height,upscaled

def main():
    args=arguments()
    if os.geteuid()!=0:os.execvp('sudo',['sudo',sys.executable,str(pathlib.Path(__file__).resolve()),*sys.argv[1:]])
    lock=(ROOT/'.interactive.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:sys.exit('Another interactive benchmark is running. Wait for it to finish.')
    for path in [ROOT/'scripts/cap_interactive',ROOT/f'models/yolov8s-{args.precision}-b16.engine',ROOT/'models/libyolo_parser.so']:
        if not path.exists():sys.exit(f'Missing required file: {path}')
    try:assignments,width,height,upscaled=resolve_sources(args)
    except (ValueError,KeyError,subprocess.SubprocessError,OSError) as error:sys.exit(f'Input validation failed: {error}')
    source=pathlib.Path(assignments[0]['path']);source_info=assignments[0]['probe']
    unique_views=len({a['camera_id'] for a in assignments})
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder=ROOT/'runs'/f'{stamp}_{args.precision}_{args.codec}_{args.tracker_preset}_mux{args.mux_size}_b{args.batch}_pre{int(args.preroll)}_n{args.n}_src{args.source_fps}_track-{args.track}_out-{args.out}_warm{args.warmup}'
    folder.mkdir(parents=True)
    folder_in='/work/'+str(folder.relative_to(ROOT))
    (folder/'sources.tsv').write_text(''.join(f'{a["uri"]}\t{a["camera_id"]}\n' for a in assignments))
    (folder/'source_assignments.json').write_text(json.dumps(assignments,indent=2))
    config_path=folder_in+'/infer.txt'
    (folder/'infer.txt').write_text(f'''[property]
gpu-id=0
net-scale-factor=0.00392156862745098
model-color-format=0
model-engine-file=/work/models/yolov8s-{args.precision}-b16.engine
labelfile-path=/work/models/labels.txt
batch-size={args.batch}
network-mode={1 if args.precision.startswith('int8') else 2}
num-detected-classes=80
interval=0
gie-unique-id=1
process-mode=1
network-type=0
cluster-mode=2
maintain-aspect-ratio=1
symmetric-padding=1
parse-bbox-func-name={'NvDsInferParseYoloPerson' if args.precision.startswith('int8') else 'NvDsInferParseYolo'}
custom-lib-path=/work/models/{'libyolo_person_parser.so' if args.precision.startswith('int8') else 'libyolo_parser.so'}
[class-attrs-all]
nms-iou-threshold=0.45
pre-cluster-threshold=0.25
topk=300
''')
    command('docker','start',CONTAINER,stdout=subprocess.DEVNULL)
    tracker_config=f'/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_{args.tracker_preset}.yml'
    mux_dimensions={'native':[width,height],'960':[960,540],'640':[640,360]}[args.mux_size]
    tracker={'enabled':args.track=='yes','algorithm':'NvDCF' if args.track=='yes' else None,'dimensions':[640,384] if args.track=='yes' else None,'config':tracker_config if args.track=='yes' else None,'stream_capacity':args.n,'identity_scope':'per camera, per run; no cross-camera ReID'}
    if args.track=='yes':
        tracker_text=command('docker','exec',CONTAINER,'cat',tracker_config,capture_output=True).stdout
        tracker['config_sha256']=hashlib.sha256(tracker_text.encode()).hexdigest()
    engine_hash=hashlib.sha256((ROOT/f'models/yolov8s-{args.precision}-b16.engine').read_bytes()).hexdigest()
    (folder/'settings.json').write_text(json.dumps(vars(args)|{'model':'YOLOv8s','engine_sha256':engine_hash,'analysis_size':640,'tracker':tracker,'person_class_filter':0,'unique_source_views':unique_views,'repeated_inputs':args.n-unique_views,'source':str(source),'source_width':width,'source_height':height,'source_probe':source_info,'source_is_upscaled':upscaled,'original_source_dimensions':[1920,1080],'mux_dimensions':mux_dimensions,'independent_decode':True,'warmup_seconds':args.warmup,'timestamp_utc':stamp},indent=2))
    mode='detection + tracking' if args.track=='yes' else 'detection only'
    print(f'\nCAMEX interactive benchmark | {mode} | n={args.n} | input={args.resolution} {width}x{height} | source={args.source_fps} FPS | analysis target={args.analysis_fps} FPS each | output={args.out}',flush=True)
    if args.analysis_fps==9:print('SAMPLING: drop one of every ten decoded source frames, phase-staggered by camera. Source/decode rate stays 10 FPS.',flush=True)
    print('Detector input remains 640x640. '+('2K is upscaled from FHD for load testing, not native-2K quality evaluation.' if upscaled else 'FHD source dimensions are unchanged.'),flush=True)
    print(f'DATASET {args.dataset}: {unique_views} distinct synthetic camera views across {args.n} independent inputs; {args.n-unique_views} repeated inputs. Not a live RTSP test.',flush=True)
    print('NvDCF person tracking is enabled. Track IDs and colours persist within each camera; no cross-camera identification.' if args.track=='yes' else 'Tracking is OFF. Person detections only, with no persistent IDs.',flush=True)
    print('Video mode adds tiling, labels, overlays and encoding. Do not compare it as headless capacity.' if args.out=='yes' else 'Headless mode: no tiler, OSD or video encoder.',flush=True)
    print(f'RUN FOLDER: {folder}\n',flush=True)
    print(f'EXPERIMENT precision={args.precision} codec={args.codec} tracker={args.tracker_preset} groups={args.tracker_groups} mux={mux_dimensions} batch={args.batch} preroll={args.preroll}',flush=True)
    if args.sync_start:print('FILE PLAYBACK: wait for all decoders, then use one common playback epoch. Initialization is reported separately; no frames are skipped for 10-FPS analysis.',flush=True)
    argv=['docker','exec','-e',f'CAMEX_TRACKER_CONFIG={tracker_config}','-e',f'CAMEX_DECODE_SURFACES={args.decode_surfaces}','-e',f'CAMEX_TRACKER_GROUPS={args.tracker_groups}','-e',f'CAMEX_MUX_W={mux_dimensions[0]}','-e',f'CAMEX_MUX_H={mux_dimensions[1]}','-e',f'CAMEX_INFER_BATCH={args.batch}','-e',f'CAMEX_PREROLL={int(args.preroll)}',CONTAINER,'/work/scripts/cap_interactive',str(args.n),config_path,folder_in,assignments[0]['uri'],'3' if args.source_fps==30 else '1','1' if args.out=='yes' else '0',str(args.seconds),str(width),str(height),'1' if args.track=='yes' else '0',folder_in+'/sources.tsv',str(args.warmup)]
    argv[2:2]=['-e',f'CAMEX_ANALYSIS_FPS={args.analysis_fps}']
    argv[2:2]=['-e',f'CAMEX_PROFILE_STAGES={int(args.profile_stages)}','-e',f'CAMEX_MUX_TIMEOUT={args.mux_timeout}']
    argv[2:2]=['-e',f'CAMEX_SYNC_START={int(args.sync_start)}']
    argv[2:2]=['-e',f'CAMEX_MOSAIC_FPS={args.mosaic_fps}']
    # Let this wrapper forward Ctrl+C once. Docker must retain its output pipe
    # until the GPU pipeline has flushed the MP4 and returned its final result.
    process=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
    done=threading.Event();started=time.monotonic()
    latest_resources={'gpu':None,'cpu':None}
    resource_stop={'reason':None}
    gpu_total=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).strip())
    def monitor():
        previous_cpu=None
        with (folder/'resources.jsonl').open('w') as out:
            while not done.is_set():
                try:
                    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,utilization.decoder,utilization.encoder,memory.used,power.draw,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
                    latest_resources['gpu']=[float(v.strip()) for v in gpu.split(',')]
                    memory=dict(line.split(':',1) for line in pathlib.Path('/proc/meminfo').read_text().splitlines())
                    available_mib=float(memory['MemAvailable'].split()[0])/1024
                    cpu_lines=pathlib.Path('/proc/stat').read_text().splitlines()
                    cores={l.split()[0]:list(map(int,l.split()[1:9])) for l in cpu_lines if l.startswith('cpu')}
                    if previous_cpu is not None:
                        delta=[b-a for a,b in zip(previous_cpu,cores['cpu'])];total=sum(delta)
                        if total>0:latest_resources['cpu']=100*(total-delta[3]-delta[4])/total
                    previous_cpu=cores['cpu']
                    out.write(json.dumps({'elapsed':time.monotonic()-started,'monotonic':time.monotonic(),'gpu_compute_decoder_encoder_MiB_W_C':gpu,'cpu':cpu_lines[0],'cpu_cores':cores,'mem_available_mib':available_mib})+'\n');out.flush()
                    reading=latest_resources['gpu']
                    if reading[3]>gpu_total-1024 or reading[5]>=85 or available_mib<6144:
                        resource_stop['reason']='GPU free-memory reserve fell below 1 GiB' if reading[3]>gpu_total-1024 else ('GPU temperature reached 85 C guard' if reading[5]>=85 else 'Host available RAM fell below 6 GiB')
                        (folder/'resource_stop.json').write_text(json.dumps(resource_stop))
                        print('RESOURCE STOP: '+resource_stop['reason'],flush=True)
                        abort()
                        if not done.wait(8):subprocess.run(['docker','exec',CONTAINER,'pkill','-KILL','-x','cap_interactive'],timeout=5)
                        break
                except Exception as error:out.write(json.dumps({'monitor_error':type(error).__name__})+'\n')
                done.wait(1)
    abort_sent=threading.Event()
    def abort(signum=None,frame=None):
        if abort_sent.is_set():return
        abort_sent.set()
        subprocess.run(['docker','exec',CONTAINER,'pkill','-INT','-x','cap_interactive'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=5)
    signal.signal(signal.SIGINT,abort);signal.signal(signal.SIGTERM,abort)
    def watchdog():
        if not done.wait(args.seconds+args.warmup+75):
            print('ERROR run exceeded its safety timeout; stopping this benchmark only.',flush=True);abort()
            if not done.wait(10):subprocess.run(['docker','exec',CONTAINER,'pkill','-KILL','-x','cap_interactive'],timeout=5)
    monitor_thread=threading.Thread(target=monitor,daemon=True)
    watchdog_thread=threading.Thread(target=watchdog,daemon=True)
    monitor_thread.start();watchdog_thread.start()
    with (folder/'run.log').open('w') as log:
        for line in process.stdout:
            log.write(line);log.flush()
            if line.startswith(('START','SOURCE START','READY','WARMUP','FPS ','FINISH','RESULT','ERROR','VIDEO','PREROLL')):
                current=latest_resources['gpu']
                if line.startswith(('FPS ','WARMUP')) and current:
                    cpu=latest_resources['cpu']
                    cpu_text=f' CPU={cpu:.1f}%' if cpu is not None else ''
                    print(line.rstrip()+f' | GPU={current[0]:.0f}% decode={current[1]:.0f}% encode={current[2]:.0f}% VRAM={current[3]:.0f} MiB temp={current[5]:.0f}C'+cpu_text,flush=True)
                else:print(line.rstrip(),flush=True)
    code=process.wait();done.set();monitor_thread.join(6);watchdog_thread.join(1)
    result_path=folder/'result.json'
    if result_path.exists():
        result=json.loads(result_path.read_text())
        result.update(dataset=args.dataset,unique_source_views=unique_views,repeated_inputs=args.n-unique_views,source_resolution=args.resolution,source_width=width,source_height=height,source_fps=args.source_fps,source_is_upscaled=upscaled)
        for c in result['per_camera']:
            assignment=assignments[c['id']-1]
            c.update(source_camera=assignment['camera_id'],replica=assignment['replica'],area=assignment['area'])
        minimum_target=args.analysis_fps*.95
        meets_target=(code==0 and not resource_stop['reason'] and not result['error'] and not result['interrupted']
                      and len(result['per_camera'])==args.n and result['minimum_fps']>=minimum_target
                      and 0<=result['p95_ms']<300
                      and all(c['frames']>0 and c['duplicates']==0 and c.get('invalid_latency',0)==0 for c in result['per_camera'])
                      and (args.track=='no' or all(c['track_observations']>0 for c in result['per_camera'])))
        result['target_check']={'passed':meets_target,'requested_stream_fps':args.analysis_fps,'minimum_stream_fps':minimum_target,'p95_limit_ms':300}
        result_path.write_text(json.dumps(result,indent=2))
        with (folder/'per_camera.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['id','source_camera','replica','area','frames','fps','detector_frames','detector_fps','person_detections','track_observations','unique_track_ids','unassigned_observations','duplicates','invalid_latency','latency_samples','paced_input_frames','sampled_out_frames','p95_ms']);w.writeheader();w.writerows(result['per_camera'])
        print(f'\nFINAL: {result["total_fps"]:.1f} {mode} FPS; detector stage {result["detector_total_fps"]:.1f} FPS; slowest stream {result["minimum_fps"]:.1f} FPS; P95 {result["p95_ms"]:.1f} ms.',flush=True)
        if args.track=='yes':print('Tracking observations and unique track-ID counts are in per_camera.csv; they are NOT tracking-accuracy or unique-person scores.',flush=True)
        if args.track=='yes':print(f'TRACK COVERAGE: {sum(c["track_observations"]>0 for c in result["per_camera"])}/{args.n} sources produced person tracks.',flush=True)
        target_label='PASS' if meets_target else ('INCOMPLETE' if code or result['error'] or result['interrupted'] else 'BELOW TARGET')
        print(f'TARGET CHECK: {target_label} (target {args.analysis_fps} FPS, each stream >={minimum_target:g} FPS, P95 <300 ms; short smoke test only).',flush=True)
    else:
        (folder/'failure.json').write_text(json.dumps({'exit_code':code,'measurement_unavailable':True,'reason':'The pipeline stopped before it could produce a complete result. See run.log.'},indent=2))
    if args.out=='yes':
        video=folder/'mosaic.mp4'
        if video.exists():print(f'VIDEO SAVED: {video}',flush=True)
        else:print('VIDEO NOT FINALIZED. See run.log; any .partial.mp4 is not verified evidence.',flush=True);code=code or 8
    print(f'RESULTS FOLDER: {folder}',flush=True)
    print(f'OPEN FOLDER: xdg-open "{folder}"',flush=True)
    if code:print(f'Run did not complete normally (exit {code}); inspect run.log.',flush=True)
    uid=int(os.environ.get('SUDO_UID',ROOT.stat().st_uid))
    gid=int(os.environ.get('SUDO_GID',ROOT.stat().st_gid))
    for path in folder.rglob('*'):
        if path.is_file():path.chmod(0o644);os.chown(path,uid,gid)
    folder.chmod(0o755);os.chown(folder,uid,gid);os.chown(folder.parent,uid,gid)
    return code

if __name__=='__main__':sys.exit(main())

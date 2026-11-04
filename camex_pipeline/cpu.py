"""CPU-only software decode, OpenVINO inference and ByteTrack reference."""
import collections
import json
import math
import os
import pathlib
import subprocess
import threading
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import av
import cv2
import numpy as np
import openvino as ov
import supervision as sv

from .common import JsonlWriter, percentile
from .postprocess import decode_person_output, letterbox_geometry

class CpuPipeline:
    def __init__(self, args, sources, folder, run_id):
        self.args, self.sources = args, sources
        self.folder, self.run_id = pathlib.Path(folder), run_id
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.epoch = None
        self.start = None
        self.error = None
        self.states = []
        self.writer = JsonlWriter(self.folder / 'metadata.jsonl')
        self.intervals = (self.folder / 'per_second.jsonl').open('w', buffering=1)
        self.video = None
        cv2.setNumThreads(1)
        for _ in sources:
            tracker = sv.ByteTrack(track_activation_threshold=args.threshold,
                lost_track_buffer=30, minimum_matching_threshold=.8,
                frame_rate=args.analysis_fps, minimum_consecutive_frames=1) if args.track == 'yes' else None
            self.states.append({'latest':None, 'busy':False, 'frames':0, 'reported':0,
                'decoded':0, 'dropped':0, 'tracks':set(), 'objects':[], 'preview':None,
                'latencies':[], 'fps':0.0, 'last_seen':None, 'reconnects':0, 'tracker':tracker,
                'last_processed':None, 'tracking_epoch':0})
        all_files = all(not s['uri'].startswith(('rtsp://','rtsps://')) for s in sources)
        self.barrier = threading.Barrier(len(sources), action=self._set_epoch) if all_files else None
        core = ov.Core()
        model = core.read_model(args.model)
        input = model.input(0)
        model.reshape({input: [1, 3, 640, 640]})
        config = {'PERFORMANCE_HINT':'LATENCY', 'INFERENCE_NUM_THREADS':args.threads,
                  'NUM_STREAMS':args.cpu_streams, 'INFERENCE_PRECISION_HINT':'f32',
                  'ENABLE_HYPER_THREADING':args.hyperthreads}
        compiled = core.compile_model(model, 'CPU', config)
        self.effective_config={k:str(compiled.get_property(k)) for k in ['INFERENCE_NUM_THREADS','NUM_STREAMS','INFERENCE_PRECISION_HINT','ENABLE_HYPER_THREADING']}
        self.execution_devices = list(compiled.get_property('EXECUTION_DEVICES'))
        if self.execution_devices != ['CPU']:
            raise RuntimeError(f'Expected CPU execution, got {self.execution_devices}')
        self.requests = ov.AsyncInferQueue(compiled, args.requests)
        self.requests.set_callback(self._inference_done)
        self.workers = [threading.Thread(target=self._decode, args=(i,), daemon=True) for i in range(len(sources))]

    def _set_epoch(self):
        self.epoch = time.monotonic() + .1
        self.start = self.epoch + self.args.warmup

    def _decode(self, i):
        source, state = self.sources[i], self.states[i]
        live = source['uri'].startswith(('rtsp://','rtsps://'))
        sequence, loop_offset, first_pts = 0, 0.0, None
        last_bucket, joined = -1, False
        while not self.stop.is_set():
            try:
                options = {'rtsp_transport':'tcp'} if live else {}
                with av.open(source['uri'], options=options, timeout=(5,2)) as container:
                    stream = container.streams.video[0]
                    stream.codec_context.thread_count = 1
                    while not self.stop.is_set():
                        last_time = 0.0
                        for frame in container.decode(stream):
                            if self.stop.is_set():
                                break
                            if (frame.width, frame.height) != (source['width'], source['height']):
                                raise ValueError(f"Unexpected dimensions on {source['camera_id']}")
                            raw_time = float(frame.pts * frame.time_base) if frame.pts is not None else sequence / source['fps']
                            if first_pts is None:
                                first_pts = raw_time
                            pts = loop_offset + raw_time - first_pts
                            last_time = pts
                            if not joined:
                                joined = True
                                if self.barrier:
                                    self.barrier.wait(timeout=20)
                                else:
                                    with self.lock:
                                        if self.epoch is None:
                                            self._set_epoch()
                            if not live:
                                self.stop.wait(max(0, self.epoch + pts - time.monotonic()))
                            measured = self.start is not None and self.start <= time.monotonic() < self.start + self.args.seconds
                            if measured:
                                state['decoded'] += 1
                            sequence += 1
                            bucket=int((pts+.00001)*self.args.analysis_fps)
                            if bucket<=last_bucket:
                                continue
                            last_bucket=bucket
                            ingest = time.monotonic()
                            w, h, x, y = letterbox_geometry(frame.width, frame.height)
                            resized = frame.reformat(width=w, height=h, format='rgb24').to_ndarray()
                            rgb = np.zeros((640,640,3), np.uint8)
                            rgb[y:y+h,x:x+w] = resized
                            packet = {'rgb':rgb, 'frame_id':sequence-1, 'source_pts_ns':int(pts*1e9),
                                      'ingest':ingest, 'width':frame.width, 'height':frame.height}
                            with self.lock:
                                if state['latest'] is not None and measured:
                                    state['dropped'] += 1
                                state['latest'] = packet
                                state['last_seen'] = ingest
                        if live or self.stop.is_set():
                            break
                        self.error='File ended before the requested run finished; use the looping RTSP replay helper'
                        self.stop.set()
                        break
                if live and not self.stop.is_set():
                    raise RuntimeError('RTSP decoder reached EOS')
            except Exception as e:
                if self.stop.is_set():
                    break
                if not live:
                    self.error = f"Decode failed for {source['camera_id']}: {type(e).__name__}"
                    self.stop.set()
                    break
                state['reconnects'] += 1
                self.stop.wait(.5)
                first_pts = None
                loop_offset = (time.monotonic() - self.epoch) if self.epoch else 0

    def _inference_done(self, request, userdata):
        i, packet, tensor = userdata
        state, source = self.states[i], self.sources[i]
        try:
            callback_time=time.monotonic()
            if state['last_processed'] is not None and callback_time-state['last_processed']>2:
                state['tracking_epoch']+=1
                if state['tracker'] is not None:state['tracker'].reset()
            state['last_processed']=callback_time
            boxes, scores = decode_person_output(request.get_output_tensor(0).data,
                packet['width'], packet['height'], self.args.threshold, self.args.nms)
            ids = [None] * len(boxes)
            if state['tracker'] is not None:
                tracked = state['tracker'].update_with_detections(sv.Detections(
                    xyxy=boxes, confidence=scores, class_id=np.zeros(len(boxes), dtype=int)))
                if tracked.tracker_id is not None:
                    for box, track_id in zip(tracked.xyxy, tracked.tracker_id):
                        distance = np.abs(boxes - box).sum(axis=1)
                        if len(distance) and distance.min() < .01:
                            ids[int(distance.argmin())] = str(int(track_id))
            objects = [{'class_id':0,'label':'person','confidence':float(score),
                        'bbox_xyxy':[float(v) for v in box], 'track_id':track_id}
                       for box, score, track_id in zip(boxes, scores, ids)]
            now = time.monotonic()
            latency = (now - packet['ingest']) * 1000
            with self.lock:
                state['preview'], state['objects'] = packet, objects
                measuring = self.start <= now < self.start + self.args.seconds and not self.stop.is_set()
                if measuring:
                    state['frames'] += 1
                    state['latencies'].append(latency)
                    state['tracks'].update(t for t in ids if t is not None)
            if measuring:
                self.writer.push({'schema_version':'1.0','type':'frame','run_id':self.run_id,
                    'backend':'openvino','camera_id':source['camera_id'], 'frame_id':packet['frame_id'],
                    'pts_ns':packet['source_pts_ns'], 'source_pts_ns':packet['source_pts_ns'],
                    'timestamp_domain':'media' if self.barrier else 'stream-relative',
                    'monotonic_s':now, 'elapsed_s':now-self.start,
                    'width':packet['width'],'height':packet['height'],'processing_ms':latency,
                    'tracking_epoch':state['tracking_epoch'],
                    'tracker':'bytetrack' if self.args.track=='yes' else 'none','objects':objects})
        except Exception as e:
            self.error = f'Inference callback failed: {type(e).__name__}: {e}'
            self.stop.set()
        finally:
            with self.lock:
                state['busy'] = False

    def _video_frame(self):
        n = len(self.sources)
        cols, rows = int(math.ceil(math.sqrt(n))), int(math.ceil(n / math.ceil(math.sqrt(n))))
        tile_w = min(640, 1920 // cols // 2 * 2)
        tile_h = tile_w * 9 // 16 // 2 * 2
        canvas = np.zeros((rows*tile_h, cols*tile_w, 3), np.uint8)
        for i,(source,state) in enumerate(zip(self.sources,self.states)):
            packet, objects = state['preview'], state['objects']
            if packet is None:
                continue
            w,h,x,y = letterbox_geometry(packet['width'], packet['height'])
            tile = cv2.resize(packet['rgb'][y:y+h,x:x+w,::-1], (tile_w,tile_h)).copy()
            for obj in objects:
                x1,y1,x2,y2 = obj['bbox_xyxy']
                track_id = obj['track_id']
                color = ((int(track_id or 0)*67)%200+55,(int(track_id or 0)*113)%200+55,(int(track_id or 0)*37)%200+55)
                p1=(int(x1*tile_w/source['width']),int(y1*tile_h/source['height']))
                p2=(int(x2*tile_w/source['width']),int(y2*tile_h/source['height']))
                cv2.rectangle(tile,p1,p2,color,2)
                if track_id:
                    cv2.putText(tile,'T'+track_id,p1,cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
            cv2.rectangle(tile,(0,0),(tile_w,42),(0,0,0),-1)
            cv2.putText(tile,f"{source['camera_id']}  {state['fps']:.1f} FPS  CPU",(8,27),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
            canvas[(i//cols)*tile_h:(i//cols+1)*tile_h,(i%cols)*tile_w:(i%cols+1)*tile_w]=tile
        if self.video is None:
            cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24',
                 '-s',f'{canvas.shape[1]}x{canvas.shape[0]}','-r',str(self.args.mosaic_fps),'-i','pipe:0',
                 '-an','-c:v','libx264','-preset','veryfast','-threads','2','-crf','24','-pix_fmt','yuv420p',
                 str(self.folder/'mosaic.mp4')]
            self.video=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=(self.folder/'encoder.log').open('w'))
        self.video.stdin.write(canvas.tobytes())

    def run(self):
        launched, reported, next_video = time.monotonic(), None, None
        for worker in self.workers:
            worker.start()
        try:
            cursor=0
            while not self.stop.is_set():
                now=time.monotonic()
                if getattr(self,'resource_monitor',None) and self.resource_monitor.guard_reason:
                    self.error=self.resource_monitor.guard_reason
                    break
                if self.start is None:
                    if now-launched>25:
                        self.error='Not every source initialized before the startup deadline'
                        break
                    time.sleep(.002)
                    continue
                if now>=self.start+self.args.seconds:
                    break
                for _ in self.states:
                    i=cursor%len(self.states);cursor+=1
                    if not self.requests.is_ready():
                        break
                    with self.lock:
                        state=self.states[i]
                        packet=state['latest'] if not state['busy'] else None
                        if packet is not None:
                            state['latest']=None;state['busy']=True
                    if packet is not None:
                        tensor=np.ascontiguousarray(packet['rgb'].transpose(2,0,1)[None],dtype=np.float32)/255.
                        self.requests.start_async({0:tensor},userdata=(i,packet,tensor),share_inputs=True)
                if now>=self.start:
                    if reported is None:
                        reported=self.start
                        next_video=self.start
                    if now-reported>=1:
                        duration=now-reported
                        with self.lock:
                            rates=[]
                            for state in self.states:
                                state['fps']=(state['frames']-state['reported'])/duration
                                state['reported']=state['frames'];rates.append(state['fps'])
                        event={'elapsed_s':now-self.start,'fps':rates,'total_fps':sum(rates)}
                        self.intervals.write(json.dumps(event)+'\n')
                        resource=self.resource_monitor.samples[-1] if getattr(self,'resource_monitor',None) and self.resource_monitor.samples else {}
                        print(f"FPS {event['elapsed_s']:.1f}s | total={sum(rates):.1f} | per-camera min/avg/max={min(rates):.1f}/{np.mean(rates):.1f}/{max(rates):.1f} | CPU={resource.get('cpu_percent',0):.1f}% GPU={resource.get('gpu_percent',0):.0f}%",flush=True)
                        reported=now
                    if self.args.out=='yes' and now>=next_video:
                        self._video_frame()
                        next_video+=1/self.args.mosaic_fps
                        next_video=max(next_video,now)
                time.sleep(.001)
        finally:
            measured_end = min(time.monotonic(), self.start + self.args.seconds) if self.start is not None else None
            self.stop.set()
            self.requests.wait_all()
            for worker in self.workers:
                worker.join(3)
            self.writer.close();self.intervals.close()
            if self.video:
                self.video.stdin.close()
                if self.video.wait(timeout=30):
                    self.error='Preview encoder failed'
        elapsed=max(0, measured_end-self.start) if measured_end is not None else 0
        cameras=[{'id':source['camera_id'],'frames':state['frames'],'fps':state['frames']/max(elapsed,.001),
                  'decoded_frames':state['decoded'],'dropped_analysis_frames':state['dropped'],
                  'unique_track_ids':len(state['tracks']),'p50_processing_ms':percentile(state['latencies'],50),
                  'p95_processing_ms':percentile(state['latencies'],95),'reconnects':state['reconnects']}
                 for source,state in zip(self.sources,self.states)]
        return {'backend':'openvino','execution_devices':self.execution_devices,'effective_config':self.effective_config,'tracker':'bytetrack' if self.args.track=='yes' else 'none',
                'measurement_start_monotonic':self.start,'seconds':elapsed,'error':self.error,
                'metadata_dropped':self.writer.dropped,'per_camera':cameras,'total_fps':sum(c['fps'] for c in cameras),
                'minimum_fps':min(c['fps'] for c in cameras)}

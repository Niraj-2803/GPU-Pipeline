import hashlib
import json
import math
import pathlib
import queue
import re
import subprocess
import threading
import time
from urllib.parse import urlsplit, urlunsplit

import numpy as np

CAMERA_ID = re.compile(r'^[A-Za-z0-9_-]{1,64}$')

def redact_uri(uri):
    p = urlsplit(uri)
    if p.scheme in ('rtsp', 'rtsps'):
        host = p.hostname or ''
        if ':' in host:
            host = f'[{host}]'
        return urlunsplit((p.scheme, host + (f':{p.port}' if p.port else ''), p.path, '', ''))
    return uri

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def cuda_ordinal(nvml_index):
    """Map nvidia-smi's index to CUDA's explicit PCI_BUS_ID enumeration."""
    result=subprocess.run(['nvidia-smi','--query-gpu=index,pci.bus_id','--format=csv,noheader'],
                          text=True,capture_output=True,check=True)
    rows=[(int(line.split(',')[0]),line.split(',')[1].strip().lower())
          for line in result.stdout.splitlines() if line.strip()]
    ordered=sorted(rows,key=lambda item:item[1])
    for ordinal,(index,_) in enumerate(ordered):
        if index==nvml_index:return ordinal
    raise ValueError(f'GPU index {nvml_index} does not exist')

def load_sources(path, count=None):
    path = pathlib.Path(path).resolve()
    document = json.loads(path.read_text())
    rows = document.get('sources', [])
    if not rows:
        raise ValueError('Source configuration must contain at least one source')
    prepared = []
    for row in rows:
        r = dict(row)
        if not CAMERA_ID.fullmatch(r.get('camera_id', '')):
            raise ValueError('camera_id must contain only letters, digits, underscore or hyphen')
        uri = r.get('uri', '')
        if not uri or any(c in uri for c in '\t\r\n'):
            raise ValueError('Invalid source URI')
        if urlsplit(uri).scheme not in ('rtsp', 'rtsps'):
            file = pathlib.Path(uri.removeprefix('file://'))
            if not file.is_absolute():
                file = path.parent / file
            file = file.resolve()
            if not file.is_file():
                raise ValueError(f'Missing video for {r["camera_id"]}')
            r['uri'] = str(file)
        for key in ('width', 'height', 'fps'):
            if not isinstance(r.get(key), (int, float)) or r[key] <= 0:
                raise ValueError(f'{key} must be a positive number for each source')
        if r['width'] % 2 or r['height'] % 2:
            raise ValueError('Frame dimensions must be even')
        prepared.append(r)
    if len({r['camera_id'] for r in prepared}) != len(prepared):
        raise ValueError('Configured camera IDs must be unique')
    count = len(prepared) if count is None else count
    if not 1 <= count <= 400:
        raise ValueError('Input count must be between 1 and 400')
    result = []
    for i in range(count):
        row = dict(prepared[i % len(prepared)])
        row['original_camera_id'] = row['camera_id']
        if i >= len(prepared):
            suffix=f'_r{i // len(prepared) + 1}'
            row['camera_id'] = row['camera_id'][:64-len(suffix)] + suffix
        row['stream_index'] = i
        result.append(row)
    if len({r['camera_id'] for r in result}) != len(result):
        raise ValueError('Expanded camera IDs collide; rename the configured cameras')
    return result

def percentile(values, q):
    return float(np.percentile(values, q)) if values else None

def validate_frame(event):
    if event.get('schema_version') != '1.0' or event.get('type') != 'frame':
        raise ValueError('Unsupported frame schema')
    for key in ('camera_id', 'run_id', 'frame_id', 'pts_ns', 'width', 'height', 'objects'):
        if key not in event:
            raise ValueError(f'Missing field: {key}')
    if not CAMERA_ID.fullmatch(event['camera_id']):
        raise ValueError('Invalid camera ID')
    for obj in event['objects']:
        box = obj['bbox_xyxy']
        if len(box) != 4 or not all(math.isfinite(float(x)) for x in box):
            raise ValueError('Invalid coordinates')
        if not (0 <= box[0] <= box[2] <= event['width'] + 1 and 0 <= box[1] <= box[3] <= event['height'] + 1):
            raise ValueError('Box lies outside native frame')
        if obj['track_id'] is not None and not isinstance(obj['track_id'], str):
            raise ValueError('Track IDs must be strings or null')

class JsonlWriter:
    def __init__(self, path, capacity=2048):
        self.path = pathlib.Path(path)
        self.queue = queue.Queue(capacity)
        self.dropped = 0
        self.error = None
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def push(self, event):
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            self.dropped += 1

    def _work(self):
        try:
            with self.path.open('w', buffering=1) as f:
                while True:
                    event = self.queue.get()
                    if event is None:
                        break
                    f.write(json.dumps(event, allow_nan=False) + '\n')
        except Exception as e:
            self.error = str(e)

    def close(self):
        self.queue.put(None, timeout=10)
        self.thread.join(10)
        if self.thread.is_alive() or self.error:
            raise RuntimeError(f'Metadata writer did not finish: {self.error}')

class ResourceMonitor:
    def __init__(self, folder, gpu_index):
        self.folder = pathlib.Path(folder)
        self.gpu_index = gpu_index
        self.stop = threading.Event()
        self.samples = []
        self.guard_reason = None
        self.thread = threading.Thread(target=self._work, daemon=True)

    def _work(self):
        import psutil
        psutil.cpu_percent()
        with (self.folder / 'resources.jsonl').open('w', buffering=1) as out:
            while not self.stop.is_set():
                m = psutil.virtual_memory()
                row = {'monotonic_s': time.monotonic(), 'cpu_percent': psutil.cpu_percent(),
                       'ram_used_mib': m.used / 2**20, 'ram_available_mib': m.available / 2**20}
                try:
                    r = subprocess.run(['nvidia-smi', '-i', str(self.gpu_index),
                        '--query-gpu=utilization.gpu,utilization.decoder,utilization.encoder,memory.used,temperature.gpu,power.draw',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
                    if r.returncode == 0:
                        vals = r.stdout.strip().split(',')
                        for k, v in zip(['gpu_percent','decoder_percent','encoder_percent','vram_mib','temperature_c','power_w'], vals):
                            row[k] = float(v.strip())
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    row['gpu_monitor_unavailable'] = True
                self.samples.append(row)
                if row.get('temperature_c',0)>=85:
                    self.guard_reason = 'GPU temperature reached the 85 C software guard'
                if row['ram_available_mib']<2048:
                    self.guard_reason = 'Available system RAM fell below 2 GiB'
                out.write(json.dumps(row) + '\n')
                self.stop.wait(1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(5)

    def summary(self, start=0, end=math.inf):
        data = [r for r in self.samples if start <= r['monotonic_s'] <= end]
        keys = ['cpu_percent','ram_used_mib','ram_available_mib','gpu_percent','decoder_percent','encoder_percent','vram_mib','temperature_c','power_w']
        return {k: {'mean': float(np.mean(v)), 'max': max(v), 'min': min(v)}
                for k in keys if (v := [r[k] for r in data if k in r])}

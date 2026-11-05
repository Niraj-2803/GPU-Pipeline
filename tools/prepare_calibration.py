#!/usr/bin/env python3
"""Prepare the fixed 640-pixel, ten-frames-per-view TensorRT calibration set."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camex_pipeline.common import load_sources

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--views', type=int, default=50)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    sources = load_sources(a.sources)
    if not 1 <= a.views <= len(sources) or not 1 <= a.workers <= 8:
        p.error('views must fit the supplied sources; workers must be 1..8')
    sources = sources[:a.views]
    if any(s['uri'].startswith(('rtsp://', 'rtsps://')) or s['fps'] != 10 for s in sources):
        p.error('Use local, encoded 10-FPS clips for this reproducible calibration recipe')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    def prepare(item):
        index, source = item
        target = a.output_dir / f'C{index:02d}.rgb'
        command = ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-threads', '1', '-filter_threads', '1',
                   '-i', source['uri'], '-vf',
                   "select='not(mod(n,40))',scale=640:640:force_original_aspect_ratio=decrease,pad=640:640:(ow-iw)/2:(oh-ih)/2:black",
                   '-frames:v', '10', '-vsync', '0', '-pix_fmt', 'rgb24', '-f', 'rawvideo', str(target)]
        subprocess.run(command, check=True)
        if target.stat().st_size != 10 * 640 * 640 * 3:
            raise RuntimeError(f'Need ten calibration frames at 0,4,...,36 seconds: {source["camera_id"]}')
        print(source['camera_id'], flush=True)
        return {'file': target.name, 'camera_id': source['camera_id'],
                'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        records = list(pool.map(prepare, enumerate(sources, 1)))
    manifest = {'views': len(sources), 'frames_per_view': 10, 'frames': len(sources) * 10,
                'seconds': list(range(0, 40, 4)), 'size': 640, 'layout': 'RGB uint8 interleaved, zero letterbox',
                'files': records}
    (a.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(a.output_dir.resolve())

if __name__ == '__main__': main()

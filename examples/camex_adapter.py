#!/usr/bin/env python3
"""Transform either backend's frames into requests for CAMEX's existing projection layer."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camex_pipeline.common import validate_frame

def to_projection_requests(frame):
    validate_frame(frame)
    requests = []
    for obj in frame['objects']:
        x1, y1, x2, y2 = obj['bbox_xyxy']
        identity = None if obj['track_id'] is None else [frame['run_id'], frame['camera_id'],
                                                       frame.get('tracking_epoch', 0), obj['track_id']]
        requests.append({'camera_id': frame['camera_id'], 'frame_id': frame['frame_id'],
                         'source_pts_ns': frame['source_pts_ns'], 'timestamp_domain': frame['timestamp_domain'],
                         'track_key': identity, 'bbox_native_px': [x1, y1, x2, y2],
                         'footpoint_native_px': [(x1 + x2) / 2, y2],
                         'confidence': None if obj['confidence'] < 0 else obj['confidence']})
    return requests

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path); p.add_argument('--follow', action='store_true')
    a = p.parse_args()
    with a.input.open() as stream:
        while True:
            position = stream.tell(); line = stream.readline()
            if not line or not line.endswith('\n'):
                if not a.follow:
                    if line: raise ValueError('Incomplete final metadata line')
                    break
                stream.seek(position); time.sleep(.05); continue
            for request in to_projection_requests(json.loads(line)):
                print(json.dumps(request, allow_nan=False), flush=True)

if __name__ == '__main__': main()

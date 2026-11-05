#!/usr/bin/env python3
"""Minimal CAMEX consumer. Replace on_frame with the existing analytics call."""
import argparse
import json
import pathlib
import time

def on_frame(event):
    camera=event['camera_id']
    # A track is scoped to one camera and run, never an identity across cameras.
    tracks=[(event['run_id'],camera,event.get('tracking_epoch',0),o['track_id']) for o in event['objects'] if o['track_id'] is not None]
    print(camera,event['frame_id'],len(event['objects']),'detections',len(tracks),'tracked',flush=True)

p=argparse.ArgumentParser()
p.add_argument('file',type=pathlib.Path)
p.add_argument('--follow',action='store_true')
a=p.parse_args()
with a.file.open() as f:
    while True:
        position=f.tell();line=f.readline()
        if not line:
            if not a.follow:break
            time.sleep(.05);continue
        if not line.endswith('\n'):
            if not a.follow:
                raise ValueError('Incomplete final metadata line')
            f.seek(position);time.sleep(.05);continue
        event=json.loads(line)
        if event.get('schema_version')!='1.0':raise ValueError('Unsupported schema')
        on_frame(event)

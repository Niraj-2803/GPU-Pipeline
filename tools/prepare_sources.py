#!/usr/bin/env python3
"""Convert the existing airport50 manifest into the portable source schema."""
import argparse
import json
import pathlib

p = argparse.ArgumentParser()
p.add_argument('--root', type=pathlib.Path, required=True)
p.add_argument('--manifest', type=pathlib.Path, required=True)
p.add_argument('--profile', required=True)
p.add_argument('--output', type=pathlib.Path, required=True)
a = p.parse_args()
d = json.loads(a.manifest.read_text())['profiles'][a.profile]
out = {'schema_version':'1.0','note':'Synthetic airport footage. Repeated inputs are independent decoders, not distinct live cameras.',
       'sources':[{'camera_id':r['camera_id'],'uri':str((a.root/r['path']).resolve()),
       'width':d['width'],'height':d['height'],'fps':d['fps'],'is_upscaled':d['upscaled'],
       'sha256':r.get('sha256')} for r in d['sources']]}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(out,indent=2))
print(a.output)

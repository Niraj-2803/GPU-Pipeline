#!/usr/bin/env python3
"""Record non-secret hardware/runtime information for a benchmark report."""
import argparse
import datetime
import importlib.metadata
import json
import pathlib
import platform
import subprocess

p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
def run(args):
    r=subprocess.run(args,capture_output=True,text=True)
    return r.stdout.strip() if r.returncode==0 else None
cpu=json.loads(run(['lscpu','-J']))['lscpu']
keep=['Architecture:','CPU(s):','Vendor ID:','Model name:','Thread(s) per core:','Core(s) per socket:','Socket(s):']
data={'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'hostname':platform.node(),'platform':platform.platform(),
      'cpu':{r['field'].rstrip(':'):r['data'] for r in cpu if r['field'] in keep},
      'gpu':run(['nvidia-smi','--query-gpu=index,name,pci.bus_id,memory.total,driver_version','--format=csv']),
      'memory':run(['free','-b']),'packages':{}}
for name in ['openvino','numpy','av','opencv-python','supervision','nncf','psutil']:
    try:data['packages'][name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:pass
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(data,indent=2));print(a.output)

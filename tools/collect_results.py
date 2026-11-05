#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib

p=argparse.ArgumentParser();p.add_argument('root',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
rows=[]
for file in sorted(a.root.rglob('summary.json')):
    s=json.loads(file.read_text());resource=s.get('resources',{})
    def val(key,stat='mean'):return resource.get(key,{}).get(stat)
    rows.append({'run':s['run_id'],'path':str(file.parent),'backend':s['backend'],'precision':s['precision'],'tracker':s['tracker'],
        'source_count':s['source_count'],'source_kind':s['source_kind'],'seconds':s['seconds'],
        'analysis_target_fps':s['analysis_target_fps'],'recording':s['recording'],'total_fps':s['total_fps'],
        'minimum_fps':s['minimum_fps'],'p95_processing_ms':s['p95_processing_ms'],'cpu_percent':val('cpu_percent'),
        'gpu_percent':val('gpu_percent'),'decoder_percent':val('decoder_percent'),
        'vram_peak_mib':val('vram_mib','max'),'temperature_peak_c':val('temperature_c','max'),
        'throughput_check':s['throughput_check'],'metadata_dropped':s['metadata_dropped'],'error':s['error']})
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.with_suffix('.json').write_text(json.dumps(rows,indent=2))
with a.output.with_suffix('.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
print('Collected',len(rows),'runs')

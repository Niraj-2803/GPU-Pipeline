import csv
import json
from pathlib import Path
import shutil
from experiments import resources

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'delivery';OUT.mkdir(exist_ok=True)
(OUT/'raw').mkdir(exist_ok=True)
rows=[]
for path in sorted((ROOT/'results').glob('*.json')):
    row=json.loads(path.read_text())
    if not row.get('result'):continue
    rows.append(row)
    folder=Path(row['folder'])
    for name in ['settings.json','result.json','per_camera.csv','per_second.jsonl','resources.jsonl','source_assignments.json','stage_profile.json','run.log']:
        if (folder/name).exists():shutil.copy2(folder/name,OUT/'raw'/(row['name']+'.'+name))
    shutil.copy2(ROOT/'logs'/(row['name']+'.log'),OUT/'raw'/(row['name']+'.console.log'))
    if row['recording']:shutil.copy2(folder/'mosaic.mp4',OUT/(row['name']+'.mp4'))
summary=[]
for row in rows:
    r=row['result']
    summary.append({'test':row['name'],'inputs':row['inputs'],'measurement_seconds':r['seconds'],'warmup_seconds':row['warmup'],
                    'synchronized_start':r.get('synchronized_file_start'), 'decoder_startup_seconds':r.get('decoder_startup_seconds'),
                    'recording':row['recording'],'total_fps':r['total_fps'],'avg_fps_per_input':r['total_fps']/row['inputs'],
                    'minimum_fps':r['minimum_fps'],'p95_ms':r['p95_ms'],'passed':r['target_check']['passed'],
                    **row['resources'],'intentional_dropped_frames':sum(c['sampled_out_frames'] for c in r['per_camera'])})
(OUT/'all_results.json').write_text(json.dumps(rows,indent=2))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
with (OUT/'Results.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
old=Path('/media/dezzex/4tb_ssd/camex-capacity-20261015')
diagnostics=[]
for folder in sorted((old/'runs').glob('20261015T145*_n150_*')):
    r=json.loads((folder/'result.json').read_text())
    diagnostics.append({'folder':str(folder),'result':r,'resources':resources(folder,r)})
(OUT/'previous_startup_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
print(json.dumps(summary,indent=2))

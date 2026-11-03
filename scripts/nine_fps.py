import json
import pathlib
import statistics
import subprocess

ROOT=pathlib.Path(__file__).resolve().parents[1]


def trial(name,n=200,seconds=10,warmup=8,video=False,analysis_fps=9):
    before=set((ROOT/'runs').iterdir())
    command=[str(ROOT/'run.sh'),'-n',str(n),'-res','2K','--source-fps','10','--codec','h265','--precision','int8-head','-track','yes',
             '-out','yes' if video else 'no','--analysis-fps',str(analysis_fps),'--warmup',str(warmup),'--seconds',str(seconds)]
    print('BEGIN',name,flush=True)
    with (ROOT/'logs'/(name+'.log')).open('w') as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:
            log.write(line);log.flush()
            if line.startswith(('FPS ','FINAL','TARGET','TRACK COVERAGE','RESOURCE','SAMPLING','ERROR')):print(line.rstrip(),flush=True)
        code=process.wait()
    paths=set((ROOT/'runs').iterdir())-before
    folder=max(paths,key=lambda p:p.stat().st_mtime)
    result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else {}
    row={'name':name,'inputs':n,'analysis_fps':analysis_fps,'seconds':seconds,'warmup':warmup,'recording':video,'command':command,'folder':str(folder),'exit_code':code,'result':result}
    samples=[json.loads(l) for l in (folder/'resources.jsonl').read_text().splitlines()]
    start=result.get('measurement_start_monotonic',0)
    measured=[s for s in samples if start<=s.get('monotonic',0)<start+result.get('seconds',0) and 'gpu_compute_decoder_encoder_MiB_W_C' in s]
    if measured:
        gpu=[list(map(float,s['gpu_compute_decoder_encoder_MiB_W_C'].split(','))) for s in measured]
        row['resources']={'gpu_compute_avg':statistics.mean(s[0] for s in gpu),'gpu_compute_peak':max(s[0] for s in gpu),
                          'decoder_avg':statistics.mean(s[1] for s in gpu),'decoder_peak':max(s[1] for s in gpu),'encoder_avg':statistics.mean(s[2] for s in gpu),
                          'vram_peak_mib':max(s[3] for s in gpu),'temperature_peak':max(s[5] for s in gpu),'ram_available_min_mib':min(s['mem_available_mib'] for s in measured)}
        loads={}
        for a,b in zip(measured,measured[1:]):
            for core,values in a.get('cpu_cores',{}).items():
                later=b['cpu_cores'][core];delta=[y-x for x,y in zip(values,later)];total=sum(delta)
                if total>0:loads.setdefault(core,[]).append(100*(total-delta[3]-delta[4])/total)
        if loads:
            row['resources']['cpu_total_avg']=statistics.mean(loads['cpu'])
            percore={c:statistics.mean(v) for c,v in loads.items() if c!='cpu'}
            core=max(percore,key=percore.get);row['resources']['busiest_core']=core;row['resources']['busiest_core_avg']=percore[core]
    print('SUMMARY',json.dumps({k:v for k,v in row.items() if k not in ['result','command']}),flush=True)
    return row


if __name__=='__main__':
    dest=ROOT/'results/nine_fps.json'
    if dest.exists():raise RuntimeError('Previous nine-FPS results already exist; preserve them')
    rows=[]
    for name,n,seconds,warm,video in [('sampling_sanity',2,3,2,False),('nine_fps_200_a',200,10,8,False),('nine_fps_200_b',200,10,8,False),('nine_fps_200_video',200,10,8,True)]:
        row=trial(name,n,seconds,warm,video);rows.append(row);dest.write_text(json.dumps(rows,indent=2))
        if row['exit_code']:raise RuntimeError('Pipeline failed; see result before continuing')
        if name=='sampling_sanity':
            assert row['result']['target_check']['passed']
            assert all(c['sampled_out_frames']>=2 and c['invalid_latency']==0 for c in row['result']['per_camera'])
    print('NINE_FPS_TESTS_COMPLETE',flush=True)

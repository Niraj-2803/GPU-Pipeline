import json
import pathlib
import statistics
import subprocess
import sys

ROOT=pathlib.Path(__file__).resolve().parents[1]
CASES={
    '150_sync':(150,8,['--sync-start'],False),
    '200_sync':(200,8,['--sync-start'],False),
    '200_sync_profile':(200,8,['--sync-start','--profile-stages'],False),
    '200_sync_b16':(200,8,['--sync-start','--batch','16'],False),
    '200_sync_groups4':(200,8,['--sync-start','--tracker-groups','4'],False),
    '200_sync_maxperf':(200,8,['--sync-start','--tracker-preset','max_perf'],False),
    '200_sync_mux960':(200,8,['--sync-start','--mux-size','960'],False),
    '200_sync_detectonly':(200,8,['--sync-start','-track','no'],False),
    '150_sync_confirm30':(150,8,['--sync-start','--seconds','30'],False),
    '200_sync_confirm30':(200,8,['--sync-start','--seconds','30'],False),
    '200_sync_video':(200,8,['--sync-start'],True),
    '200_sync_video5':(200,8,['--sync-start','--mosaic-fps','5'],True),
    '200_sync_video5_warm':(200,8,['--sync-start','--mosaic-fps','5'],True),
    '150_warm30':(150,30,[],False),
    '150_preroll':(150,8,['--preroll'],False),
    '200_preroll':(200,8,['--preroll'],False),
    '200_warm30':(200,30,[],False),
    '200_batch16':(200,30,['--batch','16'],False),
    '200_groups4':(200,30,['--tracker-groups','4'],False),
    '200_mux960':(200,30,['--mux-size','960'],False),
    '200_maxperf':(200,30,['--tracker-preset','max_perf'],False),
    '200_detectonly':(200,30,['-track','no'],False),
    '200_profile':(200,8,['--profile-stages'],False),
}


def resources(folder,result):
    samples=[json.loads(l) for l in (folder/'resources.jsonl').read_text().splitlines()]
    start=result.get('measurement_start_monotonic',0)
    measured=[s for s in samples if start<=s.get('monotonic',0)<start+result.get('seconds',0) and 'gpu_compute_decoder_encoder_MiB_W_C' in s]
    if not measured:return {}
    gpu=[list(map(float,s['gpu_compute_decoder_encoder_MiB_W_C'].split(','))) for s in measured]
    data={'gpu_compute_avg':statistics.mean(s[0] for s in gpu),'gpu_compute_peak':max(s[0] for s in gpu),
          'decoder_avg':statistics.mean(s[1] for s in gpu),'decoder_peak':max(s[1] for s in gpu),
          'encoder_avg':statistics.mean(s[2] for s in gpu),'vram_peak_mib':max(s[3] for s in gpu),
          'temperature_peak':max(s[5] for s in gpu),'ram_available_min_mib':min(s['mem_available_mib'] for s in measured)}
    loads={}
    for a,b in zip(measured,measured[1:]):
        for core,values in a.get('cpu_cores',{}).items():
            delta=[y-x for x,y in zip(values,b['cpu_cores'][core])];total=sum(delta)
            if total>0:loads.setdefault(core,[]).append(100*(total-delta[3]-delta[4])/total)
    if loads:
        data['cpu_total_avg']=statistics.mean(loads['cpu'])
        percore={c:statistics.mean(v) for c,v in loads.items() if c!='cpu'}
        core=max(percore,key=percore.get);data.update(busiest_core=core,busiest_core_avg=percore[core])
    return data


def trial(name,n,warmup,extra,video=False,seconds=10):
    dest=ROOT/'results'/(name+'.json')
    if dest.exists():raise RuntimeError('Preserve existing result: '+str(dest))
    before=set((ROOT/'runs').iterdir())
    command=[str(ROOT/'run.sh'),'-n',str(n),'-res','2K','--source-fps','10','--codec','h265',
             '--precision','int8-head','-track','yes','-out','yes' if video else 'no',
             '--analysis-fps','10','--warmup',str(warmup),'--seconds',str(seconds),*extra]
    print('BEGIN',name,flush=True)
    with (ROOT/'logs'/(name+'.log')).open('w') as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:
            log.write(line);log.flush()
            if line.startswith(('FPS ','FINAL','TARGET','TRACK COVERAGE','RESOURCE','ERROR','PREROLL')):print(line.rstrip(),flush=True)
        code=process.wait()
    paths=set((ROOT/'runs').iterdir())-before
    folder=max(paths,key=lambda p:p.stat().st_mtime)
    result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else {}
    row={'name':name,'inputs':n,'warmup':warmup,'recording':video,'command':command,'folder':str(folder),
         'exit_code':code,'result':result,'resources':resources(folder,result)}
    dest.write_text(json.dumps(row,indent=2))
    summary={k:v for k,v in row.items() if k not in ['result','command']}
    summary['performance']={k:result.get(k) for k in ['total_fps','minimum_fps','p95_ms','target_check']}
    print('SUMMARY',json.dumps(summary),flush=True)
    return row


if __name__=='__main__':
    for name in sys.argv[1:]:
        n,warm,extra,video=CASES[name]
        row=trial(name,n,warm,extra,video)
        if row['exit_code']:raise SystemExit(row['exit_code'])

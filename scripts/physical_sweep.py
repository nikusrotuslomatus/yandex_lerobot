"""Small deterministic contact-only sensitivity series, never an attachment demo."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from active_scan.core import ROOT,load_config,write_json
OUT=ROOT/'results/physical_sweep'


def run_case(case):
    cfg=load_config();cfg['seed']=42
    for group,key,value in case['changes']:cfg[group][key]=value
    out=OUT/case['name'];out.mkdir(parents=True,exist_ok=True)
    config=out/'input.json';write_json(config,cfg)
    cmd=[sys.executable,'-m','active_scan.cli','--config',str(config),'--level','C','--pick-only','--out',str(out)]
    import os
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')
    p=subprocess.run(cmd,env=env,cwd=ROOT,capture_output=True,text=True,timeout=300)
    (out/'console.txt').write_text(p.stdout+p.stderr,encoding='utf-8')
    if p.returncode:raise RuntimeError(p.stderr)
    result=json.loads((out/'summary.json').read_text())
    events=[json.loads(x) for x in (out/'events.jsonl').read_text().splitlines()]
    verification=next((e for e in events if e['kind']=='grasp_verification'),None)
    row={'case':case['name'],'changes':case['changes'],'success':result['success'],'failure':result['failure'],'verification':verification,'weld_active_final':result['weld_active_final']}
    print(case['name'],row['success'],row['failure'],flush=True)
    return row


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT)
    OUT=parser.parse_args().out.resolve()
    cases=[{'name':f'mass_{mass:.2f}_friction_{mu:.1f}','changes':[('scene','cube_mass',mass),('physics','finger_friction',mu)]}
           for mass in [.08,.12,.16] for mu in [.4,.7]]
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(run_case,cases))
    write_json(OUT/'summary.json',{'scope':'six deterministic pick-and-hold sensitivity cases; not independent statistical trials or full episodes','cases':rows,'successes':sum(x['success'] for x in rows),'total':len(rows)})

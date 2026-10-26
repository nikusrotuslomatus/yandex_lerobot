"""Sequential warm-library repetitions, including small initial object offsets."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from active_scan.core import load_config,write_json
from active_scan.simulation import SimulatorAdapter
from active_scan.control import TaskPlanner


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/factory.json')
    p.add_argument('--out',required=True,type=Path);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);rows=[]
    # These are scenario initial conditions, set before any task motion.
    offsets=[(0,0),(0,0),(0,0),(.001,0),(-.001,.001),(0,-.001)]
    for i,(dx,dy) in enumerate(offsets):
        cfg=load_config(args.config);start=time.perf_counter();sim=SimulatorAdapter(cfg,'B')
        pose=sim.cube_pose();pose[:2,3]+=[dx,dy];sim._set_cube(sim.data,pose)
        folder=args.out/f'run_{i:02d}';result=TaskPlanner(sim,folder).run()
        frames=[json.loads(f.read_text()) for f in sorted(folder.glob('frame_*.json'))]
        row={key:result[key] for key in ['success','failure','simulated_seconds','planning_seconds','max_planning_call_seconds','tracking_max','captures','planning_library','coverage']}
        row.update(initial_offset_m=[dx,dy],total_process_wall_seconds=time.perf_counter()-start,
                   first_capture_sim_seconds=frames[0]['timestamp'] if frames else None,
                   first_capture_planning_seconds=sum(c['seconds'] for c in result['planning_calls'] if frames and c['simulation_time']<frames[0]['timestamp']),
                   min_visible_pixels=min((f['frame_quality']['visible_pixels'] for f in frames),default=0),
                   min_visible_fraction=min((f['frame_quality']['visible_fraction'] for f in frames),default=0))
        rows.append(row);write_json(args.out/'progress.json',rows)
        print(json.dumps({'run':i,'success':row['success'],'planning_seconds':row['planning_seconds'],'failure':row['failure']}),flush=True)
    values=[r['planning_seconds'] for r in rows]
    report={'runs':rows,'successes':sum(r['success'] for r in rows),'total':len(rows),
            'planning_min_seconds':min(values),'planning_median_seconds':float(np.median(values)),
            'planning_max_seconds':max(values),
            'note':'Six MuJoCo B scenarios are a regression check, not hardware reliability qualification.'}
    write_json(args.out/'BENCHMARK.json',report)
    if report['successes']!=report['total']:raise SystemExit(1)


if __name__=='__main__':main()

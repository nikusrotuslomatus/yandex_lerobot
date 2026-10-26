"""Matched ten-frame geometric baselines and surface discretization check."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from active_scan.core import ROOT,load_config,write_json
from active_scan.simulation import SimulatorAdapter
from active_scan.geometry import CollisionChecker,VisibilityEvaluator,GraspPlanner
from active_scan.planning import ViewPlanner,ViewCandidate


def run(geometry=None,out=None):
    cfg=load_config();s=SimulatorAdapter(cfg);cc=CollisionChecker(s);v=VisibilityEvaluator(s)
    grasps,_=GraspPlanner(s,cc).geometric_candidates();vp=ViewPlanner(s,cc,v)
    cache=np.load(Path(geometry or ROOT/'results/geometry_final')/'candidates.npz')
    candidates={}
    for g in grasps:
        candidates[g.grasp_id]=[ViewCandidate(q,None,None,None,quality,None,0,0) for q,quality in zip(cache[g.grasp_id+'_q'],cache[g.grasp_id+'_quality'])]
    single=vp.select_views(candidates[grasps[0].grasp_id],max_views=10)
    first=vp.select_views(candidates[grasps[0].grasp_id],max_views=5)
    initial=np.max([c.quality for c in first],axis=0)
    second=vp.select_views(candidates[grasps[1].grasp_id],initial,max_views=5)
    fixed=[]
    for flex in [-.7,-.3]:
        for roll in np.linspace(-2,2,5):
            q=np.array(cfg['robot']['home']);q[3:]=[flex,roll]
            d=s.query_state(q,grasps[0].T_G_O,gripper=grasps[0].gripper)
            if not cc.check(d,'HELD')[0]:fixed.append(q)
    series={'fixed_script':[(grasps[0],q) for q in fixed],
            'optimized_single':[(grasps[0],c.q) for c in single],
            'optimized_two_5_plus_5':[(grasps[0],c.q) for c in first]+[(grasps[1],c.q) for c in second]}
    rows={}
    for grid in [12,24]:
        s.cfg['visibility']['grid']=grid;ev=VisibilityEvaluator(s)
        for name,visits in series.items():
            cov=ev.coverage()
            for g,q in visits:
                d=s.query_state(q,g.T_G_O,gripper=g.gripper);quality,directions,_=ev.evaluate(d);cov.update(quality,directions)
            rows[f'{name}_grid_{grid}']={'frames':len(visits),'metrics':cov.metrics(ev.faces)}
    write_json(out or ROOT/'results/view_comparison.json',{'scope':'geometric snapshots only; ten-frame cap; two-grasp method excludes manipulation cost; fixed poses are not optimized; grid24 changes both sampling density and pixel-scale quality','cases':rows})
    s.close();print(rows)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--geometry');parser.add_argument('--out')
    args=parser.parse_args();run(args.geometry,args.out)

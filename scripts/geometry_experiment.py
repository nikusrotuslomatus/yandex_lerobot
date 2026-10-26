"""Mandatory experiment: executable before FSM/contact control exists."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from PIL import Image
from active_scan.core import load_config,write_json,inverse
from active_scan.simulation import SimulatorAdapter
from active_scan.geometry import CollisionChecker,VisibilityEvaluator,GraspPlanner
from active_scan.planning import ViewPlanner


def run(cfg,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError(f'Existing geometry results preserved: {out}. Choose a new --out.')
    s=SimulatorAdapter(cfg,'A');cc=CollisionChecker(s);v=VisibilityEvaluator(s);gp=GraspPlanner(s,cc);planner=ViewPlanner(s,cc,v)
    write_json(out/'config.json',cfg);write_json(out/'model_inventory.json',s.inventory())
    write_json(out/'aperture.json',gp.aperture_audit())
    grasps,fit=gp.geometric_candidates();write_json(out/'grasp_fit.json',fit)
    reports={};all_candidates={};selections={};arrays={};union=np.zeros(len(v.local))
    for g in grasps:
        candidates,stats=planner.generate_candidates(g)
        all_candidates[g.grasp_id]=candidates
        selected=planner.select_views(candidates)
        selections[g.grasp_id]=selected
        cov=v.coverage()
        for c in selected:cov.update(c.quality,c.directions)
        potential=np.max([c.quality for c in candidates],axis=0) if candidates else np.zeros(len(v.local))
        reports[g.grasp_id]={'sampling':stats,'selected':len(selected),'coverage':cov.metrics(v.faces),'sampled_union_binary':float(np.mean(potential>=cfg['visibility']['quality_threshold'])),
                            'T_G_O':g.T_G_O,'gripper':g.gripper}
        arrays[g.grasp_id+'_q']=np.array([c.q for c in candidates]);arrays[g.grasp_id+'_quality']=np.array([c.quality for c in candidates])
        print(g.grasp_id,stats,'selected',len(selected),cov.metrics(v.faces)['binary_coverage'],flush=True)
        for i,c in enumerate(selected):
            s.data.qpos[s.qadr]=c.q;s.data.qpos[s.gq]=g.gripper;s._set_cube(s.data,c.T_W_O)
            rgb,depth,seg=s.capture();prefix=out/f'{g.grasp_id}_view_{i:02d}'
            Image.fromarray(rgb).save(str(prefix)+'.png')
            np.savez_compressed(str(prefix)+'.npz',depth=depth,segmentation=seg,visibility=c.quality)
            write_json(str(prefix)+'.json',{'level':'A','geometric_snapshot':True,'timestamp':float(s.data.time),'q':c.q,'grasp':g.grasp_id,'T_W_O':c.T_W_O,'T_W_G':c.T_W_G,'T_W_C':s.camera_pose(),'T_C_O':c.T_C_O,'K':s.K,'clearance':c.clearance})
        s.reset()
    pairs=[]
    for i,g1 in enumerate(grasps):
        for g2 in grasps[i+1:]:
            cov=v.coverage()
            first=selections[g1.grasp_id]
            for c in first:cov.update(c.quality,c.directions)
            second=planner.select_views(all_candidates[g2.grasp_id],cov.quality)
            for c in second:cov.update(c.quality,c.directions)
            pairs.append({'grasps':[g1.grasp_id,g2.grasp_id],'views':[len(first),len(second)],'coverage':cov.metrics(v.faces),
                          'executable_regrasp_validated':False})
    np.savez_compressed(out/'candidates.npz',**arrays)
    result={'level':'A','grasps':reports,'pairs':pairs,
            'scope':'Finite-sample geometric view potential. No proof of global maximum, path connectivity, stable grip or executable regrasp.',
            'flip_required':'undetermined_until_place_and_regrasp_feasibility'}
    write_json(out/'summary.json',result);s.close()
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--out',default='results/geometry_seed42');p.add_argument('--candidates',type=int)
    a=p.parse_args();cfg=load_config(a.config)
    if a.candidates:cfg['planning']['candidates']=a.candidates
    run(cfg,a.out)

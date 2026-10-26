"""Distinguish observability/masking problems from stochastic depth noise."""
import argparse
from pathlib import Path
import hashlib
import numpy as np
import mujoco as mj
from active_scan.core import load_config,write_json
from active_scan.simulation import SimulatorAdapter
from active_scan.perception.sim_source import MujocoRGBDSource
from active_scan.perception.pipeline import PerceptionPipeline


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    states_path=Path('results/factory_video/recorded_states.npz')
    with np.load(states_path,allow_pickle=False) as d:
        timeline=d['timeline'];positions=d['qpos'];velocities=d['qvel'];equality=d['eq_active']
    indices=np.flatnonzero(np.diff(timeline[:,1],prepend=timeline[0,1])>0)
    cfg=load_config('results/factory_video/episode/config.json')
    cfg['camera']['width']=640;cfg['camera']['height']=480
    focal=640/(2*np.tan(np.deg2rad(75)/2));cfg['camera']['fovy']=float(np.rad2deg(2*np.arctan(480/(2*focal))))
    cases=[]
    for name,mask_enabled,bound in (('noiseless_no_robot_mask_diagnostic',False,0.),
                                    ('noiseless_exact_joint_CAD_mask',True,0.),
                                    ('noiseless_4mm_robot_mask_envelope',True,.004)):
        sim=SimulatorAdapter(cfg,'A');source=MujocoRGBDSource(sim,noise=0,holes=0,mask_robot=mask_enabled,link_position_bound_m=bound)
        pipeline=PerceptionPipeline(source.calibration);views=[]
        try:
            for index in indices:
                for j in range(index-2,index+1):
                    sim.data.qpos[:]=positions[j];sim.data.qvel[:]=velocities[j];sim.data.eq_active[:]=equality[j]
                    sim.data.time=float(timeline[j,0]);mj.mj_forward(sim.model,sim.data)
                    result=pipeline.process(source.read())
                views.append({'view':int(timeline[index,1]),'status':result.status,'metrics':result.metrics})
        finally:sim.close()
        cases.append({'name':name,'tracked_saved_views':sum(v['status']=='TRACKING' for v in views),'views':views})
    write_json(out,{'cases':cases,'input_sha256':hashlib.sha256(states_path.read_bytes()).hexdigest(),
                   'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'scope':'Noiseless recorded-B replay; no-mask is an oracle diagnostic, never a deployment mode',
                   'robot_acceptance':False})
    print([(c['name'],c['tracked_saved_views']) for c in cases])


if __name__=='__main__':main()

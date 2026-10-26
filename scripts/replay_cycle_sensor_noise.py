"""Perception preflight on recorded factory-cycle states, not robot acceptance.

Replay B uses the historical ideal weld. Poses only construct/evaluate the
rendered world; the localizer receives raw RGB-D and a noisy-FK robot mask.
"""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import mujoco as mj
import numpy as np
from PIL import Image,ImageDraw
from active_scan.core import load_config,write_json,Failure
from active_scan.simulation import SimulatorAdapter
from active_scan.perception.sim_source import MujocoRGBDSource
from active_scan.perception.sensor_noise import scenario_profile
from active_scan.perception.feedback_noise import JointNoiseProfile
from active_scan.perception.pipeline import PerceptionPipeline
from active_scan.perception.uncertainty import ErrorBudget
from active_scan.perception.localizer import symmetry_distance


def main():
    p=argparse.ArgumentParser();p.add_argument('--states',default='results/factory_video/recorded_states.npz')
    p.add_argument('--config',default='results/factory_video/episode/config.json');p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    source_paths=list(Path('src/active_scan/perception').glob('*.py'))+[Path(__file__)]
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_paths}
    cfg=load_config(args.config)
    # Qualify against an approximate native 640x480 D455 FOV, with the actual
    # historical camera mounting. Real SDK K is still required on the bench.
    cfg['camera']['width']=640;cfg['camera']['height']=480
    f=640/(2*np.tan(np.deg2rad(75)/2));cfg['camera']['fovy']=float(np.rad2deg(2*np.arctan(480/(2*f))))
    sim=SimulatorAdapter(cfg,'A');source=MujocoRGBDSource(sim,noise_profile=scenario_profile('nominal'),
        joint_noise_profile=JointNoiseProfile(),link_position_bound_m=.004,seed=98765)
    pipeline=PerceptionPipeline(source.calibration,error_budget=ErrorBudget(source.calibration.calibration_id))
    with np.load(args.states,allow_pickle=False) as states:
        timeline=states['timeline'];positions=states['qpos'];velocities=states['qvel'];equality=states['eq_active']
    if positions.shape[1]!=sim.model.nq or velocities.shape[1]!=sim.model.nv:raise ValueError('Recording and robot model differ')
    capture_indices=np.flatnonzero(np.diff(timeline[:,1],prepend=timeline[0,1])>0)
    selected=sorted({j for i in capture_indices for j in range(max(0,i-4),min(len(timeline),i+4))})
    entries=[];last=-1;refused=0;start=perf_counter()
    try:
        for index in selected:
            # Reconstruct encoder hysteresis from all intermediate recorded
            # joint positions, not only the sparse observation windows.
            for j in range(last+1,index):
                q=np.r_[positions[j,sim.qadr],positions[j,sim.gq]]
                source.feedback_noise.measure(q,float(timeline[j,0]))
            last=index
            sim.data.qpos[:]=positions[index];sim.data.qvel[:]=velocities[index]
            sim.data.eq_active[:]=equality[index];sim.data.time=float(timeline[index,0]);mj.mj_forward(sim.model,sim.data)
            truth=sim.cube_pose().copy()
            with patch.object(sim,'cube_pose',side_effect=AssertionError('truth leaked into sensing')):
                frame=source.read();estimate=pipeline.process(frame,now=frame.arrival_time)
            item={'record_index':index,'timestamp':sim.data.time,'capture_counter':int(timeline[index,1]),
                  'phase':int(timeline[index,3]),'held_by_ideal_weld':bool(equality[index].any()),
                  'status':estimate.status,'metrics':estimate.metrics,'robot_mask_pixels':int(frame.robot_mask.sum())}
            if estimate.measured:
                item.update(position_error_mm=float(np.linalg.norm(estimate.pose[:3,3]-truth[:3,3])*1000),
                            rotation_error_deg=float(np.rad2deg(symmetry_distance(estimate.pose[:3,:3],truth[:3,:3]))))
            try:pipeline.camera_observer(require_hardware=False).observe(frame.arrival_time)
            except Failure:refused+=1
            if index in capture_indices:
                rgb=np.array(frame.rgb,copy=True)
                if estimate.mask is not None:
                    rgb[estimate.mask]=(rgb[estimate.mask]*.5+np.array([0,120,0])).astype(np.uint8)
                image=Image.fromarray(rgb);draw=ImageDraw.Draw(image);draw.rectangle((0,0,640,30),fill=(10,16,24))
                draw.text((8,8),f"saved view {int(timeline[index,1])}: {estimate.status.value}",fill='white')
                image.save(out/f'view_{int(timeline[index,1]):02d}.png')
            entries.append(item)
            print(f"record t={sim.data.time:.2f}: {estimate.status.value}",flush=True)
    finally:sim.close()
    views=[e for e in entries if e['record_index'] in capture_indices]
    unchanged=all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    result={'input_states':args.states,'input_states_sha256':hashlib.sha256(Path(args.states).read_bytes()).hexdigest(),
            'source_sha256':hashes,'source_unchanged':unchanged,'renderer_config':cfg,'frames':len(entries),
            'statuses':dict(Counter(e['status'] for e in entries)),'saved_views':views,
            'tracked_saved_views':sum(e['status']=='TRACKING' for e in views),
            'control_requests_refused':refused,'unqualified_control_always_refused':refused==len(entries),
            'view_library_qualified_under_this_sensor_scenario':len(views)>0 and all(e['status']=='TRACKING' for e in views),
            'validation_scope':'Recorded ideal-weld B poses re-rendered with a noisy camera and noisy encoders; no fresh manipulation run',
            'hardware_accuracy_validated':False,'robot_acceptance':False,'wall_seconds':perf_counter()-start,'detail':entries}
    write_json(out/'summary.json',result)
    print(f"Tracked saved views: {result['tracked_saved_views']}/{len(views)}",flush=True)
    if not unchanged or refused!=len(entries):raise SystemExit(1)


if __name__=='__main__':main()

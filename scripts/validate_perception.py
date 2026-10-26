"""Reproducible numerical campaign with truth exclusively in the evaluator.

This is not a physical robot acceptance or a model of all D455 errors.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
from pathlib import Path
import platform
from time import perf_counter
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
from active_scan.core import transform,write_json,load_config
from active_scan.perception.synthetic import test_calibration,render
from active_scan.perception.pipeline import PerceptionPipeline
from active_scan.perception.localizer import symmetry_distance
from active_scan.perception.recording import Recorder
from active_scan.perception.mapping import ObjectMapper
from active_scan.perception.realsense import probe


def quantiles(values):
    if not values:return None
    return dict(zip(('p50','p95','p99','max'),np.r_[np.percentile(values,[50,95,99]),max(values)].tolist()))


def overlay(path,frame,estimate):
    rgb=np.array(frame.rgb,copy=True)
    if estimate.mask is not None:
        rgb[estimate.mask]=(rgb[estimate.mask].astype(float)*.5+np.array([0,255,100])*.5).astype(np.uint8)
    image=Image.fromarray(rgb).resize((848,480));draw=ImageDraw.Draw(image)
    draw.rectangle((0,0,848,36),fill=(12,16,24))
    draw.text((10,10),f'{estimate.source}: {estimate.status.value}; {estimate.metrics["processing_ms"]:.1f} ms',fill='white')
    image.save(path)


def evaluate(frames,truths,pipeline,out,mapper=None,recorder=None):
    entries=[];maps=0
    for i,(frame,truth) in enumerate(zip(frames,truths)):
        result=pipeline.process(frame)
        entry={'sequence':frame.sequence,'status':result.status,'processing_ms':result.metrics['processing_ms'],
               'metrics':result.metrics}
        if result.measured:
            entry['translation_error_mm']=float(np.linalg.norm(result.pose[:3,3]-truth[:3,3])*1000)
            entry['rotation_error_deg_mod_cube_symmetry']=float(np.rad2deg(symmetry_distance(result.pose[:3,:3],truth[:3,:3])))
            entry['estimated_pose']=result.pose
        if mapper is not None:maps+=int(mapper.add(frame,result))
        if recorder is not None:recorder.append(frame)
        if i in (0,len(truths)//2,len(truths)-1):overlay(out/f'frame_{i:04d}.png',frame,result)
        entries.append(entry)
    timings=[e['processing_ms'] for e in entries]
    errors=[e['translation_error_mm'] for e in entries if 'translation_error_mm' in e]
    angles=[e['rotation_error_deg_mod_cube_symmetry'] for e in entries if 'translation_error_mm' in e]
    statuses=dict(Counter(e['status'] for e in entries))
    report={'frames':len(entries),'statuses':statuses,'tracking_fraction':len(errors)/len(entries),
            'processing_ms':quantiles(timings),'translation_error_mm':quantiles(errors),
            'rotation_error_deg_mod_cube_symmetry':quantiles(angles),'keyframes':maps,'frames_detail':entries}
    if mapper is not None and mapper.keyframes:
        mapper.save(out/'object_map.ply');cloud=np.asarray(mapper.cloud().points)
        report['map']={'points':len(cloud),'extent_mm':(np.ptp(cloud,axis=0)*1000).tolist(),
                       'shell_error_mm_with_60mm_pose_prior':quantiles((np.abs(np.max(np.abs(cloud),axis=1)-.03)*1000).tolist()),
                       'independent_size_metrology':False}
    write_json(out/'metrics.json',report)
    # Never store evaluator truth inside the sensor episode.
    write_json(out/'evaluation_truth.json',truths)
    return {k:v for k,v in report.items() if k!='frames_detail'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    parser.add_argument('--frames',type=int,default=180);parser.add_argument('--mujoco',action='store_true')
    parser.add_argument('--record',action='store_true');args=parser.parse_args()
    if args.frames<30:raise ValueError('Campaign requires at least 30 frames')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);start=perf_counter()
    c=test_calibration();truths=[]
    for i in range(args.frames):
        phase=i/(args.frames-1)
        lift=float(np.clip(min(phase/.15,(1-phase)/.15),0,1))
        scan=float(np.clip((phase-.15)/.7,0,1))
        rpy=[.25*np.sin(2*np.pi*scan)*lift,.35*np.sin(4*np.pi*scan)*lift,2*np.pi*scan]
        pose=transform(Rotation.from_euler('xyz',rpy).as_matrix(),[.26+.01*np.sin(2*np.pi*phase),.01,.031+.08*lift])
        truths.append(pose)
    frames=(render(c,pose,i,i/30,noise=.0009,holes=.12,outliers=.01,seed=1000+i)[0] for i,pose in enumerate(truths))
    synthetic=out/'synthetic';synthetic.mkdir()
    recorder=Recorder(synthetic/'sensor_episode',c) if args.record else None
    report={'synthetic':evaluate(frames,truths,PerceptionPipeline(c),synthetic,ObjectMapper(c),recorder)}
    if args.mujoco:
        from active_scan.simulation import SimulatorAdapter
        from active_scan.perception.sim_source import MujocoRGBDSource
        cfg=load_config('configs/factory.json')
        cfg['scene']['camera_position']=[.65,-.35,.48];cfg['scene']['camera_target']=[.24,-.04,.08]
        sim=SimulatorAdapter(cfg,'A');source=MujocoRGBDSource(sim)
        frames=[];truths=[]
        try:
            for i in range(60):
                pose=transform(Rotation.from_euler('xyz',[.15,.12,.5*np.sin(i/60*2*np.pi)]).as_matrix(),
                               [.25,-.1,.08+.01*np.sin(i/60*2*np.pi)])
                sim._set_cube(sim.data,pose);sim.data.time=i/30
                frames.append(source.read());truths.append(pose)
        finally:sim.close()
        mujoco_out=out/'mujoco';mujoco_out.mkdir();write_json(mujoco_out/'renderer_config.json',cfg)
        report['mujoco']=evaluate(frames,truths,PerceptionPipeline(source.calibration),mujoco_out)
    report['environment']={'python':platform.python_version(),'platform':platform.platform(),
                            'versions':{p:importlib.metadata.version(p) for p in ('numpy','scipy','mujoco','open3d','opencv-python-headless','pyrealsense2')},
                            'hardware_devices':probe(),'seed_policy':'1000+frame_index; MuJoCo source 42',
                            'code_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src/active_scan/perception').glob('*.py')}}
    report.update(wall_seconds=perf_counter()-start,hardware_accuracy_validated=False,robot_acceptance=False,
                  latency_scope='CPU localization only; excludes USB, sensor exposure, rendering and map fusion')
    # Accuracy gates apply to accepted poses; availability is checked separately.
    checks={name:{'availability':r['tracking_fraction']>=.85,
                  'translation':r['translation_error_mm'] is not None and r['translation_error_mm']['p95']<3.,
                  'rotation':r['rotation_error_deg_mod_cube_symmetry'] is not None and r['rotation_error_deg_mod_cube_symmetry']['p95']<3.}
            for name,r in report.items() if name in ('synthetic','mujoco')}
    report['software_gates']=checks;report['software_pass']=all(all(v.values()) for v in checks.values())
    write_json(out/'summary.json',report)
    print(__import__('json').dumps(report,indent=2))
    if not report['software_pass']:raise SystemExit(1)


if __name__=='__main__':main()

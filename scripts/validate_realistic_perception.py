"""Seeded sensor/rig/feedback uncertainty campaign; no physical acceptance.

No parameters are fitted to evaluator poses. Shared episode bias, independent
reference capture, unseen seeds, validity and control refusal are all reported.
"""
import argparse
from collections import Counter
from dataclasses import replace,asdict
import hashlib
import importlib.metadata
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.spatial.transform import Rotation
from active_scan.core import transform,write_json,load_config,Failure
from active_scan.perception.synthetic import d455_fixture_calibration,render_stereo
from active_scan.perception.sensor_noise import StereoNoiseModel,scenario_profile
from active_scan.perception.feedback_noise import JointNoiseProfile
from active_scan.perception.pipeline import PerceptionPipeline
from active_scan.perception.scene import StaticScene
from active_scan.perception.uncertainty import ErrorBudget
from active_scan.perception.localizer import symmetry_distance
from active_scan.perception.types import PoseEstimate,Status
from active_scan.perception.realsense import probe
from active_scan.perception.recording import Recorder


def q(values):
    if not values:return None
    return dict(zip(('p50','p95','p99','max'),np.r_[np.percentile(values,[50,95,99]),max(values)].tolist()))


def trajectory(t):
    return transform(Rotation.from_euler('xyz',[.2+.06*np.sin(t*2),.3+.05*np.cos(t*2),.15*np.sin(t*3)]).as_matrix(),
                     [.26+.008*np.sin(t*2),.01,.085+.008*np.sin(t*2)])


def evaluate_case(frames,truths,pipeline,reference_failure=None):
    entries=[];refusals=0
    for (frame,diagnostics),truth in zip(frames,truths):
        if reference_failure:
            estimate=PoseEstimate(Status.CALIBRATION_INVALID,frame.capture_time,frame.calibration_id,frame.sequence,
                                  frame.source,metrics={'reason':reference_failure,'processing_ms':0.})
            pipeline.latest=estimate
        else:estimate=pipeline.process(frame,now=frame.arrival_time)
        item={'sequence':frame.sequence,'status':estimate.status,'metrics':estimate.metrics,
              'generation_ms':diagnostics.get('generation_ms'),
              'valid_depth_fraction':float(np.mean(frame.depth_m>0))}
        if estimate.measured:
            position=float(np.linalg.norm(estimate.pose[:3,3]-truth[:3,3]));angle=symmetry_distance(estimate.pose[:3,:3],truth[:3,:3])
            item.update(position_error_mm=position*1000,rotation_error_deg=float(np.rad2deg(angle)),
                        position_inside_conditional_envelope=position<=estimate.metrics['position_error_bound_m'],
                        angle_inside_conditional_envelope=angle<=estimate.metrics['angular_error_bound_rad'])
            item['estimated_pose']=estimate.pose
            item['evaluator_truth_pose']=truth
        try:pipeline.camera_observer(require_hardware=False).observe(frame.arrival_time)
        except Failure:refusals+=1
        entries.append(item)
    positions=[i['position_error_mm'] for i in entries if 'position_error_mm' in i]
    rotations=[i['rotation_error_deg'] for i in entries if 'position_error_mm' in i]
    accepted=[i for i in entries if 'position_error_mm' in i]
    return {'frames':len(entries),'statuses':dict(Counter(e['status'] for e in entries)),
            'tracking_fraction':len(accepted)/len(entries),'position_error_mm':q(positions),'rotation_error_deg':q(rotations),
            'processing_ms':q([e['metrics']['processing_ms'] for e in entries]),
            'generation_ms_excluded_from_processing':q([e['generation_ms'] for e in entries if e['generation_ms'] is not None]),
            'position_envelope_coverage':sum(i['position_inside_conditional_envelope'] for i in accepted)/len(accepted) if accepted else None,
            'angle_envelope_coverage':sum(i['angle_inside_conditional_envelope'] for i in accepted)/len(accepted) if accepted else None,
            'control_refusals':refusals,'all_control_requests_refused':refusals==len(entries),'detail':entries}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    parser.add_argument('--seeds',type=int,default=6);parser.add_argument('--frames',type=int,default=24)
    parser.add_argument('--mujoco',action='store_true');parser.add_argument('--record-first',action='store_true')
    parser.add_argument('--nominal-profile');args=parser.parse_args()
    if args.seeds<2 or args.frames<8:raise ValueError('Use >=2 seeds and >=8 frames')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);start=perf_counter();cases=[]
    source_paths=list(Path('src/active_scan/perception').glob('*.py'))+[Path(__file__),Path('src/active_scan/grasp.py')]
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    custom_profile=None
    if args.nominal_profile:
        from active_scan.perception.sensor_noise import StereoNoiseProfile
        raw=__import__('json').loads(Path(args.nominal_profile).read_text(encoding='utf-8'))
        custom_profile=StereoNoiseProfile.from_dict(raw.get('noise_profile',raw))
    for kind in ('nominal','adverse','spec_limit','timing_fault'):
        for seed_index in range(args.seeds):
            seed=10001+seed_index*997;c=d455_fixture_calibration()
            # Different true camera distances, within a fixed calibrated cell.
            target=np.array([.26,.01,.085]);T=c.T_B_C.copy();factor=(.85,1.,1.15)[seed_index%3]
            T[:3,3]=target+(T[:3,3]-target)*factor;c=replace(c,T_B_C=T)
            profile=custom_profile if kind=='nominal' and custom_profile is not None else scenario_profile(kind)
            model=StereoNoiseModel(c,profile,seed)
            empty=[render_stereo(c,model,trajectory,i,.1+i/30,include_cube=False)[0] for i in range(10)]
            recorder=None
            if args.record_first and kind=='nominal' and seed_index==0:
                recorder=Recorder(out/'recorded_nominal',c)
                for frame in empty:recorder.append(frame)
            scene=None;failure=None
            try:scene=StaticScene.capture_empty(c,empty,cell_confirmed_empty=True)
            except ValueError as exc:failure=str(exc)
            budget=ErrorBudget(c.calibration_id)
            if recorder is not None:
                write_json(out/'recorded_nominal'/'error_budget.json',budget.to_dict())
                if scene is not None:scene.save(out/'recorded_nominal'/'empty_scene.npz')
            pipeline=PerceptionPipeline(c,static_scene=scene,error_budget=budget)
            times=[.1+(10+i)/30 for i in range(args.frames)]
            def images():
                for i,t in enumerate(times):
                    tick=perf_counter();frame,diag=render_stereo(c,model,trajectory,10+i,t)
                    diag['generation_ms']=(perf_counter()-tick)*1000
                    if recorder is not None:recorder.append(frame)
                    yield frame,diag
            report=evaluate_case(images(),[trajectory(t) for t in times],pipeline,failure)
            report.update(kind=kind,seed=seed,calibration=c.to_dict(),noise_profile=profile.to_dict(),
                          error_budget=budget.to_dict(),scene_reference_failure=failure,reference_frames=10)
            write_json(out/f'{kind}_{seed}.json',report);cases.append({k:v for k,v in report.items() if k!='detail'})
            print(f"{kind} seed={seed}: {report['tracking_fraction']:.1%} tracking; control refused {report['control_refusals']}/{args.frames}",flush=True)
    if args.mujoco:
        from active_scan.simulation import SimulatorAdapter
        from active_scan.perception.sim_source import MujocoRGBDSource
        cfg=load_config('configs/factory.json');cfg['scene']['camera_position']=[.65,-.35,.48];cfg['scene']['camera_target']=[.24,-.04,.08]
        sim=SimulatorAdapter(cfg,'A');source=MujocoRGBDSource(sim,noise_profile=scenario_profile('nominal'),
            joint_noise_profile=JointNoiseProfile(),link_position_bound_m=.004,seed=654321)
        pipeline=PerceptionPipeline(source.calibration,error_budget=ErrorBudget(source.calibration.calibration_id))
        truths=[]
        for i in range(args.frames):
            t=1+i/30;truths.append(transform(Rotation.from_euler('xyz',[.15,.12,.2*np.sin(t)]).as_matrix(),[.25,-.10,.08]))
        def images_mj():
            for i,pose in enumerate(truths):
                sim._set_cube(sim.data,pose);sim.data.time=1+i/30
                tick=perf_counter();frame=source.read()
                yield frame,{'generation_ms':(perf_counter()-tick)*1000}
        try:report=evaluate_case(images_mj(),truths,pipeline)
        finally:sim.close()
        report.update(kind='mujoco_nominal',seed=654321,calibration=source.calibration.to_dict(),
                      noise_profile=source.stereo.profile.to_dict(),joint_noise_profile=asdict(JointNoiseProfile()),
                      robot_mask_link_position_bound_m=.004)
        write_json(out/'mujoco_nominal.json',report);cases.append({k:v for k,v in report.items() if k!='detail'})
    nominal=[case for case in cases if case['kind']=='nominal']
    # Declared software envelope, not physical robot acceptance. Adverse cases
    # may reject every frame. Falsely enabling physical control is never allowed.
    gates={'nominal_tracking_at_least_85pct':all(c['tracking_fraction']>=.85 for c in nominal),
           'nominal_p95_position_under_6mm':all(c['position_error_mm'] is not None and c['position_error_mm']['p95']<6 for c in nominal),
           'nominal_p95_angle_under_5deg':all(c['rotation_error_deg'] is not None and c['rotation_error_deg']['p95']<5 for c in nominal),
           'conditional_position_envelope':all(c['position_envelope_coverage'] is None or c['position_envelope_coverage']>=.95 for c in cases),
           'unqualified_control_always_refused':all(c['all_control_requests_refused'] for c in cases)}
    gates['source_unchanged_during_run']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in source_hashes.items())
    summary={'cases':cases,'software_gates':gates,'software_pass':all(gates.values()),'wall_seconds':perf_counter()-start,
             'hardware_devices':probe(),'hardware_accuracy_validated':False,'robot_acceptance':False,
             'design_qualified_for_4mm_slip_detection':False,
             'environment':{'python':platform.python_version(),'platform':platform.platform(),
                'packages':{p:importlib.metadata.version(p) for p in ('numpy','scipy','mujoco','open3d','opencv-python-headless','pyrealsense2')}},
             'parameter_status':'physics-informed engineering scenarios; no D455 serial/profile was fitted',
             'latency_scope':'CPU localization; excludes noise generator, sensor, USB, rendering, disk and fusion',
             'source_sha256':source_hashes}
    write_json(out/'summary.json',summary)
    print(__import__('json').dumps(gates,indent=2),flush=True)
    if not summary['software_pass']:raise SystemExit(1)


if __name__=='__main__':main()

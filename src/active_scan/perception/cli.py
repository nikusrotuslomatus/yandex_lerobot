"""Operational entry points; no network or robot-motion commands."""
import argparse
import json
from pathlib import Path
from time import monotonic,sleep
import numpy as np
from ..core import write_json
from .calibration import load_calibration, calibrate_camera_to_base
from .types import Calibration
from .pipeline import PerceptionPipeline
from .recording import Recorder, Replay


def estimate_dict(e):
    return {k:getattr(e,k) for k in ('status','timestamp','calibration_id','sequence','source','pose','covariance',
                                    'object_id','grasp_epoch','symmetry_ambiguous','metrics')}


def replay_episode(path,out,map_enabled=False,budget_path=None,scene_path=None):
    reader=Replay(path)
    from .uncertainty import ErrorBudget
    budget=ErrorBudget.load(budget_path,reader.calibration) if budget_path else (ErrorBudget(reader.calibration.calibration_id) if reader.calibration.provenance=='hardware' else None)
    from .scene import StaticScene
    scene=StaticScene.load(scene_path,reader.calibration) if scene_path else None
    pipeline=PerceptionPipeline(reader.calibration,error_budget=budget,static_scene=scene);estimates=[];maps={}
    if map_enabled:
        from .mapping import ObjectMapper
    for frame in reader:
        estimate=pipeline.process(frame);estimates.append(estimate_dict(estimate))
        if map_enabled and estimate.measured:
            epoch=estimate.grasp_epoch
            if epoch not in maps: maps[epoch]=ObjectMapper(reader.calibration)
            maps[epoch].add(frame,estimate)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    for epoch,mapper in maps.items():
        if mapper.keyframes: mapper.save(out/f'cube_epoch_{epoch}.ply')
    times=[e['metrics']['processing_ms'] for e in estimates]
    report={'frames':len(estimates),'measured':sum(e['status']=='TRACKING' for e in estimates),
            'processing_ms':dict(zip(('p50','p95','p99'),np.percentile(times,[50,95,99]).tolist())) if times else {},
            'source':'recorded_sensor_data','hardware_accuracy_validated':False,'robot_acceptance':False,
            'estimates':estimates}
    write_json(out/'localization.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description='Fixed D455 scene/cube localization')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('probe')
    cal=sub.add_parser('calibrate');cal.add_argument('--input',required=True);cal.add_argument('--out',required=True)
    rec=sub.add_parser('record');rec.add_argument('--calibration',required=True);rec.add_argument('--out',required=True)
    rec.add_argument('--seconds',type=float,default=10);rec.add_argument('--fps',type=int,default=30)
    rec.add_argument('--bag-input');rec.add_argument('--bag-output')
    rep=sub.add_parser('replay');rep.add_argument('episode');rep.add_argument('--out',required=True);rep.add_argument('--map',action='store_true')
    scene=sub.add_parser('scene');scene.add_argument('episode');scene.add_argument('--out',required=True)
    scene.add_argument('--confirm-empty-cell',action='store_true')
    live=sub.add_parser('track');live.add_argument('--calibration',required=True);live.add_argument('--out',required=True)
    live.add_argument('--seconds',type=float,default=10);live.add_argument('--scene')
    live.add_argument('--error-budget')
    rep.add_argument('--error-budget')
    rep.add_argument('--scene')
    fit=sub.add_parser('fit-noise');fit.add_argument('--manifest',required=True);fit.add_argument('--out',required=True)
    budget=sub.add_parser('budget-template');budget.add_argument('--calibration',required=True);budget.add_argument('--out',required=True)
    args=parser.parse_args()
    if args.command=='probe':
        from .realsense import probe
        print(json.dumps({'devices':probe(),'hardware_accuracy_validated':False},indent=2));return
    if args.command=='fit-noise':
        from .noise_fit import fit_noise_manifest
        if Path(args.out).exists():raise FileExistsError(args.out)
        report=fit_noise_manifest(args.manifest);write_json(args.out,report)
        print(json.dumps({'qualified_for_control':report['qualified_for_control'],'heldout_bias_residual_m':report['heldout_bias_residual_m']}));return
    if args.command=='budget-template':
        from .uncertainty import ErrorBudget
        if Path(args.out).exists():raise FileExistsError(args.out)
        calibration=load_calibration(args.calibration)
        write_json(args.out,ErrorBudget(calibration.calibration_id).to_dict());return
    if args.command=='calibrate':
        data=json.loads(Path(args.input).read_text(encoding='utf-8'))
        pose,metrics=calibrate_camera_to_base(data['points_B'],data['pixels'],np.array(data['profile']['K']))
        calibration=Calibration(**{**data['profile'],'T_B_C':pose})
        target=Path(args.out)
        if target.exists():raise FileExistsError(target)
        write_json(target,calibration.to_dict());write_json(target.with_suffix('.quality.json'),metrics)
        print(json.dumps(metrics));return
    if args.command=='scene':
        from .scene import StaticScene
        if Path(args.out).exists():raise FileExistsError(args.out)
        replay=Replay(args.episode)
        from itertools import islice
        reference=StaticScene.capture_empty(replay.calibration,islice(replay,30),cell_confirmed_empty=args.confirm_empty_cell)
        reference.save(args.out);print(json.dumps({'table_error_m':reference.table_error_m}));return
    if args.command=='track':
        from .realsense import D455Source
        from .runtime import LocalizationRuntime
        from .scene import StaticScene
        if args.seconds<=0:raise ValueError('Tracking duration must be positive')
        if Path(args.out).exists():raise FileExistsError(args.out)
        c=load_calibration(args.calibration);reference=StaticScene.load(args.scene,c) if args.scene else None
        from .uncertainty import ErrorBudget
        budget=ErrorBudget.load(args.error_budget,c) if args.error_budget else ErrorBudget(c.calibration_id)
        estimates=[];sequence=-1;start=monotonic()
        with LocalizationRuntime(D455Source(c),static_scene=reference,error_budget=budget) as runtime:
            while monotonic()-start<args.seconds:
                if runtime.error is not None or runtime.capture.error is not None:raise RuntimeError('Perception worker failed')
                estimate=runtime.pipeline.latest
                if estimate is not None and estimate.sequence!=sequence:
                    estimates.append(estimate_dict(estimate));sequence=estimate.sequence
                sleep(.005)
        write_json(args.out,{'estimates':estimates,'processed':runtime.processed,'dropped':runtime.capture.buffer.dropped,
                             'robot_control_enabled':False,'robot_acceptance':False,'hardware_accuracy_validated':False})
        return
    if args.command=='replay':
        result=replay_episode(args.episode,args.out,args.map,args.error_budget,args.scene)
        print(json.dumps({k:v for k,v in result.items() if k!='estimates'},indent=2));return
    if args.command=='record':
        from .realsense import D455Source
        from .runtime import CaptureWorker
        c=load_calibration(args.calibration)
        if args.seconds<=0:raise ValueError('Recording duration must be positive')
        recorder=Recorder(args.out,c);start=monotonic();count=0
        with D455Source(c,args.fps,args.bag_input,args.bag_output) as source:
            worker=CaptureWorker(source).start()
            try:
                while monotonic()-start<args.seconds:
                    frame=worker.take()
                    if frame is not None:recorder.append(frame);count+=1
            finally:worker.close()
        write_json(Path(args.out)/'capture_summary.json',{'frames':count,'dropped_before_recording':worker.buffer.dropped,
                                                        'wall_seconds':monotonic()-start,'robot_acceptance':False})


if __name__=='__main__':main()

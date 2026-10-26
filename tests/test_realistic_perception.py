from dataclasses import replace
import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from active_scan.core import transform,load_config,Failure
from active_scan.perception.synthetic import test_calibration as make_calibration,render_stereo,render
from active_scan.perception.sensor_noise import StereoNoiseModel,scenario_profile
from active_scan.perception.scene import StaticScene
from active_scan.perception.pipeline import PerceptionPipeline
from active_scan.perception.uncertainty import ErrorBudget
from active_scan.perception.types import Calibration,FramePacket,Status
from active_scan.perception.recording import Recorder,Replay
from active_scan.perception.noise_fit import fit_noise_manifest


@pytest.mark.parametrize('seed',[42,73,101,151])
def test_correlated_stereo_scene_pose_and_uncertainty(seed):
    c=make_calibration();model=StereoNoiseModel(c,scenario_profile('nominal'),seed)
    truth=transform(Rotation.from_euler('xyz',[.2,.3,.1]).as_matrix(),[.26,.01,.08])
    empty=[render_stereo(c,model,lambda t:truth,i,.1+i/30,include_cube=False)[0] for i in range(10)]
    scene=StaticScene.capture_empty(c,empty,cell_confirmed_empty=True)
    pipeline=PerceptionPipeline(c,static_scene=scene,error_budget=ErrorBudget(c.calibration_id));accepted=[]
    for i in range(10,18):
        frame,_=render_stereo(c,model,lambda t:truth,i,.1+i/30)
        estimate=pipeline.process(frame,now=frame.arrival_time)
        if estimate.measured:
            error=np.linalg.norm(estimate.pose[:3,3]-truth[:3,3]);accepted.append(error)
            assert error<=estimate.metrics['position_error_bound_m']
            with pytest.raises(Failure):pipeline.camera_observer(require_hardware=False).observe(frame.arrival_time)
    assert len(accepted)>=4,(seed,accepted)
    assert max(accepted)<.006


def test_depth_profile_fit_uses_heldout_independent_plane_reference(tmp_path):
    c=Calibration(160,120,np.array([[120.,0,79.5],[0,120.,59.5],[0,0,1]]),np.eye(4),
                  min_depth=.18,max_depth=1.2,depth_scale=.0001,provenance='synthetic')
    sessions=[];rng=np.random.default_rng(123)
    # Independent affine depth error + temporal white noise, not SensorNoiseModel.
    for j,z in enumerate((.4,.6,.8,.5,.7)):
        recorder=Recorder(tmp_path/f'session_{j}',c)
        for i in range(32):
            raw=z*1.004+.0007+rng.normal(0,.0005,(120,160))
            raw=np.rint(raw/.0001)*.0001
            frame=FramePacket(i,1+i/30,np.full((120,160,3),100,np.uint8),raw,c.calibration_id,'synthetic')
            recorder.append(frame)
        sessions.append({'episode':f'session_{j}','roi':[50,40,100,80],
                         'plane_C':[0,0,1,z],'split':'train' if j<3 else 'holdout'})
    manifest={'sessions':sessions,'reference_uncertainty_m':.0001,'reference_evidence':'independent analytic plane fixture'}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    result=fit_noise_manifest(path)
    assert result['noise_profile']['depth_scale_error']==pytest.approx(.004,abs=.0001)
    assert result['noise_profile']['depth_offset_m']==pytest.approx(.0007,abs=.0001)
    assert max(abs(x) for x in result['heldout_bias_residual_m'])<.0001
    assert not result['qualified_for_control'] and 'extrinsics' in result['unidentified_terms']
    # Moving holdout references cannot alter the trained profile coefficients.
    manifest['sessions'][-1]['plane_C'][-1]+=.002;path.write_text(json.dumps(manifest))
    altered=fit_noise_manifest(path)
    assert altered['noise_profile']['depth_scale_error']==result['noise_profile']['depth_scale_error']
    assert abs(altered['heldout_bias_residual_m'][-1])>.001


@pytest.mark.integration
def test_mujoco_sensor_rig_and_joint_noise_do_not_leak_or_mutate_live_state(monkeypatch):
    from active_scan.simulation import SimulatorAdapter
    from active_scan.perception.sim_source import MujocoRGBDSource
    from active_scan.perception.feedback_noise import JointNoiseProfile
    cfg=load_config('configs/factory.json');cfg['scene']['camera_position']=[.65,-.35,.48];cfg['scene']['camera_target']=[.24,-.04,.08]
    sim=SimulatorAdapter(cfg,'A')
    try:
        source=MujocoRGBDSource(sim,noise_profile=scenario_profile('nominal'),joint_noise_profile=JointNoiseProfile(),seed=17)
        truth=transform(Rotation.from_euler('xyz',[.15,.12,.15]).as_matrix(),[.25,-.10,.08])
        sim._set_cube(sim.data,truth);sim.data.time=1.
        old_q=sim.data.qpos.copy();old_camera=sim.camera_pose().copy()
        monkeypatch.setattr(sim,'cube_pose',lambda *a: (_ for _ in ()).throw(AssertionError('truth leaked')))
        f=source.read()
        assert f.source=='mujoco_stereo' and f.exposure_s>0
        assert np.array_equal(sim.data.qpos,old_q) and np.allclose(sim.camera_pose(),old_camera,atol=1e-12)
        assert np.allclose(f.depth_m[f.depth_m>0]/.001,np.rint(f.depth_m[f.depth_m>0]/.001),atol=.0001)
        assert f.calibration_id==source.calibration.calibration_id and f.robot_mask.any()
    finally:sim.close()


def test_uncertain_pose_cannot_contaminate_tsdf():
    from active_scan.perception.mapping import ObjectMapper
    c=make_calibration();pipeline=PerceptionPipeline(c,error_budget=ErrorBudget(c.calibration_id))
    truth=transform(Rotation.from_euler('xyz',[.2,.3,.1]).as_matrix(),[.26,.01,.08])
    for i in range(2):f,_=render(c,truth,i,1+i/30);e=pipeline.process(f)
    assert e.measured
    mapper=ObjectMapper(c)
    assert not mapper.add(f,e) and not mapper.keyframes


def test_budget_identity_and_new_metadata_survive_replay(tmp_path):
    c=make_calibration();m=StereoNoiseModel(c,scenario_profile('nominal'),2)
    frame,_=render_stereo(c,m,lambda t:transform(p=[.26,0,.08]),0,1.)
    recorder=Recorder(tmp_path/'episode',c);recorder.append(frame)
    replayed=next(iter(Replay(tmp_path/'episode')))
    assert replayed.exposure_s==frame.exposure_s and replayed.arrival_time==frame.arrival_time
    assert np.array_equal(replayed.depth_m,frame.depth_m)
    with pytest.raises(ValueError):PerceptionPipeline(c,error_budget=ErrorBudget('different-calibration'))


def test_small_surface_residual_can_hide_centimeter_world_pose_bias():
    from active_scan.perception.synthetic import d455_fixture_calibration
    c=d455_fixture_calibration();model=StereoNoiseModel(c,scenario_profile('spec_limit'),10998)
    truth=transform(Rotation.from_euler('xyz',[.2,.3,.1]).as_matrix(),[.26,.01,.085])
    empty=[render_stereo(c,model,lambda t:truth,i,.1+i/30,include_cube=False)[0] for i in range(10)]
    scene=StaticScene.capture_empty(c,empty,cell_confirmed_empty=True)
    pipeline=PerceptionPipeline(c,static_scene=scene,error_budget=ErrorBudget(c.calibration_id))
    for i in range(10,13):
        f,_=render_stereo(c,model,lambda t:truth,i,.1+i/30);e=pipeline.process(f,now=f.arrival_time)
    assert e.measured
    actual=np.linalg.norm(e.pose[:3,3]-truth[:3,3])
    assert actual>.01 and e.metrics['surface_rmse_m']<.002
    assert actual<e.metrics['position_error_bound_m']
    with pytest.raises(Failure,match='GRASP_EVIDENCE_UNCERTAIN'):pipeline.camera_observer(require_hardware=False).observe(f.arrival_time)

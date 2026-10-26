from dataclasses import replace,fields
import json
import numpy as np
import pytest
from active_scan.core import transform,Failure
from active_scan.grasp import ObjectObservation,verify_test_lift,CameraObserver
from active_scan.perception.types import Calibration,FramePacket,Status
from active_scan.perception.sensor_noise import StereoNoiseProfile,StereoNoiseModel,scenario_profile,disparity_depth_sigma,stereo_visibility
from active_scan.perception.synthetic import test_calibration as make_calibration,render,render_stereo
from active_scan.perception.feedback_noise import JointNoiseProfile,JointNoiseModel,ImuNoiseProfile,ImuNoiseModel,linearized_fk_position_bound
from active_scan.perception.pipeline import PerceptionPipeline
from active_scan.perception.uncertainty import ErrorBudget
from active_scan.perception.scene import StaticScene


def flat_camera(scale=.00001):
    return Calibration(160,100,np.array([[120.,0,79.5],[0,120.,49.5],[0,0,1]]),np.eye(4),
                       min_depth=.18,max_depth=2.,depth_scale=scale)


def quiet_profile(**changes):
    p=StereoNoiseProfile()
    zeros={f.name:0 for f in fields(p) if isinstance(getattr(p,f.name),(float,int))}
    zeros.update(baseline_m=.095,temporal_reference_dt_s=1/30,vibration_hz=18.,texture_quality=.999999,
                 correlation_length_px=0.,edge_width_px=0)
    return replace(p,**{**zeros,**changes})


def plane(c,z,sequence=0,timestamp=1.):
    return FramePacket(sequence,timestamp,np.full((c.height,c.width,3),100,np.uint8),
                       np.full((c.height,c.width),z,np.float32),c.calibration_id,'synthetic')


def test_quadratic_stereo_variance_is_realized_in_images():
    c=flat_camera();deviations=[]
    for z in (.4,.8):
        model=StereoNoiseModel(c,quiet_profile(disparity_white_px=.08),seed=3)
        f,_=model.apply(plane(c,z),1.)
        values=f.depth_m[f.depth_m>0];deviations.append(float(values.std()))
        assert values.std()==pytest.approx(disparity_depth_sigma(z,120,.095,.08),rel=.12)
    assert deviations[1]/deviations[0]==pytest.approx(4.,rel=.15)


def test_quantization_invalid_depth_and_seed_reproducibility():
    c=flat_camera(.001);p=quiet_profile(disparity_white_px=.07)
    clean=plane(c,.6004);d=clean.depth_m.copy();d[10:20,60:80]=0;clean=replace(clean,depth_m=d)
    a,_=StereoNoiseModel(c,p,5).apply(clean,1.);b,_=StereoNoiseModel(c,p,5).apply(clean,1.)
    assert np.array_equal(a.depth_m,b.depth_m)
    assert np.max(np.abs(a.depth_m/.001-np.rint(a.depth_m/.001)))<.0001
    assert np.count_nonzero(a.depth_m[10:20,60:80])==0
    assert a.calibration_id==c.calibration_id and not a.depth_m.flags.writeable


def test_persistent_bias_does_not_average_to_zero():
    c=flat_camera();model=StereoNoiseModel(c,quiet_profile(depth_offset_m=.004,depth_scale_error=.01,fixed_pattern_m=.0005),1)
    values=[]
    for i in range(10):
        f,_=model.apply(plane(c,.6,i),1+i/30);values.append(f.depth_m[:,40:])
    assert np.max(np.std(np.stack(values),axis=0))<1e-6
    assert np.mean(values[0])-.6==pytest.approx(.010,abs=.0003)


def test_correlated_noise_is_temporally_and_spatially_correlated():
    c=flat_camera();model=StereoNoiseModel(c,quiet_profile(disparity_correlated_px=.12,correlation_length_px=2,temporal_correlation=.85),2)
    frames=[model.apply(plane(c,.6,i),1+i/30)[0].depth_m[:,40:] for i in range(12)]
    a=frames[3]-.6;b=frames[4]-.6
    assert np.corrcoef(a.ravel(),b.ravel())[0,1]>.65
    assert np.corrcoef(a[:,:-1].ravel(),a[:,1:].ravel())[0,1]>.70


def test_mount_and_intrinsics_are_sampled_once_not_exposed_to_estimator():
    c=make_calibration();m=StereoNoiseModel(c,scenario_profile('nominal'),10)
    actual=m.true_calibration(1.);actual2=m.true_calibration(1.)
    assert actual.calibration_id!=c.calibration_id
    assert np.array_equal(actual.K,actual2.K) and np.array_equal(actual.T_B_C,actual2.T_B_C)
    assert not np.allclose(m.true_calibration(1.01).T_B_C,actual.T_B_C,atol=1e-8)
    pose=transform(p=[.26,0,.08]);f,diag=render_stereo(c,m,lambda t:pose,0,1.)
    assert f.calibration_id==c.calibration_id and 'actual_scale_error' not in f.__dataclass_fields__
    assert 'actual_scale_error' in diag and f.exposure_s>0 and f.arrival_time>=1.


def test_bad_return_regions_reduce_validity_and_stereo_border_has_no_matches():
    c=flat_camera();clean=plane(c,.6)
    a,_=StereoNoiseModel(c,quiet_profile(),0).apply(clean,1.)
    b,_=StereoNoiseModel(c,quiet_profile(dropout_probability=.2,cluster_dropout_probability=.1,specular_fraction=.15,texture_quality=.2),0).apply(clean,1.)
    assert np.count_nonzero(b.depth_m)<np.count_nonzero(a.depth_m)*.7
    assert not stereo_visibility(clean.depth_m,120,.095)[:,:19].any()


def test_exposure_motion_edges_and_rgb_blur_are_simulated():
    from scipy.spatial.transform import Rotation
    c=make_calibration();m=StereoNoiseModel(c,quiet_profile(exposure_s=.03),4)
    def pose(t):return transform(Rotation.from_euler('xyz',[.2,.3,.1]).as_matrix(),[.26+(t-1)*.5,0,.08])
    f,diag=render_stereo(c,m,pose,0,1.)
    assert diag['moving_edge_fraction']>0
    assert len(np.unique(f.rgb.reshape(-1,3),axis=0))>3


@pytest.mark.parametrize('name',['nominal','adverse','spec_limit','timing_fault'])
def test_profiles_roundtrip_and_unmeasured_provenance(name):
    p=scenario_profile(name);assert StereoNoiseProfile.from_dict(p.to_dict())==p
    assert 'assumption' in p.provenance
    d=p.to_dict();d['baseline_m']=.05
    with pytest.raises(ValueError):StereoNoiseProfile.from_dict(d)


@pytest.mark.parametrize('value',[float('nan'),-1,1.1])
def test_invalid_probability_rejected(value):
    with pytest.raises(ValueError):replace(StereoNoiseProfile(),dropout_probability=value)


def test_clock_and_transport_fault_is_rejected():
    c=make_calibration();m=StereoNoiseModel(c,scenario_profile('timing_fault'),0)
    f,_=render_stereo(c,m,lambda t:transform(p=[.26,0,.08]),0,1.)
    assert PerceptionPipeline(c).process(f,now=f.arrival_time).status==Status.STALE


def test_systematic_budget_never_divides_by_pixel_count():
    c=make_calibration();b=ErrorBudget(c.calibration_id);f=plane(c,.6)
    pose=transform(p=[.26,0,.08]);small=np.zeros(f.depth_m.shape,bool);small[:10,:10]=True
    large=np.ones(f.depth_m.shape,bool)
    cv,a=b.evaluate(pose,small,f,c,{'visible_planes':3});_,z=b.evaluate(pose,large,f,c,{'visible_planes':3})
    assert a['systematic_position_bound_m']==z['systematic_position_bound_m']
    assert z['position_error_bound_m']>.012
    assert np.linalg.eigvalsh(cv).min()>=0
    with pytest.raises(ValueError):replace(b,qualified=True)


def test_low_residual_is_not_hardware_error_qualification():
    from scipy.spatial.transform import Rotation
    c=make_calibration();pipeline=PerceptionPipeline(c,error_budget=ErrorBudget(c.calibration_id))
    pose=transform(Rotation.from_euler('xyz',[.2,.3,.1]).as_matrix(),[.26,.01,.08])
    for i in range(2):f,_=render(c,pose,i,1+i/30);e=pipeline.process(f)
    assert e.measured and e.metrics['surface_rmse_m']<.001
    assert e.metrics['position_error_bound_m']>.003
    with pytest.raises(Failure):pipeline.camera_observer(require_hardware=False).observe(e.timestamp)


def test_age_and_relative_uncertainty_block_false_grasp_acceptance():
    a=ObjectObservation(1.,transform(),'camera',position_error_bound_m=.001,relative_error_bound_m=.0015)
    b=replace(a,timestamp=1.1,pose=transform(p=[.002,0,.012]))
    with pytest.raises(Failure,match='GRASP_EVIDENCE_UNCERTAIN'):
        verify_test_lift(a,b,transform(),transform(p=[0,0,.012]),.006,.004)
    observer=CameraObserver(lambda:b,motion_speed_bound=.5,max_position_error=.003)
    with pytest.raises(Failure,match='GRASP_EVIDENCE_UNCERTAIN'):observer.observe(1.12)


def test_encoder_zero_backlash_quantization_and_fk_envelope():
    profile=JointNoiseProfile(white_std_rad=0,encoder_step_rad=0,zero_std_rad=.001)
    model=JointNoiseModel(profile,seed=4)
    first,_=model.measure(np.zeros(6),1.)
    forward,_=model.measure(np.ones(6)*.1,1.1)
    backward,_=model.measure(np.zeros(6),1.2)
    assert np.allclose(forward-.1-first,profile.backlash_rad/2,atol=2e-6)
    assert np.allclose(backward-first,-profile.backlash_rad/2,atol=3e-6)
    assert linearized_fk_position_bound(np.ones((3,6))*.2,profile.angle_envelope())>.001


def test_imu_noise_density_dt_scaling_and_diagnostic_only():
    p=ImuNoiseProfile(gyro_bias_rw_rad_s2_sqrt_hz=0,accel_bias_rw_m_s3_sqrt_hz=0,
                      gyro_initial_bias_std_rad_s=0,accel_initial_bias_std_m_s2=0)
    deviations=[]
    for dt in (.01,.0025):
        model=ImuNoiseModel(p,1);samples=np.array([model.measure(np.zeros(3),np.zeros(3),dt)[0] for _ in range(2000)])
        deviations.append(samples.std())
    assert deviations[1]/deviations[0]==pytest.approx(2.,rel=.03)
    with pytest.raises(ValueError):ImuNoiseModel().measure([0,0,0],[0,0,0],0)


def test_static_background_drift_blocks_pose():
    c=make_calibration();pose=transform(p=[.26,0,.08])
    frames=[render(c,pose,i,1+i/30,include_cube=False)[0] for i in range(5)]
    scene=StaticScene.capture_empty(c,frames,cell_confirmed_empty=True)
    f=replace(frames[-1],sequence=10,capture_time=2.,depth_m=np.where(frames[-1].depth_m>0,frames[-1].depth_m+.015,0))
    e=PerceptionPipeline(c,static_scene=scene).process(f)
    assert e.status==Status.CALIBRATION_INVALID and e.metrics['reason']=='fixed_background_changed'


@pytest.mark.parametrize('bad',['tool_nan','object_nan','nonrigid','timestamp_nan','negative_bound'])
def test_grasp_verification_rejects_invalid_inputs(bad):
    a=ObjectObservation(1.,transform(),'camera',relative_error_bound_m=.0002,angular_error_bound_rad=.001)
    b=replace(a,timestamp=1.1,pose=transform(p=[0,0,.012]));tool=transform(p=[0,0,.012])
    if bad=='tool_nan':tool[0,3]=np.nan
    if bad=='object_nan':b.pose[0,3]=np.nan
    if bad=='nonrigid':tool[0,0]=2.
    if bad=='timestamp_nan':b.timestamp=np.nan
    if bad=='negative_bound':b.tool_position_bound_m=-1.
    with pytest.raises(Failure):verify_test_lift(a,b,transform(),tool,.006,.004)


def test_tight_validated_bounds_can_confirm_a_software_fixture():
    a=ObjectObservation(1.,transform(),'camera',relative_error_bound_m=.0002,angular_error_bound_rad=.001,tool_position_bound_m=.0001)
    b=replace(a,timestamp=1.1,pose=transform(p=[0,0,.012]))
    assert verify_test_lift(a,b,transform(),transform(p=[0,0,.012]),.006,.004)['uncertainty_checked']


def test_legacy_heuristic_covariance_cannot_enable_hardware_control():
    from scipy.spatial.transform import Rotation
    c=replace(make_calibration(),provenance='hardware',serial='fixture_never_a_real_camera')
    pipeline=PerceptionPipeline(c)
    pose=transform(Rotation.from_euler('xyz',[.2,.3,.1]).as_matrix(),[.26,.01,.08])
    for i in range(2):
        f,_=render(c,pose,i,1+i/30);pipeline.process(replace(f,source='realsense'))
    assert pipeline.latest.measured
    with pytest.raises(Failure,match='GRASP_EVIDENCE_UNCERTAIN'):pipeline.camera_observer().observe(1+1/30)

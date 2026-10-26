"""Independent sensor images and fault injection; no estimator truth injection."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from active_scan.core import transform, points, inverse, Failure
from active_scan.grasp import verify_test_lift
from active_scan.perception.types import Calibration, Status
from active_scan.perception.calibration import deproject, calibrate_camera_to_base
from active_scan.perception.localizer import SYMMETRIES, symmetry_distance
from active_scan.perception.pipeline import PerceptionPipeline
from active_scan.perception.synthetic import test_calibration as make_calibration, render
from active_scan.perception.recording import Recorder, Replay
from active_scan.perception.sync import ClockMapper, FeedbackBuffer, LatestFrameBuffer


@pytest.fixture
def calibration(): return make_calibration()


def cube_pose(rpy=(.2,.3,.1), center=(.26,.01,.08)):
    return transform(Rotation.from_euler('xyz',rpy).as_matrix(), center)


def tracked(calibration, pose=None, **kwargs):
    pose=cube_pose() if pose is None else pose
    pipeline=PerceptionPipeline(calibration)
    for i in range(2):
        frame,truth=render(calibration,pose,i,i/30,seed=i,**kwargs)
        estimate=pipeline.process(frame)
    return pipeline,frame,estimate,truth


@pytest.mark.parametrize('seed',range(10))
def test_pose_accuracy_on_independent_noisy_images(calibration,seed):
    rng=np.random.default_rng(seed)
    pose=cube_pose(rng.uniform(-.5,.5,3),(.25+rng.uniform(-.02,.02),.01,.07+seed*.004))
    _,frame,result,truth=tracked(calibration,pose,noise=.0009,holes=.12,outliers=.01)
    assert result.status==Status.TRACKING, result.metrics
    assert np.linalg.norm(result.pose[:3,3]-truth[:3,3])<.002
    assert symmetry_distance(result.pose[:3,:3],truth[:3,:3])<np.deg2rad(3)
    assert result.symmetry_ambiguous
    assert np.count_nonzero(result.mask & (frame.depth_m==0))==0


@pytest.mark.parametrize('noise,holes,outliers',[(.0012,.25,.02),(.0005,.4,.005),(0,0,0)])
def test_depth_fault_tolerance(calibration,noise,holes,outliers):
    _,_,result,truth=tracked(calibration,noise=noise,holes=holes,outliers=outliers)
    assert result.measured, result.metrics
    assert np.linalg.norm(result.pose[:3,3]-truth[:3,3])<.003


def test_blank_occlusion_and_recovery(calibration):
    pipeline,frame,result,_=tracked(calibration)
    missing=replace(frame,sequence=2,capture_time=.07,depth_m=np.zeros_like(frame.depth_m))
    predicted=pipeline.process(missing)
    assert predicted.status==Status.PREDICTED
    with pytest.raises(Failure): predicted.require_measured()
    assert pipeline.process(replace(missing,sequence=3,capture_time=.4)).status==Status.LOST
    fresh,_=render(calibration,cube_pose(),4,.43)
    assert pipeline.process(fresh).status==Status.RELOCALIZING
    assert pipeline.process(replace(fresh,sequence=5,capture_time=.47)).status==Status.TRACKING


@pytest.mark.parametrize('mode',['stale','future','duplicate','reordered','clock_uncertain','rgb_skew','calibration','shape'])
def test_health_gates_fail_closed(calibration,mode):
    pipeline,frame,_,_=tracked(calibration)
    f=replace(frame,sequence=2,capture_time=.08);now=.08
    if mode=='stale': now=.3
    if mode=='future': now=.07
    if mode=='duplicate': f=frame
    if mode=='reordered': f=replace(f,capture_time=.01)
    if mode=='clock_uncertain': f=replace(f,time_uncertainty=.03)
    if mode=='rgb_skew': f=replace(f,color_time_offset=.03)
    if mode=='calibration': f=replace(f,calibration_id='wrong')
    if mode=='shape': f=replace(f,rgb=f.rgb[:10],depth_m=f.depth_m[:10],robot_mask=f.robot_mask[:10])
    result=pipeline.process(f,now)
    assert result.status==(Status.CALIBRATION_INVALID if mode in ('calibration','shape') else Status.STALE)
    assert not result.measured
    with pytest.raises(Failure): pipeline.camera_observer(require_hardware=False).observe(now)


def test_table_is_not_object(calibration):
    frame,_=render(calibration,cube_pose(),include_cube=False)
    assert PerceptionPipeline(calibration).process(frame).status==Status.LOST


def test_robot_mask_removes_robot_without_filling_depth(calibration):
    frame,_=render(calibration,cube_pose())
    result=PerceptionPipeline(calibration).process(replace(frame,robot_mask=np.ones_like(frame.depth_m,bool)))
    assert result.status==Status.LOST


def test_single_front_plane_is_unobservable(calibration):
    c=calibration
    pose=transform(c.T_B_C[:3,:3], c.T_B_C[:3,3]+c.T_B_C[:3,2]*.62)
    frame,_=render(c,pose,noise=.0002,holes=0,outliers=0)
    result=PerceptionPipeline(c).process(frame)
    assert result.status in (Status.DEGENERATE,Status.LOST),result.metrics


def test_two_cubes_are_ambiguous(calibration):
    p1=cube_pose(center=(.20,-.07,.08));p2=cube_pose(center=(.33,.08,.08))
    frame,_=render(calibration,p1,occluders=[(p2,[.03]*3)])
    result=PerceptionPipeline(calibration).process(frame)
    assert result.status==Status.AMBIGUOUS,result.metrics


def test_cube_symmetry_has_24_proper_rotations():
    assert len(SYMMETRIES)==24 and np.allclose(np.linalg.det(SYMMETRIES),1)
    r=Rotation.from_euler('xyz',[.3,.1,.7]).as_matrix()
    for s in SYMMETRIES: assert symmetry_distance(r,r@s)<1e-7


def test_symmetry_continuity_and_large_reposition(calibration):
    pipeline,frame,result,truth=tracked(calibration)
    symmetric=truth.copy();symmetric[:3,:3]=symmetric[:3,:3]@SYMMETRIES[3]
    frame,_=render(calibration,symmetric,2,.08);after=pipeline.process(frame)
    assert after.measured
    assert Rotation.from_matrix(after.pose[:3,:3].T@result.pose[:3,:3]).magnitude()<.02
    moved=truth.copy();moved[:3,3]+=[.10,0,0]
    frame,_=render(calibration,moved,3,.10)
    assert pipeline.process(frame).status==Status.RELOCALIZING


def test_camera_evidence_detects_failed_lift(calibration):
    pipeline,frame,result,_=tracked(calibration)
    observer=pipeline.camera_observer(require_hardware=False);before=observer.observe(frame.capture_time)
    f,_=render(calibration,cube_pose(),2,.08)
    pipeline.process(f);after=observer.observe(.08)
    with pytest.raises(Failure,match='TEST_LIFT_FAILED'):
        verify_test_lift(before,after,transform(),transform(p=[0,0,.01]),.006,.004)
    with pytest.raises(Failure): pipeline.camera_observer().observe(.08)


@pytest.mark.parametrize('slip,expected',[(0,True),(.002,True),(.008,False)])
def test_lift_and_slip_from_images(calibration,slip,expected):
    pipeline,frame,result,truth=tracked(calibration)
    observer=pipeline.camera_observer(require_hardware=False);before=observer.observe(frame.capture_time)
    lifted=truth.copy();lifted[:3,3]+=[slip,0,.012]
    f,_=render(calibration,lifted,2,.08);pipeline.process(f);after=observer.observe(.08)
    if expected:
        assert verify_test_lift(before,after,transform(),transform(p=[0,0,.012]),.006,.004)['lift']>.009
    else:
        with pytest.raises(Failure,match='TEST_LIFT_FAILED'):
            verify_test_lift(before,after,transform(),transform(p=[0,0,.012]),.006,.004)


def test_record_replay_is_identical_and_sensor_only(calibration,tmp_path):
    recorder=Recorder(tmp_path/'episode',calibration);original=[]
    for i in range(4):
        frame,_=render(calibration,cube_pose(),i,i/30,seed=i)
        recorder.append(frame);original.append(frame)
    replay=Replay(tmp_path/'episode')
    a=PerceptionPipeline(calibration);b=PerceptionPipeline(replay.calibration)
    for first,second in zip(original,replay):
        assert np.array_equal(first.depth_m,second.depth_m)
        ea=a.process(first);eb=b.process(second)
        assert ea.status==eb.status and np.allclose(ea.pose,eb.pose,atol=1e-10)
    assert 'truth' not in (tmp_path/'episode'/'index.json').read_text()
    with pytest.raises(FileExistsError): Recorder(tmp_path/'episode',calibration)
    path=tmp_path/'episode'/'frame_000000.npz';path.write_bytes(path.read_bytes()+b'corruption')
    with pytest.raises(ValueError,match='checksum'): list(Replay(tmp_path/'episode'))


def test_calibration_checksum_and_units(calibration):
    c=Calibration.from_dict(calibration.to_dict());assert c.calibration_id==calibration.calibration_id
    data=c.to_dict();data['depth_scale']=.01
    with pytest.raises(ValueError,match='checksum'): Calibration.from_dict(data)
    with pytest.raises(ValueError): replace(c,depth_scale=1000)
    with pytest.raises(ValueError): replace(c,T_B_C=np.zeros((4,4)))
    d=np.zeros((c.height,c.width));d[100,200]=.5;p=deproject(d,c)
    assert p.shape==(1,3) and p[0,2]==.5


def test_camera_base_calibration_on_heldout_points(calibration):
    rng=np.random.default_rng(41);c=calibration
    world=rng.uniform([.15,-.1,.01],[.35,.1,.15],(50,3));camera=points(inverse(c.T_B_C),world)
    uv=camera@c.K.T;uv=uv[:,:2]/uv[:,2,None]
    train=uv[:35]+rng.normal(0,.15,(35,2));train[0]+=[40,-40]
    estimated,metrics=calibrate_camera_to_base(world[:35],train,c.K)
    held=points(inverse(estimated),world[35:]);held=held@c.K.T;held=held[:,:2]/held[:,2,None]
    assert np.sqrt(np.mean((held-uv[35:])**2))<.3
    assert np.linalg.norm(estimated[:3,3]-c.T_B_C[:3,3])<.003
    assert metrics['inliers']>=33 and not metrics['absolute_accuracy_validated']


def test_clock_reset_feedback_interpolation_and_latest_queue():
    clock=ClockMapper()
    for i in range(20):
        t,u,e=clock.map(i*33.333,10+i/30+.002*(i%3))
        assert t<=10+i/30+.002*(i%3) and u<.025 and e==0
    assert clock.map(0,11)[2]==1
    feedback=FeedbackBuffer();feedback.add(0,np.zeros(5));feedback.add(.02,np.ones(5))
    assert np.allclose(feedback.at(.01),.5)
    with pytest.raises(Failure): feedback.at(.1)
    with pytest.raises(ValueError): feedback.add(.01,np.ones(5))
    queue=LatestFrameBuffer()
    for i in range(1000): queue.publish(i)
    assert queue.take()==999 and queue.take() is None and queue.dropped==999


def test_frame_owns_immutable_buffers(calibration):
    f,_=render(calibration,cube_pose())
    with pytest.raises(ValueError): f.depth_m[:]=0
    bad=f.depth_m.copy();bad[:]=np.nan
    assert np.count_nonzero(replace(f,depth_m=bad).depth_m)==0


def test_usb_restart_requires_reconfirmation(calibration):
    pipeline,frame,_,_=tracked(calibration)
    result=pipeline.process(replace(frame,epoch=1,sequence=0,capture_time=.2))
    assert result.status==Status.RELOCALIZING


@pytest.mark.hardware
def test_real_d455_probe():
    from active_scan.perception.realsense import probe
    devices=probe()
    if not devices: pytest.skip('No RealSense USB device present; physical acceptance untested')
    assert any('D455' in d['name'] for d in devices)

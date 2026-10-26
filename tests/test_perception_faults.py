from dataclasses import replace
import json
import time
import numpy as np
import pytest
from active_scan.core import transform,Failure
from active_scan.grasp import verify_test_lift,ObjectObservation
from active_scan.perception.synthetic import test_calibration as make_calibration,render
from active_scan.perception.pipeline import PerceptionPipeline
from active_scan.perception.runtime import LocalizationRuntime
from active_scan.perception.scene import StaticScene
from active_scan.perception.types import Status


@pytest.mark.parametrize('size',[.03,.04,.05,.09,.12])
def test_wrong_size_distractors_not_accepted(size):
    c=make_calibration();frame,_=render(c,transform(p=[.26,0,.09]),size=size,noise=.0004)
    result=PerceptionPipeline(c).process(frame)
    assert result.status in (Status.DEGENERATE,Status.LOST),result.metrics


def test_tiny_exposed_patch_cannot_confirm_pose():
    c=make_calibration();frame,_=render(c,transform(p=[.26,0,.09]))
    # Keep just a small sensor crop, matching a strong gripper occlusion.
    depth=np.zeros_like(frame.depth_m);depth[110:117,207:215]=frame.depth_m[110:117,207:215]
    pipeline=PerceptionPipeline(c)
    for i in range(3):
        result=pipeline.process(replace(frame,depth_m=depth,sequence=i,capture_time=i/30))
        assert not result.measured


def test_scene_reference_roundtrip_and_foreground_only(tmp_path):
    c=make_calibration();pose=transform(p=[.26,0,.09])
    frames=[render(c,pose,i,i/30,include_cube=False,seed=i)[0] for i in range(5)]
    scene=StaticScene.capture_empty(c,frames,cell_confirmed_empty=True)
    scene.save(tmp_path/'scene.npz');scene=StaticScene.load(tmp_path/'scene.npz',c)
    pipeline=PerceptionPipeline(c,static_scene=scene)
    for i in range(2):
        f,_=render(c,pose,i,i/30);result=pipeline.process(f)
    assert result.measured
    assert not pipeline.process(replace(frames[-1],sequence=2,capture_time=.08)).measured


def test_runtime_localizes_off_control_thread_and_fails_after_stop():
    c=make_calibration();template,_=render(c,transform(p=[.26,0,.09]))
    class Source:
        calibration=c
        def __init__(self):self.sequence=0;self.closed=False
        def read(self):
            time.sleep(.005);self.sequence+=1
            return replace(template,sequence=self.sequence,capture_time=time.monotonic())
        def close(self):self.closed=True
    source=Source();runtime=LocalizationRuntime(source).start()
    observer=runtime.camera_observer(require_hardware=False)
    try:
        deadline=time.monotonic()+3.
        while runtime.processed<3 and time.monotonic()<deadline:time.sleep(.01)
        assert runtime.processed>=3 and runtime.capture.buffer.dropped>0
        assert observer.observe(time.monotonic()).provenance=='synthetic'
    finally:runtime.close()
    assert source.closed and not runtime.thread.is_alive() and not runtime.capture.thread.is_alive()
    with pytest.raises(Failure):observer.observe(time.monotonic())


@pytest.mark.parametrize('seed',range(5))
@pytest.mark.parametrize('slip,expected',[(0,True),(.002,True),(.008,False)])
def test_measured_lift_slip_campaign(seed,slip,expected):
    from scipy.spatial.transform import Rotation
    c=make_calibration();pipeline=PerceptionPipeline(c)
    pose=transform(Rotation.from_euler('xyz',[.2,.3,.1]).as_matrix(),[.26,.01,.08])
    for i in range(2):
        f,_=render(c,pose,i,i/30,noise=.0009,holes=.12,outliers=.01,seed=seed*10+i)
        pipeline.process(f)
    observer=pipeline.camera_observer(require_hardware=False);before=observer.observe(1/30)
    moved=pose.copy();moved[:3,3]+=[slip,0,.012]
    f,_=render(c,moved,2,.08,noise=.0009,holes=.12,outliers=.01,seed=seed*10+2)
    pipeline.process(f);after=observer.observe(.08)
    if expected:verify_test_lift(before,after,transform(),transform(p=[0,0,.012]),.006,.004)
    else:
        with pytest.raises(Failure,match='TEST_LIFT_FAILED'):
            verify_test_lift(before,after,transform(),transform(p=[0,0,.012]),.006,.004)


def test_changed_identity_or_epoch_cannot_verify_lift():
    a=ObjectObservation(1.,transform(),'camera')
    b=ObjectObservation(1.1,transform(p=[0,0,.01]),'camera',grasp_epoch=1)
    with pytest.raises(Failure,match='GRASP_EVIDENCE_UNAVAILABLE'):
        verify_test_lift(a,b,transform(),transform(p=[0,0,.01]),.006,.004)


def test_bad_calibration_points_are_rejected():
    from active_scan.perception.calibration import calibrate_camera_to_base
    c=make_calibration()
    with pytest.raises(ValueError):calibrate_camera_to_base(np.zeros((10,3)),np.zeros((10,2)),c.K)
    with pytest.raises(ValueError):calibrate_camera_to_base(np.ones((4,3)),np.zeros((4,2)),c.K)


def test_measured_scene_snapshot_requires_aligned_feedback():
    from active_scan.perception.snapshot import MeasuredSceneProvider
    from scipy.spatial.transform import Rotation
    c=make_calibration();pipeline=PerceptionPipeline(c)
    for i in range(2):
        f,_=render(c,transform(Rotation.from_euler('xyz',[.2,.3,.1]).as_matrix(),[.26,.01,.08]),i,i/30)
        estimate=pipeline.process(f)
    provider=MeasuredSceneProvider(pipeline)
    snap=provider.snapshot(.04,1/30,np.zeros(5),np.zeros(5),1.,transform())
    assert np.array_equal(snap.state.T_world_cube,estimate.pose)
    assert snap.state.timestamp==estimate.timestamp and snap.source=='synthetic'
    assert snap.state.cube_velocity.shape==(6,)
    with pytest.raises(Failure,match='TIME_MISMATCH'):provider.snapshot(.04,0,np.zeros(5),np.zeros(5),1.,transform())
    with pytest.raises(Failure,match='STALE'):provider.snapshot(.4,1/30,np.zeros(5),np.zeros(5),1.,transform())

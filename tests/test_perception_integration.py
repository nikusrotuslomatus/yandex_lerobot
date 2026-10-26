from dataclasses import replace
import threading
import time
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from active_scan.core import load_config, transform, Failure
from active_scan.simulation import SimulatorAdapter
from active_scan.perception.pipeline import PerceptionPipeline
from active_scan.perception.synthetic import test_calibration as make_calibration, render
from active_scan.perception.localizer import symmetry_distance
from active_scan.perception.mapping import ObjectMapper
from active_scan.perception.scene import StaticScene
from active_scan.perception.runtime import CaptureWorker
from active_scan.perception.types import Status
from active_scan.perception.realsense import D455Source,profile_min_depth


@pytest.mark.integration
def test_mujoco_raw_rgbd_pose_and_camera_observer(monkeypatch):
    from active_scan.perception.sim_source import MujocoRGBDSource
    cfg=load_config('configs/factory.json')
    cfg['scene']['camera_position']=[.65,-.35,.48]
    cfg['scene']['camera_target']=[.24,-.04,.08]
    sim=SimulatorAdapter(cfg,'A')
    try:
        source=MujocoRGBDSource(sim);pipeline=PerceptionPipeline(source.calibration)
        expected=[];actual=[]
        for i in range(8):
            truth=transform(Rotation.from_euler('xyz',[.15,.12,i*.06]).as_matrix(),[.25,-.10,.08+i*.001])
            sim._set_cube(sim.data,truth);sim.data.time=i/30
            # Ground truth accessor deliberately unavailable in the sensing path.
            with monkeypatch.context() as m:
                m.setattr(sim,'cube_pose',lambda *a: (_ for _ in ()).throw(AssertionError('truth leaked')))
                frame=source.read();result=pipeline.process(frame)
            if i:
                assert result.measured,result.metrics
                expected.append(truth);actual.append(result.pose)
        errors=[np.linalg.norm(a[:3,3]-b[:3,3]) for a,b in zip(actual,expected)]
        angles=[symmetry_distance(a[:3,:3],b[:3,:3]) for a,b in zip(actual,expected)]
        assert max(errors)<.003 and max(angles)<np.deg2rad(3)
        sim.object_observer=pipeline.camera_observer(require_hardware=False)
        observation=sim.observe_object()
        assert observation.source=='camera' and observation.provenance=='mujoco'
        assert np.array_equal(observation.pose,result.pose)
    finally: sim.close()


@pytest.mark.integration
def test_mujoco_bad_occluded_pose_is_rejected():
    from active_scan.perception.sim_source import MujocoRGBDSource
    sim=SimulatorAdapter(load_config('configs/factory.json'),'A')
    try:
        sim._set_cube(sim.data,transform(p=[.24,0,.15]))
        source=MujocoRGBDSource(sim)
        result=PerceptionPipeline(source.calibration).process(source.read())
        assert result.status in (Status.DEGENERATE,Status.LOST)
    finally:sim.close()


@pytest.mark.integration
def test_tsdf_backend_metric_depth_regression():
    import open3d as o
    volume=o.pipelines.integration.ScalableTSDFVolume(.005,.02,o.pipelines.integration.TSDFVolumeColorType.RGB8)
    rgbd=o.geometry.RGBDImage.create_from_color_and_depth(
        o.geometry.Image(np.full((48,64,3),100,np.uint8)),o.geometry.Image(np.full((48,64),.6,np.float32)),
        depth_scale=1.,depth_trunc=1.,convert_rgb_to_intensity=False)
    volume.integrate(rgbd,o.camera.PinholeCameraIntrinsic(64,48,50,50,31.5,23.5),np.eye(4))
    cloud=np.asarray(volume.extract_point_cloud().points)
    assert len(cloud)>500 and abs(np.median(cloud[:,2])-.6)<.001


@pytest.mark.integration
def test_tsdf_fuses_measured_object_only_and_checks_epoch(tmp_path):
    c=make_calibration();pipeline=PerceptionPipeline(c);mapper=ObjectMapper(c,max_keyframes=8)
    accepted=0
    for i in range(18):
        pose=transform(Rotation.from_euler('xyz',[.2,.3,i*.07]).as_matrix(),[.26,.01,.08])
        frame,_=render(c,pose,i,i/30,seed=i)
        result=pipeline.process(frame)
        accepted+=int(mapper.add(frame,result))
    assert 3<=accepted<=8
    cloud=np.asarray(mapper.cloud().points)
    assert len(cloud)>300
    assert np.max(np.abs(cloud))<.037  # No table or distant robot geometry.
    shell=np.abs(np.max(np.abs(cloud),axis=1)-.03)
    assert np.quantile(shell,.95)<.0025
    mapper.save(tmp_path/'cube.ply');assert (tmp_path/'cube.ply').stat().st_size>1000
    lost=replace(result,status=Status.PREDICTED)
    assert not mapper.add(frame,lost)
    with pytest.raises(ValueError,match='provenance'):mapper.add(replace(frame,sequence=99),result)
    with pytest.raises(ValueError,match='epoch'):mapper.add(frame,replace(result,grasp_epoch=99))


def test_empty_scene_reference_and_calibration_mismatch(tmp_path):
    c=make_calibration();frames=[]
    for i in range(5):
        frame,_=render(c,transform(p=[.26,0,.03]),i,i/30,include_cube=False,noise=.0005)
        frames.append(frame)
    with pytest.raises(ValueError):StaticScene.capture_empty(c,frames)
    scene=StaticScene.capture_empty(c,frames,cell_confirmed_empty=True)
    assert abs(scene.table_error_m)<.001
    object_frame,_=render(c,transform(p=[.26,0,.08]))
    foreground=scene.foreground(object_frame)
    assert foreground.sum()>500
    scene.save(tmp_path/'scene.npz')
    with pytest.raises(ValueError):scene.foreground(replace(object_frame,calibration_id='invalid'))
    bad=replace(c,T_B_C=c.T_B_C@transform(p=[0,0,.015]))
    badframes=[replace(f,calibration_id=bad.calibration_id) for f in frames]
    with pytest.raises(ValueError,match='mismatch'):StaticScene.capture_empty(bad,badframes,cell_confirmed_empty=True)


def test_capture_worker_surfaces_fault_and_stops():
    class BrokenSource:
        def read(self): raise OSError('USB disconnected')
    worker=CaptureWorker(BrokenSource()).start()
    with pytest.raises(RuntimeError,match='worker failed'):worker.take(1.)
    worker.close();assert not worker.thread.is_alive()


def test_capture_worker_is_latest_only_under_load():
    class Source:
        def __init__(self):self.n=0
        def read(self):
            time.sleep(.001);self.n+=1;return self.n
    worker=CaptureWorker(Source()).start()
    try:
        time.sleep(.04)
        value=worker.take(.5)
        assert value>=2 and worker.buffer.dropped>=1
    finally:worker.close()


@pytest.mark.parametrize('width,height,z',[(1280,720,.52),(848,480,.35),(640,480,.32),(640,360,.26)])
def test_d455_min_z_is_profile_specific(width,height,z):
    assert profile_min_depth(width,height)==z


def test_driver_rejects_unmeasured_extrinsics_before_start():
    with pytest.raises(ValueError,match='measured'):D455Source(make_calibration())

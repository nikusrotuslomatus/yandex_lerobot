import numpy as np
import pytest
from active_scan.core import load_config,Failure,transform,points
from active_scan.simulation import SimulatorAdapter
from active_scan.geometry import VisibilityEvaluator,CollisionChecker,GraspPlanner
from active_scan.planning import SplineTrajectory,MotionPlanner,RegraspPlanner
from active_scan.control import TaskPlanner
from active_scan.control import Executor,EventLog
from active_scan.planning import Trajectory,motion_duration


def test_cartesian_spline_does_not_stop_at_interior_ik_nodes():
    cfg=load_config()
    nodes=np.array([[0,0,0,0,0],[.02,.01,-.01,.02,0],[.04,.03,-.02,.04,.01],[.06,.04,-.03,.05,.02]])
    tr=SplineTrajectory(nodes,cfg)
    q0,v0,a0=tr.sample(0);q1,v1,a1=tr.sample(tr.duration)
    np.testing.assert_allclose(q0,nodes[0]);np.testing.assert_allclose(q1,nodes[-1])
    np.testing.assert_allclose([v0,v1,a0,a1],0,atol=1e-12)
    samples=[tr.sample(t) for t in np.linspace(0,tr.duration,4001)]
    assert np.linalg.norm(tr.sample(tr.duration/2)[1])>.01
    assert np.all(np.max(np.abs([s[1] for s in samples]),axis=0)<=cfg['motion']['velocity'])
    assert np.all(np.max(np.abs([s[2] for s in samples]),axis=0)<=cfg['motion']['acceleration'])
    assert tr.duration<1.0


def test_frame_12_regression_gate():
    s=SimulatorAdapter(load_config());v=VisibilityEvaluator(s)
    # Measured from original one-based frame 12 at 640x480, not a fabricated good view.
    old={'distance_m':.7961109562,'bbox_short_px':59.1343215,'visible_pixels':844,
         'visible_fraction':.4143902,'border_padding_px':20.}
    assert set(v.frame_rejections(old))=={'distance','object_scale','visible_pixels','occlusion'}
    s.close()


def test_far_cube_capture_cannot_increment_coverage(tmp_path):
    s=SimulatorAdapter(load_config(),'A');task=TaskPlanner(s,tmp_path)
    camera=s.camera_pose()
    pose=transform(camera[:3,:3],points(camera,[0,-.22,.80]))
    s._set_cube(s.data,pose)
    class G:grasp_id='test_far_cube'
    with pytest.raises(Failure,match='FRAME_QUALITY_REJECTED'):task.capture(G())
    assert task.captures==0 and not task.coverage.quality.any()
    assert not list(tmp_path.glob('frame_*.png'))
    task.log.close();s.close()


def test_rendered_image_scale_and_planning_estimate_agree():
    s=SimulatorAdapter(load_config());v=VisibilityEvaluator(s);camera=s.camera_pose()
    # A front-facing, near cube with an unobstructed silhouette.
    pose=transform(camera[:3,:3],points(camera,[0,0,.24]));s._set_cube(s.data,pose)
    evaluation=v.evaluate();estimate=v.frame_metrics(evaluation=evaluation)
    _,_,seg=s.capture();measured=v.frame_metrics(evaluation=evaluation,segmentation=seg)
    assert not v.frame_rejections(measured)
    assert abs(estimate['visible_pixels']/measured['visible_pixels']-1)<.03
    assert measured['visible_fraction']>.99
    assert measured['bbox_short_px']>100
    s.close()


def test_fast_geometry_queries_match_full_forward_dynamics():
    import mujoco as mj
    cfg=load_config();cfg['planning']['query_kinematics_only']=True
    s=SimulatorAdapter(cfg);v=VisibilityEvaluator(s);cc=CollisionChecker(s)
    grasp=GraspPlanner(s,cc).geometric_candidates()[0][0]
    rng=np.random.default_rng(42)
    for q in [np.array(cfg['robot']['home']),*rng.uniform(s.limits[:,0],s.limits[:,1],(20,5))]:
        d=s.query_state(q,grasp.T_G_O,gripper=grasp.gripper)
        fast_pose=s.tcp_pose(d);fast_cube=s.cube_pose(d);fast_quality=v.evaluate(d)[0];fast_bad,_=cc.check(d,'HELD')
        mj.mj_forward(s.model,d)
        np.testing.assert_allclose(s.tcp_pose(d),fast_pose,atol=1e-12)
        np.testing.assert_allclose(s.cube_pose(d),fast_cube,atol=1e-12)
        np.testing.assert_allclose(v.evaluate(d)[0],fast_quality,atol=1e-12)
        assert [(b['a'],b['b']) for b in cc.check(d,'HELD')[0]]==[(b['a'],b['b']) for b in fast_bad]
    s.close()


def test_flip_plan_changes_support_without_mutating_live_cube():
    s=SimulatorAdapter(load_config(),'B');cc=CollisionChecker(s);motion=MotionPlanner(s,cc)
    grasp=GraspPlanner(s,cc).geometric_candidates()[0][0]
    s.data.qpos[s.gq]=grasp.gripper;s._set_cube(s.data,s.tcp_pose()@grasp.T_G_O)
    before=s.cube_pose().copy();planner=RegraspPlanner(s,motion);planner.support_axis_before_pick=2
    pose,upper,route,lower=planner.flip_placement(s.state().q_arm,grasp.T_G_O)
    np.testing.assert_array_equal(s.cube_pose(),before)
    assert int(np.argmax(np.abs(pose[2,:3])))!=2
    assert abs(pose[2,3]-.031)<.002
    assert route and lower
    actual=s.cube_pose(s.query_state(lower[-1].goal,grasp.T_G_O,gripper=grasp.gripper))
    np.testing.assert_allclose(actual,pose,atol=1e-7)
    s.close()


def test_graph_finds_more_than_one_safe_intermediate_pose():
    s=SimulatorAdapter(load_config());cc=CollisionChecker(s);motion=MotionPlanner(s,cc)
    # Regression: side-placement retreat -> top pregrasp needs home AND a safe waypoint.
    start=np.array([-.28681173,.59127206,.86701475,-1.6479,1.56704976])
    goal=np.array([-.35586819,.50443448,-.59397763,1.6480,.13016340])
    cube=transform(np.array([[-.4235549,0,-.9058704],[.9058704,0,-.4235549],[0,-1,0]]),[.29986,.09996,.029892])
    # Exact orthonormal matrix for the test setup.
    from scipy.spatial.transform import Rotation
    cube[:3,:3]=Rotation.from_matrix(cube[:3,:3]).as_matrix()
    s.data.qpos[s.qadr]=start;s._set_cube(s.data,cube)
    route=motion.via_home(start,goal,gripper=s.cfg['robot']['open'],mode='TABLE')
    np.testing.assert_allclose(route[0].start,start);np.testing.assert_allclose(route[-1].goal,goal)
    assert len(route)>=3 or len(route[0].waypoints)>=4
    s.close()


def test_interpolated_reference_tracks_without_staircase_speed_spikes(tmp_path):
    cfg=load_config();cfg['motion'].update(continuous_reference=True,reference_limit_fraction=.98,enforce_measured_velocity=True)
    s=SimulatorAdapter(cfg,'B');log=EventLog(tmp_path,s);ex=Executor(s,CollisionChecker(s),log)
    start=s.state().q_arm;goal=start.copy();goal[0]=.4
    tr=Trajectory(start,goal,motion_duration(goal-start,cfg))
    ex.execute([tr],'TABLE')
    assert ex.velocity_max[0]<=cfg['motion']['velocity'][0]
    assert max(np.abs(ex.errors))<.01
    refs=np.array(log.samples)[:,1:6]
    # A held 50 Hz staircase has many identical commands; interpolated motion does not.
    assert np.count_nonzero(np.diff(refs[:,0])>1e-9)>tr.duration/cfg['motion']['dt']*.95
    assert len(log.dynamics_samples)==ex.ticks
    log.close();s.close()


def test_measured_speed_guard_stops_execution(tmp_path):
    cfg=load_config();cfg['motion'].update(enforce_measured_velocity=True,velocity=[.01]*5)
    s=SimulatorAdapter(cfg,'B');log=EventLog(tmp_path,s);ex=Executor(s,CollisionChecker(s),log)
    goal=s.state().q_arm;goal[0]=.2;s.set_joint_targets(goal)
    with pytest.raises(Failure,match='MEASURED_SPEED_LIMIT'):
        for _ in range(100):ex.tick('TABLE')
    log.close();s.close()


def test_new_trajectory_preserves_previous_command_at_start(tmp_path):
    cfg=load_config();cfg['motion'].update(continuous_reference=True,reference_reconcile_time=.12)
    s=SimulatorAdapter(cfg,'A');log=EventLog(tmp_path,s);ex=Executor(s,CollisionChecker(s),log)
    start=s.state().q_arm;previous=start.copy();previous[0]+=.001;s.set_joint_targets(previous)
    goal=start.copy();goal[0]+=.1;tr=Trajectory(start,goal,motion_duration(goal-start,cfg))
    ex.execute([tr],'TABLE')
    first=log.samples[0][1:6]
    assert abs(first[0]-previous[0])<1e-6
    np.testing.assert_allclose(s.data.ctrl[s.aids],goal)
    log.close();s.close()

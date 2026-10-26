import copy
import json
import numpy as np
import pytest
from active_scan.core import load_config,Failure,transform
from active_scan.simulation import SimulatorAdapter
from active_scan.geometry import CollisionChecker
from active_scan.planning import MotionPlanner,BrakeTrajectory,SplineTrajectory
from active_scan.control import Executor,EventLog,TaskPlanner
from active_scan.library import PlanningLibrary
from active_scan.grasp import ObjectObservation,CameraObserver,verify_test_lift


def test_vectorized_spline_matches_scalar_samples():
    tr=SplineTrajectory(np.array([[0]*5,[.1,.05,-.1,.02,.1],[.3,.2,-.2,.1,.3]]),load_config())
    times=np.linspace(0,tr.duration,101);batch=tr.sample(times)
    for i,t in enumerate(times):
        for many,one in zip(batch,tr.sample(t)):np.testing.assert_allclose(many[i],one,atol=1e-13)


def test_camera_evidence_without_contact_sensors():
    before=ObjectObservation(1.,transform(p=[.2,0,.03]),'camera')
    after=ObjectObservation(1.1,transform(p=[.2,0,.04]),'camera')
    observer=CameraObserver(lambda:after)
    assert observer.observe(1.12) is after
    result=verify_test_lift(before,after,transform(),transform(p=[0,0,.01]),.006,.004)
    assert result['source']=='camera' and result['slip']<1e-9
    with pytest.raises(Failure,match='GRASP_EVIDENCE_STALE'):observer.observe(2.)
    with pytest.raises(Failure,match='TEST_LIFT_FAILED'):
        verify_test_lift(before,ObjectObservation(1.1,before.pose,'camera'),transform(),transform(p=[0,0,.01]),.006,.004)
    with pytest.raises(Failure,match='GRASP_EVIDENCE_UNAVAILABLE'):CameraObserver(lambda:None).observe(1.)


def test_brake_starts_at_measured_velocity_and_ends_at_rest():
    velocity=np.array([.6,-.2,.1,0,.3]);q=np.zeros(5)
    tr=BrakeTrajectory(q,velocity,[3]*5)
    np.testing.assert_allclose(tr.sample(0)[0],q);np.testing.assert_allclose(tr.sample(0)[1],velocity)
    np.testing.assert_allclose(tr.sample(tr.duration)[0],tr.goal);np.testing.assert_allclose(tr.sample(tr.duration)[1:],0,atol=1e-12)
    for t in np.linspace(0,tr.duration,101):assert np.max(np.abs(tr.sample(t)[2]))<=3+1e-12


def test_controlled_stop_executes_and_keeps_torque_limits(tmp_path):
    cfg=load_config('configs/tuned.json');s=SimulatorAdapter(cfg,'B');cc=CollisionChecker(s);log=EventLog(tmp_path,s);ex=Executor(s,cc,log)
    # Establish measured pan motion before injecting an operator stop request.
    q=s.state().q_arm
    for i in range(150):
        target=q.copy();target[0]+=.5*(i+1)*.002;s.set_joint_reference(target,[.5,0,0,0,0]);ex.tick('TABLE')
    assert s.state().dq_arm[0]>.3
    ex.controlled_stop('OPERATOR_STOP')
    assert ex.stop_result['method']=='controlled_brake'
    assert np.max(np.abs(s.state().dq_arm))<.05
    assert np.all(ex.force_max<=np.max(np.abs(s.model.actuator_forcerange),axis=1)+1e-10)
    log.close();s.close()


def test_blocked_brake_is_reported_as_fallback(tmp_path):
    cfg=load_config('configs/tuned.json');s=SimulatorAdapter(cfg,'B');log=EventLog(tmp_path,s);cc=CollisionChecker(s);ex=Executor(s,cc,log)
    s.data.qvel[s.vadr[0]]=.4
    cc.check=lambda *args,**kwargs:([{'blocked':True}],-.01)
    ex.controlled_stop('COLLISION_EXECUTION')
    assert ex.stop_result['method']=='hold_fallback' and ex.stop_result['reason']=='BRAKE_PATH_BLOCKED'
    assert s.data.time==0
    log.close();s.close()


def test_missing_library_fails_before_robot_motion(tmp_path):
    cfg=load_config();cfg['library']={'enabled':True,'prepare':False,'directory':str(tmp_path/'cache')}
    s=SimulatorAdapter(cfg,'B');task=TaskPlanner(s,tmp_path/'episode');result=task.run()
    assert result['failure']['code']=='LIBRARY_NOT_PREPARED'
    assert result['simulated_seconds']==0 and result['captures']==0


def test_library_invalidates_changed_camera(tmp_path):
    cfg=load_config();cfg['library']={'enabled':True,'prepare':True,'directory':str(tmp_path)}
    s=SimulatorAdapter(cfg);a=PlanningLibrary(s);a.put('test',{'value':1})
    changed=copy.deepcopy(cfg);changed['scene']['camera_position'][0]+=.001
    other=SimulatorAdapter(changed);b=PlanningLibrary(other)
    assert a.fingerprint!=b.fingerprint and b.get('test') is None
    s.model.geom_size[s.cubegeom,0]+=.001
    assert PlanningLibrary(s).fingerprint!=a.fingerprint
    s.close();other.close()


def test_path_certificate_rechecks_changed_object(tmp_path):
    cfg=load_config();cfg['library']={'enabled':True,'prepare':True,'directory':str(tmp_path)}
    s=SimulatorAdapter(cfg);cc=CollisionChecker(s);motion=MotionPlanner(s,cc);motion.library=PlanningLibrary(s)
    start=s.state().q_arm;goal=start.copy();goal[0]+=.1
    motion.plan_joint_motion(start,goal,mode='TABLE')
    hits=motion.library.certificate_hits
    motion.plan_joint_motion(start,goal,mode='TABLE')
    assert motion.library.certificate_hits==hits+1
    s._set_cube(s.data,s.tcp_pose())
    with pytest.raises(Failure,match='COLLISION_PATH'):motion.plan_joint_motion(start,goal,mode='TABLE')
    s.close()


def test_spline_certificate_does_not_hide_object_collision(tmp_path):
    cfg=load_config();cfg['library']={'enabled':True,'prepare':True,'directory':str(tmp_path)}
    s=SimulatorAdapter(cfg);cc=CollisionChecker(s);motion=MotionPlanner(s,cc);motion.library=PlanningLibrary(s)
    start=s.state().q_arm;middle=start.copy();middle[0]=.15;goal=start.copy();goal[0]=.3
    nodes=[start,middle,goal]
    motion.plan_spline(nodes,mode='TABLE');hits=motion.library.certificate_hits
    motion.plan_spline(nodes,mode='TABLE');assert motion.library.certificate_hits==hits+1
    goal[0]+=1e-10
    motion.plan_spline(nodes,mode='TABLE');assert motion.library.certificate_hits==hits+2
    s._set_cube(s.data,s.tcp_pose())
    with pytest.raises(Failure,match='COLLISION_PATH'):motion.plan_spline(nodes,mode='TABLE')
    s.close()


def test_cartesian_template_refines_a_changed_target(tmp_path):
    cfg=load_config();cfg['library']={'enabled':True,'prepare':True,'directory':str(tmp_path)}
    s=SimulatorAdapter(cfg);motion=MotionPlanner(s,CollisionChecker(s));motion.library=PlanningLibrary(s)
    start=s.state().q_arm;target=s.tcp_pose();target[2,3]+=.005
    motion.cartesian(start,target,mode='TABLE',orientation=False)
    target[2,3]+=.001
    paths=motion.cartesian(start,target,mode='TABLE',orientation=False)
    assert np.linalg.norm(motion.k.forward(paths[-1].goal)[:3,3]-target[:3,3])<.00005
    s.close()


def test_acceleration_profile_and_force_caps_preserved():
    baseline=load_config('configs/efficient.json');factory=load_config('configs/factory.json')
    assert factory['motion']['acceleration']==baseline['motion']['acceleration']
    assert factory['motion']['velocity']==baseline['motion']['velocity']
    for field in ['force_limit','gripper_force_limit']:assert factory['robot'][field]==baseline['robot'][field]
    assert factory['physics']==baseline['physics']


def test_aabb_broadphase_matches_original_narrowphase():
    cfg=load_config();s=SimulatorAdapter(cfg);cc=CollisionChecker(s)
    rng=np.random.default_rng(173)
    for q in rng.uniform(s.limits[:,0],s.limits[:,1],(100,5)):
        d=s.query_state(q,gripper=rng.uniform(.3,1.4))
        for mode in ['TABLE','GRASP','HELD','PLACE']:
            s.cfg['planning']['aabb_broadphase']=False;expected=cc.check(d,mode)[0]
            s.cfg['planning']['aabb_broadphase']=True;actual=cc.check(d,mode)[0]
            assert actual==expected
    s.close()

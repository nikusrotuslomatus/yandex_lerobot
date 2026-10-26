import json
import numpy as np
import pytest
import mujoco as mj
from active_scan.core import load_config,Failure,transform,inverse
from active_scan.simulation import SimulatorAdapter
from active_scan.geometry import CollisionChecker,VisibilityEvaluator,GraspPlanner
from active_scan.planning import Trajectory,motion_duration,MotionPlanner,ViewPlanner
from active_scan.control import TaskPlanner,EventLog,Executor


def test_quintic_endpoints_timing_and_limits():
    cfg=load_config();start=np.zeros(5);end=np.array([.3,-1,.5,.8,-2])
    t=Trajectory(start,end,motion_duration(end,cfg))
    for u,p in [(0,start),(t.duration,end)]:
        q,dq,ddq=t.sample(u);np.testing.assert_allclose(q,p);np.testing.assert_allclose(dq,0,atol=1e-13);np.testing.assert_allclose(ddq,0,atol=1e-13)
    values=np.array([t.sample(u) for u in np.linspace(0,t.duration,2001)])
    assert np.all(np.max(np.abs(values[:,1,:]),axis=0)<=np.array(cfg['motion']['velocity'])+1e-9)
    assert np.all(np.max(np.abs(values[:,2,:]),axis=0)<=np.array(cfg['motion']['acceleration'])+1e-9)
    assert motion_duration(2*end,cfg)>motion_duration(end,cfg)


def test_dynamics_pan_and_no_base_assembly_lock():
    s=SimulatorAdapter(load_config(),'C')
    q=s.state().q_arm;q[0]=.1;s.set_joint_targets(q)
    for _ in range(800):s.step()
    assert s.state().q_arm[0]==pytest.approx(.1,abs=.01)
    assert not s.data.eq_active.any()
    s.close()


def test_weld_attachment_is_explicit_and_reset_clears_it():
    cfg=load_config();s=SimulatorAdapter(cfg,'B')
    # Test-only setup in free space; not a pick experiment or physical grasp result.
    T=s.tcp_pose();s._set_cube(s.data,T);before=s.cube_pose().copy();s.attach('test')
    np.testing.assert_allclose(s.cube_pose(),before)
    assert s.data.eq_active[s.weld]
    q=s.state().q_arm;q[0]+=.1;s.set_joint_targets(q)
    for _ in range(1000):s.step()
    actual=inverse(s.tcp_pose())@s.cube_pose()
    assert np.linalg.norm(actual[:3,3])<.006
    s.reset();assert not s.data.eq_active.any() and s.T_G_O is None
    s.close()


def test_physical_attach_never_activates_constraint_or_moves_cube():
    s=SimulatorAdapter(load_config(),'C');T=s.cube_pose().copy();s.attach('test')
    assert not s.data.eq_active.any();np.testing.assert_array_equal(T,s.cube_pose());s.release();s.close()


def test_settle_timeout_is_bounded(tmp_path):
    cfg=load_config();cfg['motion']['settle_timeout']=.02
    s=SimulatorAdapter(cfg,'C');log=EventLog(tmp_path,s);ex=Executor(s,CollisionChecker(s),log)
    with pytest.raises(Failure,match='SETTLE_TIMEOUT'):ex.settle(s.state().q_arm+.1,'TABLE')
    assert s.data.time<.03
    log.close();s.close()


def test_fsm_failure_records_state(tmp_path):
    s=SimulatorAdapter(load_config(),'A');task=TaskPlanner(s,tmp_path)
    task.gp.geometric_candidates=lambda **kwargs:([],{'failure':'test_injected'})
    result=task.run()
    assert not result['success'] and result['fsm']=='FAILED'
    assert result['failure']['code']=='GRIPPER_OPENING_INSUFFICIENT'
    events=[json.loads(x) for x in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert events[-1]['target']=='FAILED' and events[-3]['target']=='RECOVERY'


def test_view_seed_reproducibility_and_collision_rejection():
    cfg=load_config();cfg['planning']['candidates']=64
    s=SimulatorAdapter(cfg);cc=CollisionChecker(s);v=VisibilityEvaluator(s);gp=GraspPlanner(s,cc)
    grasp=gp.geometric_candidates()[0][0];vp=ViewPlanner(s,cc,v)
    a,stats=vp.generate_candidates(grasp);b,_=vp.generate_candidates(grasp)
    assert len(a)>0 and stats['rejected']['height']>0
    np.testing.assert_array_equal([x.q for x in a],[x.q for x in b])
    np.testing.assert_array_equal([x.quality for x in a],[x.quality for x in b])
    s.close()


def test_path_detects_collision_between_free_endpoints():
    s=SimulatorAdapter(load_config());cc=CollisionChecker(s);p=MotionPlanner(s,cc)
    home=s.state().q_arm;goal=home.copy();goal[0]=.6
    original=cc.check
    def blocked(d=None,mode='FREE',clearance=None):
        if .25<float(d.qpos[s.qadr[0]])<.35:return [{'a':'test','b':'test','distance':-.01}],-.01
        return [],1.
    cc.check=blocked
    with pytest.raises(Failure,match='COLLISION_PATH'):p.plan_joint_motion(home,goal)
    cc.check=original;s.close()


@pytest.mark.parametrize('stage,code',[
    ('pick','PHYSICAL_GRASP_UNSTABLE'),('pick','INITIAL_GRASP_UNREACHABLE'),
    ('observe','GRASP_SLIP'),('observe','SETTLE_TIMEOUT'),
    ('place','UNSTABLE_PLACEMENT'),('place','FLIP_KINEMATICALLY_INFEASIBLE')])
def test_fsm_operation_failure_paths(tmp_path,stage,code):
    s=SimulatorAdapter(load_config(),'A');task=TaskPlanner(s,tmp_path)
    task.gp.geometric_candidates=lambda **kwargs:([object()],{'test':True})
    task.pick=lambda *args,**kwargs:object()
    task.observe=lambda *args:None;task.place=lambda **kwargs:None
    def fail(*args,**kwargs):raise Failure(code,{'injected':True})
    setattr(task,stage,fail)
    result=task.run()
    assert result['failure']['code']==code and result['fsm']=='FAILED'
    assert not result['success'] and result['captures']==0


def test_existing_results_are_preserved(tmp_path):
    s=SimulatorAdapter(load_config());log=EventLog(tmp_path,s);log.close()
    with pytest.raises(FileExistsError,match='Existing episode preserved'):EventLog(tmp_path,s)
    s.close()


def test_live_slip_supervision(tmp_path):
    s=SimulatorAdapter(load_config(),'C');cc=CollisionChecker(s)
    grasp=GraspPlanner(s,cc).geometric_candidates()[0][0]
    d=s.query_state(s.state().q_arm,grasp.T_G_O,gripper=1.4)
    s._set_cube(s.data,s.cube_pose(d))
    log=EventLog(tmp_path,s);ex=Executor(s,cc,log)
    ex.expected_relative=inverse(s.tcp_pose())@s.cube_pose();ex.expected_relative[0,3]-=.05
    with pytest.raises(Failure,match='GRASP_SLIP'):ex.tick('HELD')
    log.close();s.close()

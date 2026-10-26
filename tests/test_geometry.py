import numpy as np
import pytest
import mujoco as mj
from scipy.spatial.transform import Rotation
from active_scan.core import *
from active_scan.simulation import SimulatorAdapter
from active_scan.geometry import *


@pytest.fixture(scope='module')
def sim():
    s=SimulatorAdapter(load_config())
    yield s
    s.close()


def test_se3_composition_and_quaternion():
    a=transform(Rotation.from_euler('xyz',[.2,-.3,.7]).as_matrix(),[.1,.2,.3])
    b=transform(Rotation.from_euler('y',.6).as_matrix(),[-.4,.8,.3])
    p=np.array([[.1,.5,-.2]])
    np.testing.assert_allclose(a@inverse(a),np.eye(4),atol=1e-14)
    np.testing.assert_allclose(points(a@b,p),points(a,points(b,p)),atol=1e-14)
    np.testing.assert_allclose(Rotation.from_quat(quat_wxyz(a[:3,:3]),scalar_first=True).as_matrix(),a[:3,:3])


def test_named_indices_and_mass(sim):
    assert len(sim.qadr)==5 and sim.model.nu==6 and sim.model.nq==13 and sim.model.nv==12
    assert sim.model.jnt_qposadr[sim.model.joint('cube_free').id] not in sim.qadr
    np.testing.assert_allclose(sim.model.geom_size[sim.cubegeom],[.03]*3)
    np.testing.assert_allclose(sim.model.body_inertia[sim.cube],np.ones(3)*.12*.06**2/6)


def test_fk_and_jacobian(sim):
    k=Kinematics(sim); q=np.array(sim.cfg['robot']['home'])
    np.testing.assert_allclose(k.forward(q),sim.tcp_pose(),atol=1e-12)
    J=k.jacobian(q)
    assert J.shape==(6,5)
    for j in range(5):
        delta=np.zeros(5);delta[j]=1e-6
        a=k.forward(q+delta); b=k.forward(q-delta)
        np.testing.assert_allclose((a[:3,3]-b[:3,3])/2e-6,J[:3,j],atol=2e-8)
        np.testing.assert_allclose(Rotation.from_matrix(a[:3,:3]@b[:3,:3].T).as_rotvec()/2e-6,J[3:,j],atol=2e-8)


def test_ik_reachable_unreachable_and_continuity(sim):
    k=Kinematics(sim);q=np.array(sim.cfg['robot']['home'])
    target=k.forward(q+np.array([.05,.02,-.03,.01,.04]))
    result=k.solve_ik(target,q)
    assert np.linalg.norm(k.forward(result)[:3,3]-target[:3,3])<.002
    assert np.linalg.norm(result-q)<.2
    target[:3,3]=[4,4,4]
    with pytest.raises(Failure,match='TARGET_POSE_UNREACHABLE'): k.solve_ik(target,q)


def test_surface_identity_and_stable_orientations():
    p,n,f=cube_surface(.06,12)
    assert p.shape==(864,3) and len(np.unique(f))==6
    np.testing.assert_allclose(np.linalg.norm(n,axis=1),1)
    np.testing.assert_allclose(np.max(np.abs(p),axis=1),.03)
    rots=stable_cube_rotations()
    assert len(rots)==24
    for R in rots: np.testing.assert_allclose(R.T@R,np.eye(3))


def test_camera_axes_projection_and_rendering(sim):
    v=VisibilityEvaluator(sim); cam=sim.camera_pose()
    center=points(cam,[0,0,.35])
    uv,z=v.project(center[None])
    np.testing.assert_allclose(uv[0],sim.K[:2,2],atol=1e-10)
    assert z[0]>.3
    assert v.project(points(cam,[[0,0,-.2]]))[1][0]<0
    # Render a cube in front of camera; compare predicted centre pixel against segmentation.
    sim.reset();T=transform(cam[:3,:3],center)
    sim._set_cube(sim.data,T)
    rgb,depth,seg=sim.capture()
    u,vv=np.rint(uv[0]).astype(int)
    assert tuple(rgb.shape)==(480,640,3)
    assert seg[vv,u,0]==sim.cubegeom
    assert depth[vv,u]==pytest.approx(.35-.03,abs=.001)
    sim.reset()


def test_visibility_front_two_back_offscreen_and_clipped(sim):
    sim.reset();v=VisibilityEvaluator(sim);cam=sim.camera_pose()
    center=points(cam,[0,0,.22])
    T=transform(cam[:3,:3],center)
    sim._set_cube(sim.data,T)
    q,dirs,_=v.evaluate()
    assert np.mean(q[v.faces==5]>.35)>.95  # -Z local faces camera
    assert np.all(q[v.faces==4]==0)
    T[:3,:3]=cam[:3,:3]@Rotation.from_euler('y',.65).as_matrix()
    sim._set_cube(sim.data,T); q,_,_=v.evaluate()
    assert np.any(q[v.faces==0]>0) and np.any(q[v.faces==5]>0)
    T[:3,3]=points(cam,[3,0,.22]);sim._set_cube(sim.data,T)
    assert not np.any(v.evaluate()[0])
    T[:3,:3]=cam[:3,:3];T[:3,3]=points(cam,[.025,0,.065]);sim._set_cube(sim.data,T)
    q,_,_=v.evaluate()
    assert np.count_nonzero(q[v.faces==5])<144
    sim.reset()


def test_finger_occlusion(sim):
    sim.reset(); v=VisibilityEvaluator(sim)
    gp=GraspPlanner(sim,CollisionChecker(sim));grasps,_=gp.geometric_candidates()
    assert grasps
    robot_q=np.array(sim.cfg['robot']['home']);robot_q[4]=1.0
    d=sim.query_state(robot_q,grasps[0].T_G_O,gripper=grasps[0].gripper)
    q,_,occ=v.evaluate(d)
    T=sim.cube_pose(d);world=points(T,v.local);norm=v.normals@T[:3,:3].T
    dirs=sim.data.cam_xpos[sim.camera]-world;dirs/=np.linalg.norm(dirs,axis=1)[:,None]
    front=(norm*dirs).sum(axis=1)>0
    assert np.any(front&occ)
    assert np.all(q[occ]==0)


def test_coverage_diversity():
    c=CoverageState(2,.35,20)
    c.update(np.array([.7,.1]),np.array([[1,0,0],[1,0,0]]))
    c.update(np.array([.6,.8]),np.array([[1,0,0],[1,0,0]]))
    assert len(c.directions[0])==1
    c.update(np.array([.5,.2]),np.array([[0,1,0],[1,0,0]]))
    np.testing.assert_allclose(c.quality,[.7,.8]);assert len(c.directions[0])==2


def test_collision_and_reset(sim):
    sim.reset();cc=CollisionChecker(sim)
    assert not cc.check(mode='TABLE')[0]
    T=sim.cube_pose();T[2,3]=-.02;sim._set_cube(sim.data,T)
    assert any(x['a']=='table' or x['b']=='table' for x in cc.check(mode='TABLE')[0])
    sim.attach('test');sim.reset()
    assert sim.T_G_O is None and sim.grasp_id is None and not sim.data.eq_active.any()
    np.testing.assert_array_equal(sim.data.qvel,0)
    np.testing.assert_allclose(sim.cube_pose()[:3,3],sim.cfg['scene']['cube_position'])

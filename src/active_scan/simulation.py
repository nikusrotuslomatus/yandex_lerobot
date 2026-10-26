"""Named-index MuJoCo adapter; geometry queries never mutate the live simulator."""
import copy
import xml.etree.ElementTree as ET
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation
from .core import ROOT, SceneState, Failure, transform, inverse, points, quat_wxyz, look_at, optical_pose


def _s(values):
    return " ".join(str(float(v)) for v in values)


def scene_xml(cfg):
    root = ET.parse(ROOT / cfg['robot']['model']).getroot()
    root.attrib['model'] = 'so101_active_scan'
    model_dir = (ROOT / cfg['robot']['model']).parent
    compiler = root.find('compiler')
    compiler.set('meshdir', str((model_dir / 'assets').resolve()))
    opt = root.find('option')
    opt.set('timestep', str(cfg['motion']['dt']))
    opt.set('iterations', str(cfg['physics']['solver_iterations']))
    if cfg['robot']['remove_wrist_camera']:
        grip = root.find('.//body[@name="gripper"]')
        grip.remove(grip.find('body[@name="camera_mount"]'))
    else:
        grip = root.find('.//body[@name="gripper"]')
    # G shares axes with the gripper body; translation is an explicit hypothesis.
    ET.SubElement(grip, 'site', name='scan_tcp', pos=_s(cfg['robot']['tcp_local']), size='.003', group='5')
    for i, geom in enumerate(root.findall('.//worldbody//geom')):
        if 'name' not in geom.attrib:
            geom.set('name', f'robot_geom_{i:03d}')
    for pos in root.findall('.//default[@class="sts3215"]/position'):
        pos.set('kp', str(cfg['robot']['kp'])); pos.set('kv', str(cfg['robot']['kv']))
        f = cfg['robot']['force_limit']; pos.set('forcerange', _s([-f, f]))
    finger_actuator=root.find('.//actuator/position[@name="gripper"]')
    gf=cfg['robot']['gripper_force_limit']
    finger_actuator.set('forcerange',_s([-gf,gf]))
    for geom in root.findall('.//default[@class="collision_gripper"]/geom') + root.findall('.//default[@class="collision_gripper_mesh"]/geom'):
        geom.set('friction', _s([cfg['physics']['finger_friction'], cfg['physics']['spin_friction'], cfg['physics']['roll_friction']]))
    world = root.find('worldbody')
    ET.SubElement(world, 'geom', name='table', type='box', pos='0 0 -0.025', size=_s(cfg['scene']['table_halfsize']), rgba='.55 .58 .61 1', friction=_s([cfg['physics']['table_friction'], .005, .0001]))
    # Original base has no collision geoms. This conservative box is an explicit addition.
    ET.SubElement(root.find('.//body[@name="base"]'), 'geom', name='base_collision', type='box', pos=_s(cfg['robot']['base_collision_position']), size=_s(cfg['robot']['base_collision_halfsize']), rgba='.5 .5 .5 0', group='3')
    contact = ET.SubElement(root, 'contact')
    # Parent-child assembly overlap; also excluded explicitly in CollisionChecker.
    ET.SubElement(contact, 'exclude', body1='base', body2='shoulder')
    ET.SubElement(world, 'geom', name='camera_pole', type='cylinder', pos=_s(cfg['scene']['pole_position']), size=_s([cfg['scene']['pole_radius'], cfg['scene']['pole_halfheight']]), rgba='.2 .2 .2 1')
    camT = look_at(cfg['scene']['camera_position'], np.array(cfg['scene']['camera_target']))
    camera = cfg['camera']
    ET.SubElement(world, 'camera', name='scan_camera', pos=_s(camT[:3, 3]), quat=_s(quat_wxyz(camT[:3, :3])), fovy=str(camera['fovy']))
    # Housing behind the optical centre, so it cannot self-occlude forward rays.
    ET.SubElement(world, 'geom', name='camera_housing', type='box', pos=_s(points(camT, [0, 0, .026])), quat=_s(quat_wxyz(camT[:3, :3])), size='.045 .023 .023', rgba='.1 .12 .14 1')
    ET.SubElement(world, 'light', pos='.3 -.4 .8', dir='-.2 .2 -1', diffuse='.8 .8 .8')
    cube = ET.SubElement(world, 'body', name='cube', pos=_s(cfg['scene']['cube_position']), quat=_s(Rotation.from_euler('xyz', cfg['scene']['cube_rpy']).as_quat(scalar_first=True)))
    ET.SubElement(cube, 'freejoint', name='cube_free')
    asset = root.find('asset')
    ET.SubElement(asset, 'texture', name='wood_texture', type='2d', builtin='checker', rgb1='.62 .39 .20', rgb2='.68 .44 .24', width='256', height='256')
    ET.SubElement(asset, 'material', name='wood', texture='wood_texture', texrepeat='3 12', specular='.05', shininess='.05')
    ET.SubElement(cube, 'geom', name='cube_geom', type='box', size=_s([cfg['scene']['cube_size']/2]*3), mass=str(cfg['scene']['cube_mass']), material='wood', friction=_s([cfg['physics']['table_friction'], .005, .0001]))
    equality = ET.SubElement(root, 'equality')
    ET.SubElement(equality, 'weld', name='ideal_grasp', body1='gripper', body2='cube', active='false', solref='.008 1')
    visual = root.find('visual')
    ET.SubElement(visual, 'global', offwidth=str(camera['width']), offheight=str(camera['height']))
    # MuJoCo clip distances scale with model extent; adapter sets compiled values.
    return ET.tostring(root, encoding='unicode')


class SimulatorAdapter:
    def __init__(self, cfg, level='A'):
        if level not in ('A', 'B', 'C'):
            raise ValueError(level)
        self.cfg, self.level = copy.deepcopy(cfg), level
        self.xml = scene_xml(cfg)
        self.model = m = mj.MjModel.from_xml_string(self.xml)
        self.data = d = mj.MjData(m)
        self.query = mj.MjData(m)
        self.jids = np.array([m.joint(n).id for n in cfg['robot']['joint_names']])
        self.qadr = m.jnt_qposadr[self.jids]; self.vadr = m.jnt_dofadr[self.jids]
        self.aids = np.array([m.actuator(n).id for n in cfg['robot']['joint_names']])
        self.gj = m.joint('gripper').id; self.ga = m.actuator('gripper').id
        self.gq = m.jnt_qposadr[self.gj]; self.gv = m.jnt_dofadr[self.gj]
        self.cq = m.jnt_qposadr[m.joint('cube_free').id]; self.cv = m.jnt_dofadr[m.joint('cube_free').id]
        self.tcp = m.site('scan_tcp').id; self.cube = m.body('cube').id
        self.grip = m.body('gripper').id; self.moving = m.body('moving_jaw_so101_v1').id
        self.cubegeom = m.geom('cube_geom').id; self.table = m.geom('table').id
        self.camera = m.camera('scan_camera').id; self.weld = m.equality('ideal_grasp').id
        self.limits = m.jnt_range[self.jids].copy()
        fy = cfg['camera']['height'] / (2*np.tan(np.deg2rad(cfg['camera']['fovy'])/2))
        self.K = np.array([[fy, 0, cfg['camera']['width']/2], [0, fy, cfg['camera']['height']/2], [0, 0, 1.]])
        m.vis.map.znear = cfg['camera']['near']/m.stat.extent
        m.vis.map.zfar = cfg['camera']['far']/m.stat.extent
        self.grasp_id = None; self.T_G_O = None; self.renderer = None
        self.controller=cfg.get('controller',{})
        self.configure_arm_gains(False)
        if 'gripper_kp' in self.controller:
            kp=self.controller['gripper_kp'];kv=self.controller['gripper_kv']
            m.actuator_gainprm[self.ga,0]=kp;m.actuator_biasprm[self.ga,1:3]=[-kp,-kv]
        self.reset()

    def configure_arm_gains(self,loaded):
        cfg=self.cfg.get('controller',{});profile=cfg.get('loaded' if loaded else 'empty',{})
        self.arm_kp=np.broadcast_to(profile.get('kp',self.cfg['robot']['kp']),(5,)).copy()
        self.arm_kv=np.broadcast_to(profile.get('kv',self.cfg['robot']['kv']),(5,)).copy()
        self.model.actuator_gainprm[self.aids,0]=self.arm_kp
        self.model.actuator_biasprm[self.aids,1]=-self.arm_kp
        self.model.actuator_biasprm[self.aids,2]=-self.arm_kv

    def reset(self):
        self.configure_arm_gains(False)
        mj.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = self.cfg['robot']['home']
        self.data.qpos[self.gq] = self.cfg['robot']['open']
        self.data.ctrl[self.aids] = self.data.qpos[self.qadr]
        self.command_reference=self.data.qpos[self.qadr].copy()
        self.data.ctrl[self.ga] = self.data.qpos[self.gq]
        self.data.eq_active[:] = False
        self.data.qvel[:] = 0
        self.grasp_id = None; self.T_G_O = None
        mj.mj_forward(self.model, self.data)

    def tcp_pose(self, data=None):
        d = data or self.data
        return transform(d.site_xmat[self.tcp].reshape(3, 3).copy(), d.site_xpos[self.tcp].copy())

    def cube_pose(self, data=None):
        d = data or self.data
        return transform(d.xmat[self.cube].reshape(3, 3).copy(), d.xpos[self.cube].copy())

    def camera_pose(self):
        return optical_pose(transform(self.data.cam_xmat[self.camera].reshape(3, 3), self.data.cam_xpos[self.camera]))

    def state(self):
        d = self.data
        obstacles={n:{'T_world':transform(d.geom_xmat[self.model.geom(n).id].reshape(3,3).copy(),d.geom_xpos[self.model.geom(n).id].copy()),'size':self.model.geom(n).size.copy(),'type':int(self.model.geom(n).type[0])} for n in ('table','camera_pole','camera_housing')}
        return SceneState(float(d.time), d.qpos[self.qadr].copy(), d.qvel[self.vadr].copy(), float(d.qpos[self.gq]), transform(), self.tcp_pose(), self.cube_pose(), self.camera_pose(), self.K.copy(), self.grasp_id, obstacles, d.qvel[self.cv:self.cv+6].copy())

    def observe_object(self):
        from .grasp import ObjectObservation
        observer=getattr(self,'object_observer',None)
        if observer is not None:return observer.observe(float(self.data.time))
        return ObjectObservation(float(self.data.time),self.cube_pose(),'simulation_ground_truth')

    def set_joint_targets(self, q):
        self.set_joint_reference(q)

    def set_joint_reference(self,q,dq=None):
        q = np.asarray(q)
        if not np.all(np.isfinite(q)) or np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            raise Failure('JOINT_LIMIT_VIOLATION')
        self.command_reference=q.copy();target=q.copy()
        if self.controller.get('velocity_feedforward',False) and dq is not None:
            target=target+self.arm_kv/self.arm_kp*np.asarray(dq)
        if self.controller.get('bias_feedforward',False):
            # Feedforward acts THROUGH the bounded actuator, never as an
            # unbounded qfrc_applied force outside motor limits.
            target=target+self.data.qfrc_bias[self.vadr]/self.arm_kp
        self.data.ctrl[self.aids] = np.clip(target,self.limits[:,0],self.limits[:,1])

    def set_gripper_target(self, value):
        self.data.ctrl[self.ga] = np.clip(value, *self.model.jnt_range[self.gj])

    def stop(self):
        self.data.ctrl[self.aids] = self.data.qpos[self.qadr]
        self.data.ctrl[self.ga] = self.data.qpos[self.gq]

    def step(self):
        if self.level == 'A':
            self.data.qpos[self.qadr] = self.data.ctrl[self.aids]
            self.data.qpos[self.gq] = self.data.ctrl[self.ga]
            self.data.qvel[:] = 0
            mj.mj_forward(self.model, self.data)
            if self.T_G_O is not None:
                self._set_cube(self.data, self.tcp_pose() @ self.T_G_O)
            self.data.time += self.model.opt.timestep
        else:
            mj.mj_step(self.model, self.data)
            mj.mj_forward(self.model, self.data)
        if not np.all(np.isfinite(self.data.qpos)) or any(w.number for w in self.data.warning):
            raise Failure('NUMERICAL_INSTABILITY')

    def _set_cube(self, d, T):
        d.qpos[self.cq:self.cq+3] = T[:3, 3]
        d.qpos[self.cq+3:self.cq+7] = quat_wxyz(T[:3, :3])
        d.qvel[self.cv:self.cv+6] = 0
        if d is self.query and self.cfg['planning'].get('query_kinematics_only',False):mj.mj_fwdKinematics(self.model,d)
        else:mj.mj_forward(self.model, d)

    def query_state(self, q, T_G_O=None, gripper=None, cube_pose=None):
        d = self.query
        d.qpos[:] = self.data.qpos; d.qvel[:] = 0
        d.qpos[self.qadr] = q
        if gripper is not None:
            d.qpos[self.gq] = gripper
        if self.cfg['planning'].get('query_kinematics_only',False):mj.mj_fwdKinematics(self.model,d)
        else:mj.mj_forward(self.model, d)
        if T_G_O is not None:
            self._set_cube(d, self.tcp_pose(d) @ T_G_O)
        elif cube_pose is not None:
            self._set_cube(d, cube_pose)
        return d

    def attach(self, grasp_id):
        self.configure_arm_gains(True)
        self.grasp_id = grasp_id
        self.T_G_O = inverse(self.tcp_pose()) @ self.cube_pose()
        if self.level == 'B':
            d, m = self.data, self.model
            rel = inverse(transform(d.xmat[self.grip].reshape(3, 3), d.xpos[self.grip])) @ self.cube_pose()
            # Weld eq_data: anchor (3), relative translation (3), quaternion (4), torque scale (1).
            m.eq_data[self.weld, 3:6] = rel[:3, 3]
            m.eq_data[self.weld, 6:10] = quat_wxyz(rel[:3, :3])
            d.eq_active[self.weld] = True
        if self.level == 'C':
            assert not np.any(self.data.eq_active)

    def release(self):
        self.configure_arm_gains(False)
        self.data.eq_active[self.weld] = False
        self.T_G_O = None; self.grasp_id = None

    def capture_rgbd(self):
        """Sensor-only images; perception must never consume segmentation IDs."""
        if self.renderer is None:
            self.renderer = mj.Renderer(self.model, self.cfg['camera']['height'], self.cfg['camera']['width'])
        r = self.renderer
        options = mj.MjvOption(); options.geomgroup[3:] = False
        r.update_scene(self.data, camera='scan_camera', scene_option=options)
        rgb = r.render().copy()
        r.enable_depth_rendering(); depth = r.render().copy(); r.disable_depth_rendering()
        return rgb, depth

    def capture(self):
        rgb, depth = self.capture_rgbd()
        r = self.renderer
        r.enable_segmentation_rendering(); segmentation = r.render().copy(); r.disable_segmentation_rendering()
        return rgb, depth, segmentation

    def close(self):
        if self.renderer is not None:
            self.renderer.close(); self.renderer = None

    def inventory(self):
        m = self.model
        return {'nq':m.nq, 'nv':m.nv, 'nu':m.nu,
                'joints':[{'name':m.joint(i).name, 'type':int(m.jnt_type[i]), 'range':m.jnt_range[i], 'qposadr':int(m.jnt_qposadr[i]), 'dofadr':int(m.jnt_dofadr[i])} for i in range(m.njnt)],
                'actuators':[{'name':m.actuator(i).name, 'joint':m.joint(int(m.actuator_trnid[i,0])).name, 'ctrlrange':m.actuator_ctrlrange[i], 'forcerange':m.actuator_forcerange[i], 'gain':m.actuator_gainprm[i], 'bias':m.actuator_biasprm[i]} for i in range(m.nu)],
                'bodies':[{'name':m.body(i).name,'mass':m.body_mass[i], 'inertia_diagonal':m.body_inertia[i], 'inertia_quat':m.body_iquat[i]} for i in range(m.nbody)],
                'geoms':[{'name':m.geom(i).name, 'body':m.body(int(m.geom_bodyid[i])).name, 'type':int(m.geom_type[i]), 'size':m.geom_size[i], 'group':int(m.geom_group[i]), 'contype':int(m.geom_contype[i]), 'conaffinity':int(m.geom_conaffinity[i])} for i in range(m.ngeom)],
                'tcp_local':self.cfg['robot']['tcp_local'], 'tcp_world':self.tcp_pose(),
                'camera_optical':self.camera_pose(), 'K':self.K}

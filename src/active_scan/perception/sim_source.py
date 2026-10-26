"""Independent MuJoCo renderer input; cube pose stays outside the estimator."""
import numpy as np
import cv2
import mujoco as mj
from ..core import quat_wxyz
from .types import Calibration, FramePacket
from .robot_mask import SimulatorRobotMask


class MujocoRGBDSource:
    def __init__(self, simulator, noise=.0006, holes=.05, seed=42, mask_robot=True,noise_profile=None,joint_noise_profile=None,link_position_bound_m=0.):
        self.sim=simulator;cfg=simulator.cfg
        # OpenGL pixels are sampled at half-integer window coordinates. Our
        # pinhole API (and RealSense) addresses pixel centers with integers.
        K=simulator.K.copy();K[0,2]-=.5;K[1,2]-=.5
        self.calibration=Calibration(cfg['camera']['width'],cfg['camera']['height'],K,
                                     simulator.camera_pose(),min_depth=.32,max_depth=1.2,
                                     provenance='mujoco',workspace=(.08,.48,-.25,.25,.003,.35))
        self.noise=noise;self.holes=holes;self.rng=np.random.default_rng(seed);self.sequence=0
        self.masker=SimulatorRobotMask(simulator,self.calibration,link_position_bound_m=link_position_bound_m) if mask_robot else None
        self.stereo=None;self.feedback_noise=None
        if noise_profile is not None:
            from .sensor_noise import StereoNoiseModel
            self.stereo=StereoNoiseModel(self.calibration,noise_profile,seed)
        if joint_noise_profile is not None:
            from .feedback_noise import JointNoiseModel
            self.feedback_noise=JointNoiseModel(joint_noise_profile,seed=seed+1)

    def read(self):
        s=self.sim;t=float(s.data.time)
        if self.stereo is None:
            rgb,depth=s.capture_rgbd()
            depth=depth+self.rng.normal(0,self.noise,depth.shape)
            depth[self.rng.random(depth.shape)<self.holes]=0
        else:
            true=self.stereo.true_calibration(t);m=s.model;cam=s.camera
            old_pos=m.cam_pos[cam].copy();old_quat=m.cam_quat[cam].copy()
            old_qpos=s.data.qpos.copy();samples=[]
            try:
                # Constant-velocity exposure quadrature, including free-body
                # motion. No integration step or controller command is issued.
                exposure=self.stereo.profile.exposure_s
                for offset in (-exposure/2,0.,exposure/2):
                    actual=self.stereo.true_calibration(t+offset)
                    m.cam_pos[cam]=actual.T_B_C[:3,3]
                    m.cam_quat[cam]=quat_wxyz(actual.T_B_C[:3,:3]@np.diag([1.,-1.,-1.]))
                    q=old_qpos.copy();mj.mj_integratePos(m,q,s.data.qvel,offset);s.data.qpos[:]=q
                    mj.mj_forward(m,s.data);samples.append(s.capture_rgbd())
                rgb,depth=samples[1]
            finally:
                s.data.qpos[:]=old_qpos;m.cam_pos[cam]=old_pos;m.cam_quat[cam]=old_quat;mj.mj_forward(m,s.data)
            # Reproject the rendered pinhole grid into the true, slightly
            # miscalibrated optics. Native depth keeps nearest samples.
            y,x=np.indices(depth.shape,dtype=np.float32);K=self.calibration.K;kt=true.K
            u=((x-kt[0,2])/kt[0,0]*K[0,0]+K[0,2]).astype(np.float32)
            v=((y-kt[1,2])/kt[1,1]*K[1,1]+K[1,2]).astype(np.float32)
            depth=cv2.remap(depth,u,v,cv2.INTER_NEAREST);rgb=cv2.remap(rgb,u,v,cv2.INTER_LINEAR)
            exposure_depths=[cv2.remap(samples[j][1],u,v,cv2.INTER_NEAREST) for j in (0,2)]
            exposure_colors=[cv2.remap(samples[j][0],u,v,cv2.INTER_LINEAR) for j in (0,2)]
        mask=None
        if self.masker is not None:
            q=np.r_[s.data.qpos[s.qadr],s.data.qpos[s.gq]]
            if self.feedback_noise is not None:
                q,reported=self.feedback_noise.measure(q,t)
                velocity=np.r_[s.data.qvel[s.vadr],s.data.qvel[s.gv]]
                q=q-velocity*(reported-t)
            mask=self.masker.mask(q[:5],float(q[5]))
        frame=FramePacket(self.sequence,t,rgb,depth,self.calibration.calibration_id,'mujoco',robot_mask=mask)
        if self.stereo is not None:
            frame,self.last_noise_diagnostics=self.stereo.apply(frame,t,exposure_depths,exposure_colors,source='mujoco_stereo')
        self.sequence+=1
        return frame

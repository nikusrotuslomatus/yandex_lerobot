"""Bounded object-centric TSDF map. Only accepted RGB-D measurements fuse.

The camera is fixed: no camera odometry or mobile SLAM is required. This backend
reconstructs the manipulated object in its own coordinate frame. Pose errors
are not hidden by a robot FK pose. Cube symmetry cannot establish material-face
identity; fusion stops when a symmetry branch cannot be followed continuously.
"""
import numpy as np
from ..core import inverse
from .types import Status


class ObjectMapper:
    def __init__(self, calibration, size=.06, voxel=.0015, max_keyframes=60,max_fusion_error_m=.003):
        import open3d as o3d
        if o3d.__version__ != '0.19.0':
            raise RuntimeError('Qualified TSDF backend is Open3D 0.19.0; 0.20.0 has a metric-depth activation regression')
        self.o3d=o3d; self.calibration=calibration; self.size=size; self.voxel=voxel
        self.max_keyframes=max_keyframes; self.keyframes=[]; self.epoch=None
        self.volume=self._volume()
        self.max_fusion_error_m=max_fusion_error_m

    def _volume(self):
        return self.o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel,sdf_trunc=4*self.voxel,
            color_type=self.o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    def _integrate(self, frame, estimate):
        c=self.calibration; o3d=self.o3d
        depth=np.where(estimate.mask,frame.depth_m,0).astype(np.float32)
        import cv2
        # Reject flying pixels and the silhouette's mixed-depth border. Median
        # values only reject samples; original depths are never replaced/filled.
        median=cv2.medianBlur(depth,3)
        support=cv2.morphologyEx(estimate.mask.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        support=cv2.erode(support,np.ones((3,3),np.uint8)).astype(bool)
        depth[(np.abs(depth-median)>.0025)|~support]=0
        # Independent model-consistency gate before fusion, in object space.
        from .calibration import rays
        pc=rays(c)*depth[...,None]
        T_O_C=inverse(estimate.pose) @ c.T_B_C
        po=pc @ T_O_C[:3,:3].T+T_O_C[:3,3]
        shell=np.abs(np.max(np.abs(po),axis=-1)-self.size/2)<.0025
        depth[~shell]=0
        if np.count_nonzero(depth)<80: return False
        rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.array(frame.rgb,copy=True)),o3d.geometry.Image(depth),
            depth_scale=1.,depth_trunc=c.max_depth,convert_rgb_to_intensity=False)
        intr=o3d.camera.PinholeCameraIntrinsic(c.width,c.height,c.K[0,0],c.K[1,1],c.K[0,2],c.K[1,2])
        T_C_O=inverse(c.T_B_C) @ estimate.pose
        self.volume.integrate(rgbd,intr,T_C_O)
        if not self.keyframes and len(self.volume.extract_voxel_point_cloud().points)==0:
            raise RuntimeError('TSDF backend produced an empty volume from valid metric depth')
        return True

    def add(self, frame, estimate):
        if not estimate.measured or estimate.mask is None: return False
        if estimate.metrics.get('position_error_bound_m',0)>self.max_fusion_error_m:return False
        if frame.source=='realsense' and not estimate.metrics.get('error_budget_qualified',False):return False
        if (frame.calibration_id!=self.calibration.calibration_id or estimate.calibration_id!=frame.calibration_id
            or frame.sequence!=estimate.sequence or frame.capture_time!=estimate.timestamp
            or frame.source!=estimate.source): raise ValueError('Frame/pose provenance mismatch')
        if self.epoch is not None and self.epoch != estimate.grasp_epoch:
            # A relocalized symmetric cube has no guaranteed material-face match.
            raise ValueError('New grasp/identity epoch requires a new map')
        if len(self.keyframes)>=self.max_keyframes: return False
        if self.keyframes:
            previous=self.keyframes[-1][1].pose
            from scipy.spatial.transform import Rotation
            angle=Rotation.from_matrix(previous[:3,:3].T @ estimate.pose[:3,:3]).magnitude()
            move=np.linalg.norm(previous[:3,3]-estimate.pose[:3,3])
            if angle<.12 and move<.012: return False
        if not self._integrate(frame,estimate): return False
        self.epoch=estimate.grasp_epoch; self.keyframes.append((frame,estimate))
        return True

    def cloud(self): return self.volume.extract_point_cloud()

    def save(self,path):
        if not self.o3d.io.write_point_cloud(str(path),self.cloud()): raise RuntimeError('PLY write failed')

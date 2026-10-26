"""Object-motion evidence works with the existing camera, without contact sensors."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from .core import Failure,inverse


@dataclass
class ObjectObservation:
    timestamp: float
    pose: np.ndarray
    source: str
    confidence: float=1.
    object_id: str='cube'
    grasp_epoch: int=0
    provenance: str='unspecified'
    position_error_bound_m: float | None=None
    relative_error_bound_m: float | None=None
    angular_error_bound_rad: float | None=None
    tool_position_bound_m: float=0.


class CameraObserver:
    """Adapter for an existing camera pose estimator; no MuJoCo contact access.

    The callback supplies a timestamped world-from-object pose. Image-based
    pose estimation and camera calibration must be supplied by the real cell.
    """
    def __init__(self,read_pose,max_age=.15,min_confidence=.8,motion_speed_bound=0.,max_position_error=.003):
        self.read_pose=read_pose;self.max_age=max_age;self.min_confidence=min_confidence
        self.motion_speed_bound=motion_speed_bound;self.max_position_error=max_position_error

    def observe(self,now):
        try:observation=self.read_pose()
        except Failure:raise
        except Exception as error:raise Failure('GRASP_EVIDENCE_UNAVAILABLE') from error
        if not isinstance(observation,ObjectObservation) or observation.source!='camera':raise Failure('GRASP_EVIDENCE_UNAVAILABLE')
        if not 0<=now-observation.timestamp<=self.max_age:raise Failure('GRASP_EVIDENCE_STALE')
        if not np.isfinite(observation.confidence) or not self.min_confidence<=observation.confidence<=1 or np.shape(observation.pose)!=(4,4) or not np.all(np.isfinite(observation.pose)):
            raise Failure('GRASP_EVIDENCE_UNCERTAIN')
        for bound in (observation.position_error_bound_m,observation.relative_error_bound_m,observation.angular_error_bound_rad,observation.tool_position_bound_m):
            if bound is not None and (not np.isfinite(bound) or bound<0):raise Failure('GRASP_EVIDENCE_UNCERTAIN')
        if observation.position_error_bound_m is not None:
            current_bound=observation.position_error_bound_m+self.motion_speed_bound*(now-observation.timestamp)
            if current_bound>self.max_position_error:raise Failure('GRASP_EVIDENCE_UNCERTAIN',{'current_position_bound_m':current_bound})
        return observation


def verify_test_lift(before,after,tool_before,tool_after,min_lift,max_slip,max_angle=.15):
    for pose in (before.pose,after.pose,tool_before,tool_after):
        if np.shape(pose)!=(4,4) or not np.all(np.isfinite(pose)):
            raise Failure('GRASP_EVIDENCE_UNAVAILABLE',{'reason':'nonfinite_or_invalid_pose'})
        R=np.asarray(pose)[:3,:3]
        if not np.allclose(R.T@R,np.eye(3),atol=1e-6) or np.linalg.det(R)<.999 or not np.allclose(pose[3],[0,0,0,1]):
            raise Failure('GRASP_EVIDENCE_UNAVAILABLE',{'reason':'invalid_rigid_pose'})
    if not np.all(np.isfinite([before.timestamp,after.timestamp,min_lift,max_slip,max_angle])) or min(min_lift,max_slip,max_angle)<0:
        raise Failure('GRASP_EVIDENCE_UNAVAILABLE',{'reason':'invalid_time_or_threshold'})
    if before.object_id!=after.object_id or before.grasp_epoch!=after.grasp_epoch:
        raise Failure('GRASP_EVIDENCE_UNAVAILABLE',{'reason':'object_identity_changed'})
    if after.timestamp<=before.timestamp:raise Failure('GRASP_EVIDENCE_STALE')
    lift=float(after.pose[2,3]-before.pose[2,3])
    relative_before=inverse(tool_before)@before.pose
    relative_after=inverse(tool_after)@after.pose
    slip=float(np.linalg.norm((inverse(relative_before)@relative_after)[:3,3]))
    angle=float(np.linalg.norm(Rotation.from_matrix(relative_before[:3,:3].T@relative_after[:3,:3]).as_rotvec()))
    if lift<min_lift or slip>max_slip or angle>max_angle:
        raise Failure('TEST_LIFT_FAILED',{'lift':lift,'slip':slip,'angle':angle,'source':after.source})
    bounds=(before.relative_error_bound_m,after.relative_error_bound_m)
    if any(b is not None for b in bounds):
        if any(b is None or not np.isfinite(b) or b<0 for b in bounds):raise Failure('GRASP_EVIDENCE_UNCERTAIN')
        other=(before.angular_error_bound_rad,after.angular_error_bound_rad,before.tool_position_bound_m,after.tool_position_bound_m)
        if any(b is None or not np.isfinite(b) or b<0 for b in other):raise Failure('GRASP_EVIDENCE_UNCERTAIN')
        uncertainty=sum(bounds)
        tool=before.tool_position_bound_m+after.tool_position_bound_m
        angle_bound=(before.angular_error_bound_rad or 0)+(after.angular_error_bound_rad or 0)
        if lift-uncertainty<min_lift or slip+uncertainty+tool>max_slip or angle+angle_bound>max_angle:
            raise Failure('GRASP_EVIDENCE_UNCERTAIN',{'lift_lower_m':lift-uncertainty,'slip_upper_m':slip+uncertainty+tool})
    return {'lift':lift,'slip':slip,'angle':angle,'source':after.source,'uncertainty_checked':all(b is not None for b in bounds)}

"""Build a timestamp-consistent planner snapshot without simulator truth."""
from dataclasses import dataclass
import numpy as np
from ..core import SceneState, Failure, transform
from .types import rigid, frozen_array


@dataclass(frozen=True)
class MeasuredSceneSnapshot:
    state: SceneState
    object_covariance: np.ndarray
    calibration_id: str
    source: str
    symmetry_ambiguous: bool
    grasp_epoch: int


class MeasuredSceneProvider:
    def __init__(self,pipeline,fixed_obstacles=None):
        self.pipeline=pipeline;self.obstacles={} if fixed_obstacles is None else fixed_obstacles

    def snapshot(self,now,feedback_timestamp,q_arm,dq_arm,gripper,T_B_G,grasp_id=None,max_position_error_m=None):
        estimate=self.pipeline.latest
        if estimate is None:raise Failure('PERCEPTION_UNAVAILABLE')
        estimate.require_measured()
        if not 0<=now-estimate.timestamp<=self.pipeline.max_age:raise Failure('PERCEPTION_STALE')
        if not np.isfinite(feedback_timestamp) or abs(feedback_timestamp-estimate.timestamp)>.005:
            raise Failure('ROBOT_CAMERA_TIME_MISMATCH')
        if not np.isfinite(gripper):raise ValueError('Invalid measured gripper position')
        if estimate.source=='realsense' and not estimate.metrics.get('error_budget_qualified',False):
            raise Failure('PERCEPTION_UNCERTAIN',{'reason':'unqualified_error_budget'})
        tolerance=max_position_error_m if max_position_error_m is not None else (.003 if estimate.source=='realsense' else None)
        if tolerance is not None and estimate.metrics.get('position_error_bound_m',float('inf'))>tolerance:
            raise Failure('PERCEPTION_UNCERTAIN',{'reason':'error_budget_exceeds_tolerance'})
        velocity=estimate.metrics.get('velocity_B')
        if velocity is None:raise Failure('PERCEPTION_VELOCITY_UNAVAILABLE')
        c=self.pipeline.calibration
        state=SceneState(estimate.timestamp,frozen_array(q_arm,(5,)),frozen_array(dq_arm,(5,)),float(gripper),
                         transform(),rigid(T_B_G),estimate.pose,c.T_B_C,c.K,grasp_id,self.obstacles,
                         frozen_array(velocity,(6,)))
        return MeasuredSceneSnapshot(state,estimate.covariance,estimate.calibration_id,estimate.source,
                                     estimate.symmetry_ambiguous,estimate.grasp_epoch)

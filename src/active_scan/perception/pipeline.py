"""Measurement health, temporal continuity and controller evidence boundary."""
from time import perf_counter
import numpy as np
from scipy.spatial.transform import Rotation
from ..core import Failure
from ..grasp import ObjectObservation, CameraObserver
from .localizer import CubeLocalizer, align_symmetry, symmetry_distance
from .types import PoseEstimate, Status


class PerceptionPipeline:
    def __init__(self, calibration, size=.06, max_age=.15, prediction_ttl=.20, static_scene=None,error_budget=None,max_control_error_m=.003):
        self.calibration = calibration; self.localizer = CubeLocalizer(calibration, size,static_scene=static_scene)
        self.max_age = max_age; self.prediction_ttl = prediction_ttl
        self.last = None; self.latest = None; self.last_sequence = -1; self.last_time = -1.
        self.epoch = None; self.confirmations = 0; self.grasp_epoch = 0
        if error_budget is not None and error_budget.calibration_id!=calibration.calibration_id:
            raise ValueError('Error budget calibration mismatch')
        self.error_budget=error_budget;self.max_control_error_m=max_control_error_m

    def reset(self):
        self.last = None; self.latest = None; self.confirmations = 0
        self.last_sequence = -1; self.last_time = -1.; self.grasp_epoch += 1

    def process(self, frame, now=None):
        start = perf_counter(); now = frame.capture_time if now is None else now
        if self.epoch != frame.epoch:
            self.reset(); self.epoch = frame.epoch
        status = None; metrics = {}; pose = cov = mask = None
        if frame.calibration_id != self.calibration.calibration_id or frame.depth_m.shape != (self.calibration.height, self.calibration.width):
            status = Status.CALIBRATION_INVALID
        elif (not np.isfinite(now) or not 0 <= now-frame.capture_time <= self.max_age
              or abs(frame.color_time_offset) > .025 or frame.time_uncertainty > .025
              or frame.sequence <= self.last_sequence or frame.capture_time <= self.last_time):
            status = Status.STALE
        else:
            self.last_sequence = frame.sequence; self.last_time = frame.capture_time
            scene=self.localizer.static_scene
            anchor={} if scene is None else scene.check_drift(frame)
            if anchor.get('drift_detected',False):
                status=Status.CALIBRATION_INVALID;metrics={'reason':'fixed_background_changed',**anchor}
            else:
                status, pose, cov, metrics, mask = self.localizer.locate(frame)
                if scene is not None:metrics['scene_anchor']=anchor
            if status == Status.TRACKING:
                if self.error_budget is not None:
                    cov,info=self.error_budget.evaluate(pose,mask,frame,self.calibration,metrics)
                    metrics.update(info)
                if self.last is not None:
                    dt = frame.capture_time-self.last.timestamp
                    if dt > self.prediction_ttl:
                        self.confirmations = 0; self.grasp_epoch += 1
                    jump = np.linalg.norm(pose[:3, 3]-self.last.pose[:3, 3])
                    angular_jump = symmetry_distance(pose[:3, :3], self.last.pose[:3, :3])
                    pose[:3, :3] = align_symmetry(pose[:3, :3], self.last.pose[:3, :3])
                    metrics.update(translation_jump_m=float(jump), angular_jump_rad=angular_jump)
                    if dt>0:
                        metrics['velocity_B']=np.r_[(pose[:3,3]-self.last.pose[:3,3])/dt,
                            Rotation.from_matrix(pose[:3,:3]@self.last.pose[:3,:3].T).as_rotvec()/dt].tolist()
                    # A large change is reinitialized, never silently extrapolated.
                    if jump > .02 + dt*1.5 or angular_jump > .20 + dt*8:
                        self.confirmations = 0; self.grasp_epoch += 1
                self.confirmations += 1
                status = Status.TRACKING if self.confirmations >= 2 else Status.RELOCALIZING
                self.last = PoseEstimate(status, frame.capture_time, frame.calibration_id, frame.sequence,
                                         frame.source, pose, cov, grasp_epoch=self.grasp_epoch, metrics=metrics, mask=mask)
            else:
                self.confirmations = 0
                # Prediction is a held pose with inflated covariance; never a measurement.
                if status == Status.LOST and self.last is not None and frame.capture_time-self.last.timestamp <= self.prediction_ttl:
                    status = Status.PREDICTED; pose = self.last.pose
                    cov = self.last.covariance + np.eye(6)*(frame.capture_time-self.last.timestamp)**2
        metrics = {**metrics, 'processing_ms': (perf_counter()-start)*1000}
        self.latest = PoseEstimate(status, frame.capture_time, frame.calibration_id, frame.sequence,
                                   frame.source, pose, cov, grasp_epoch=self.grasp_epoch, metrics=metrics, mask=mask)
        return self.latest

    def camera_observer(self, require_hardware=True):
        def read():
            estimate = self.latest
            if estimate is None: raise Failure('GRASP_EVIDENCE_UNAVAILABLE')
            estimate.require_measured()
            if require_hardware and (estimate.source != 'realsense' or self.calibration.provenance != 'hardware'):
                raise Failure('GRASP_EVIDENCE_UNAVAILABLE', {'reason': 'hardware_measurement_required'})
            if require_hardware and (self.error_budget is None or not self.error_budget.qualified):
                raise Failure('GRASP_EVIDENCE_UNCERTAIN',{'reason':'independent_error_budget_not_qualified'})
            if estimate.metrics.get('position_error_bound_m',0)>self.max_control_error_m:
                raise Failure('GRASP_EVIDENCE_UNCERTAIN',{'reason':'position_error_budget_exceeds_tolerance'})
            quality=float(np.clip(1-estimate.metrics.get('surface_rmse_m',.01)/.01,0,.99))
            return ObjectObservation(estimate.timestamp, estimate.pose, 'camera', confidence=quality,
                                     object_id=estimate.object_id, grasp_epoch=estimate.grasp_epoch,
                                     provenance=estimate.source,
                                     position_error_bound_m=estimate.metrics.get('position_error_bound_m'),
                                     relative_error_bound_m=estimate.metrics.get('relative_error_bound_m'),
                                     angular_error_bound_rad=estimate.metrics.get('angular_error_bound_rad'),
                                     tool_position_bound_m=estimate.metrics.get('tool_position_bound_m',0.))
        speed=0. if self.error_budget is None else self.error_budget.motion_speed_bound_m_s
        return CameraObserver(read, max_age=self.max_age,motion_speed_bound=speed,max_position_error=self.max_control_error_m)

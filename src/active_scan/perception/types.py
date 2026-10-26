from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import json
import numpy as np
from ..core import Failure


def frozen_array(value, shape=None, dtype=float):
    a = np.array(value, dtype=dtype, copy=True)
    if (shape is not None and a.shape != shape) or not np.all(np.isfinite(a)):
        raise ValueError('Invalid array shape or non-finite values')
    a.setflags(write=False)
    return a


def rigid(value):
    a = frozen_array(value, (4, 4))
    if not np.allclose(a[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError('Invalid homogeneous transform')
    if not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-6) or np.linalg.det(a[:3, :3]) < .999:
        raise ValueError('Transform must contain a proper rotation')
    return a


class Status(StrEnum):
    UNINITIALIZED = 'UNINITIALIZED'
    TRACKING = 'TRACKING'
    RELOCALIZING = 'RELOCALIZING'
    PREDICTED = 'PREDICTED'
    DEGENERATE = 'DEGENERATE'
    LOST = 'LOST'
    STALE = 'STALE'
    AMBIGUOUS = 'AMBIGUOUS'
    CALIBRATION_INVALID = 'CALIBRATION_INVALID'


@dataclass(frozen=True)
class Calibration:
    width: int
    height: int
    K: np.ndarray
    T_B_C: np.ndarray
    depth_scale: float = .001
    min_depth: float = .35
    max_depth: float = 1.2
    serial: str = 'unassigned'
    profile: str = 'rectified-depth'
    provenance: str = 'synthetic'
    table_z: float = 0.
    workspace: tuple = (.08, .48, -.25, .25, .003, .4)
    distortion_model: str = 'none'
    calibration_id: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'K', frozen_array(self.K, (3, 3)))
        object.__setattr__(self, 'T_B_C', rigid(self.T_B_C))
        object.__setattr__(self, 'workspace', tuple(float(x) for x in self.workspace))
        if self.width < 2 or self.height < 2 or self.K[0, 0] <= 0 or self.K[1, 1] <= 0:
            raise ValueError('Invalid intrinsics')
        if not np.allclose(self.K[2], [0, 0, 1]) or abs(self.K[0, 1]) > 1e-9:
            raise ValueError('Expected zero-skew pinhole intrinsics')
        if not 0 < self.depth_scale < .1 or not 0 < self.min_depth < self.max_depth:
            raise ValueError('Invalid depth units/range')
        if len(self.workspace) != 6 or any(self.workspace[i] >= self.workspace[i+1] for i in (0, 2, 4)):
            raise ValueError('Invalid workspace bounds')
        if self.distortion_model != 'none':
            raise ValueError('Localizer requires rectified depth; rectify before creating packets')
        data = self.to_dict(include_id=False)
        digest = hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()
        object.__setattr__(self, 'calibration_id', digest)

    def to_dict(self, include_id=True):
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != 'calibration_id'}
        d['K'] = self.K.tolist(); d['T_B_C'] = self.T_B_C.tolist()
        if include_id: d['calibration_id'] = self.calibration_id
        return d

    @classmethod
    def from_dict(cls, data):
        data = dict(data); expected = data.pop('calibration_id', None)
        result = cls(**data)
        if expected is not None and result.calibration_id != expected:
            raise ValueError('Calibration checksum mismatch')
        return result


@dataclass(frozen=True)
class FramePacket:
    sequence: int
    capture_time: float
    rgb: np.ndarray
    depth_m: np.ndarray
    calibration_id: str
    source: str
    epoch: int = 0
    clock_domain: str = 'monotonic'
    color_time_offset: float = 0.
    time_uncertainty: float = 0.
    robot_mask: np.ndarray | None = None
    exposure_s: float = 0.
    arrival_time: float | None = None

    def __post_init__(self):
        if self.sequence < 0 or self.epoch < 0 or not np.isfinite(self.capture_time) or self.capture_time < 0:
            raise ValueError('Invalid frame sequence/time')
        if not np.isfinite(self.color_time_offset) or not 0 <= self.time_uncertainty < 10:
            raise ValueError('Invalid synchronization metadata')
        if not np.isfinite(self.exposure_s) or not 0<=self.exposure_s<=1:
            raise ValueError('Invalid exposure duration')
        if self.arrival_time is not None and (not np.isfinite(self.arrival_time) or self.arrival_time<0):
            raise ValueError('Invalid host arrival time')
        rgb = np.array(self.rgb, copy=True)
        depth = np.array(self.depth_m, dtype=np.float32, copy=True)
        if rgb.dtype != np.uint8 or rgb.shape != (*depth.shape, 3) or depth.ndim != 2:
            raise ValueError('RGB must be uint8, aligned to the native depth grid')
        # Invalid sensor returns stay invalid. No hole filling in the evidence path.
        depth[~np.isfinite(depth) | (depth < 0)] = 0
        mask = np.zeros(depth.shape, bool) if self.robot_mask is None else np.array(self.robot_mask, dtype=bool, copy=True)
        if mask.shape != depth.shape: raise ValueError('Robot mask shape mismatch')
        for a in (rgb, depth, mask): a.setflags(write=False)
        object.__setattr__(self, 'rgb', rgb); object.__setattr__(self, 'depth_m', depth)
        object.__setattr__(self, 'robot_mask', mask)


@dataclass(frozen=True)
class PoseEstimate:
    status: Status
    timestamp: float
    calibration_id: str
    sequence: int
    source: str
    pose: np.ndarray | None = None
    covariance: np.ndarray | None = None
    object_id: str = 'cube'
    grasp_epoch: int = 0
    symmetry_ambiguous: bool = True
    metrics: dict = field(default_factory=dict)
    mask: np.ndarray | None = None

    def __post_init__(self):
        if self.pose is not None: object.__setattr__(self, 'pose', rigid(self.pose))
        if self.covariance is not None:
            c = frozen_array(self.covariance, (6, 6))
            if not np.allclose(c, c.T, atol=1e-8) or np.linalg.eigvalsh(c).min() < -1e-12:
                raise ValueError('Covariance must be positive semidefinite')
            object.__setattr__(self, 'covariance', c)
        if self.status == Status.TRACKING and (self.pose is None or self.covariance is None):
            raise ValueError('Tracking requires a pose and covariance')
        if self.mask is not None:
            a = np.array(self.mask, dtype=bool, copy=True); a.setflags(write=False)
            object.__setattr__(self, 'mask', a)

    @property
    def measured(self):
        return self.status == Status.TRACKING and self.pose is not None

    def require_measured(self):
        if not self.measured: raise Failure('GRASP_EVIDENCE_UNAVAILABLE', {'status': self.status.value})
        return self

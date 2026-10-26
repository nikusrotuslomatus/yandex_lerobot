from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import json
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[2]
FACE_NAMES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")


def transform(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def inverse(T):
    return transform(T[:3, :3].T, -T[:3, :3].T @ T[:3, 3])


def points(T, p):
    return np.asarray(p) @ T[:3, :3].T + T[:3, 3]


def quat_wxyz(R):
    return Rotation.from_matrix(R).as_quat(scalar_first=True)


def look_at(position, target):
    """World-from-MuJoCo camera: +X right, +Y up, -Z forward."""
    z = np.asarray(position, float) - target
    z /= np.linalg.norm(z)
    x = np.cross([0., 0., 1.], z)
    x /= np.linalg.norm(x)
    return transform(np.column_stack([x, np.cross(z, x), z]), position)


def optical_pose(T_mujoco):
    return T_mujoco @ np.diag([1., -1., -1., 1.])


def load_config(path=None):
    with open(path or ROOT / "configs/default.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, default=convert, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@dataclass
class SceneState:
    timestamp: float
    q_arm: np.ndarray
    dq_arm: np.ndarray
    gripper_position: float
    T_world_base: np.ndarray
    T_world_gripper: np.ndarray
    T_world_cube: np.ndarray
    T_world_camera: np.ndarray
    camera_intrinsics: np.ndarray
    grasp_id: str | None
    obstacles: dict
    cube_velocity: np.ndarray


class RobotInterface(Protocol):
    def reset(self): ...
    def state(self) -> SceneState: ...
    def set_joint_targets(self, q): ...
    def set_joint_reference(self,q,dq=None): ...
    def observe_object(self): ...
    def set_gripper_target(self, value): ...
    def step(self): ...
    def stop(self): ...


class Failure(RuntimeError):
    def __init__(self, code, details=None):
        self.code, self.details = code, details or {}
        super().__init__(code)


def cube_surface(side, n):
    centers = (np.arange(n) + .5) * side / n - side / 2
    grid = np.array(np.meshgrid(centers, centers, indexing="ij")).reshape(2, -1).T
    ps, ns, fs = [], [], []
    for face in range(6):
        axis, sign = face // 2, 1 if face % 2 == 0 else -1
        p = np.zeros((n*n, 3)); p[:, axis] = sign * side / 2
        p[:, [i for i in range(3) if i != axis]] = grid
        normal = np.zeros_like(p); normal[:, axis] = sign
        ps.extend(p); ns.extend(normal); fs.extend([face] * (n*n))
    return np.array(ps), np.array(ns), np.array(fs)


class CoverageState:
    def __init__(self, count, threshold, diversity_deg):
        self.quality = np.zeros(count)
        self.counts = np.zeros(count, int)
        self.directions = [[] for _ in range(count)]
        self.threshold = threshold
        self.cos_diversity = np.cos(np.deg2rad(diversity_deg))

    def update(self, quality, directions):
        self.quality = np.maximum(self.quality, quality)
        for i in np.flatnonzero(quality >= self.threshold):
            self.counts[i] += 1
            if all(np.dot(directions[i], d) <= self.cos_diversity for d in self.directions[i]):
                self.directions[i].append(directions[i].copy())

    def metrics(self, faces):
        return {"binary_coverage": float(np.mean(self.quality >= self.threshold)),
                "any_visibility": float(np.mean(self.quality > 0)),
                "quality_weighted": float(self.quality.mean()),
                "multiview_coverage": float(np.mean([len(x) >= 2 for x in self.directions])),
                "per_face": {name: {"binary": float(np.mean(self.quality[faces == i] >= self.threshold)),
                                    "quality": float(self.quality[faces == i].mean())}
                             for i, name in enumerate(FACE_NAMES)}}

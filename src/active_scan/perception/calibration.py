import json
import numpy as np
from ..core import inverse, transform
from .types import Calibration


def rays(calibration):
    y, x = np.indices((calibration.height, calibration.width))
    K = calibration.K
    return np.stack([(x-K[0, 2])/K[0, 0], (y-K[1, 2])/K[1, 1], np.ones_like(x)], axis=-1)


def deproject(depth, calibration, mask=None):
    if depth.shape != (calibration.height, calibration.width): raise ValueError('Depth profile mismatch')
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None: valid &= mask
    return rays(calibration)[valid] * depth[valid, None]


def calibrate_camera_to_base(points_B, pixels, K, distortion=None):
    """Known BASE-frame target points, not an unregistered board coordinate frame."""
    import cv2
    p = np.asarray(points_B, float); uv = np.asarray(pixels, float)
    if p.ndim!=2 or uv.ndim!=2 or p.shape[0] < 6 or p.shape != (len(uv), 3) or uv.shape[1:] != (2,):
        raise ValueError('At least six known base-frame points required')
    if np.linalg.svd(p-p.mean(0), compute_uv=False)[1] < 1e-4:
        raise ValueError('Collinear target')
    distortion = np.zeros(5) if distortion is None else np.asarray(distortion, float)
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(uv)):raise ValueError('Non-finite correspondences')
    ok, rv, tv, inliers = cv2.solvePnPRansac(p, uv, np.asarray(K, float), distortion,
                                           iterationsCount=200, reprojectionError=1.5,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inliers is None or len(inliers) < max(6, .8*len(p)):
        raise ValueError('Extrinsic calibration failed')
    ids=inliers.ravel()
    rv,tv=cv2.solvePnPRefineLM(p[ids],uv[ids],np.asarray(K,float),distortion,rv,tv)
    T_C_B = transform(cv2.Rodrigues(rv)[0], tv.ravel())
    if np.any((p @ T_C_B[:3, :3].T+T_C_B[:3, 3])[:, 2] <= 0):
        raise ValueError('Target behind camera')
    projected = cv2.projectPoints(p, rv, tv, K, distortion)[0].reshape(-1, 2)
    return inverse(T_C_B), {'reprojection_rms_px': float(np.sqrt(np.mean((projected[ids]-uv[ids])**2))),
                            'all_points_rms_px':float(np.sqrt(np.mean((projected-uv)**2))),
                            'inliers': len(inliers), 'absolute_accuracy_validated': False}


def load_calibration(path):
    with open(path, encoding='utf-8') as f: return Calibration.from_dict(json.load(f))

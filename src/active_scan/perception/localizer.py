"""Metric cube localization from native depth. No robot or object truth input."""
from itertools import permutations, product
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .calibration import rays
from .types import Status


def cube_symmetries():
    result = []
    for order in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            r = np.eye(3)[:, order] @ np.diag(signs)
            if np.linalg.det(r) > .5: result.append(r)
    return np.asarray(result)


SYMMETRIES = cube_symmetries()


def symmetry_distance(a, b):
    relatives = np.einsum('ij,njk->nik', a.T @ b, SYMMETRIES)
    angles = Rotation.from_matrix(relatives).magnitude()
    return float(angles.min())


def align_symmetry(rotation, reference):
    candidates = rotation @ SYMMETRIES
    angles = Rotation.from_matrix(reference.T @ candidates).magnitude()
    return candidates[int(np.argmin(angles))]


def surface_residual(points, rotation, center, half):
    q = np.abs((points - center) @ rotation) - half
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(np.max(q, axis=1), 0)


class CubeLocalizer:
    def __init__(self, calibration, size=.06, seed=42, min_points=90, static_scene=None):
        self.calibration = calibration; self.size = size; self.seed = seed
        self.min_points = min_points
        if not .02<=size<=.15:raise ValueError('Cube size must be in meters')
        self.static_scene=static_scene
        self.camera_rays = rays(calibration).astype(np.float32)
        self.world_rays = (self.camera_rays @ calibration.T_B_C[:3, :3].T).astype(np.float32)
        self.origin = calibration.T_B_C[:3, 3].astype(np.float32)

    def segment(self, frame):
        c = self.calibration; d = frame.depth_m
        points = self.world_rays * d[..., None] + self.origin
        valid = (d >= c.min_depth) & (d <= c.max_depth) & ~frame.robot_mask
        for axis in range(3):
            valid &= (points[..., axis] >= c.workspace[2*axis]) & (points[..., axis] <= c.workspace[2*axis+1])
        valid &= points[..., 2] > c.table_z + .004
        # Stereo silhouettes contain mixed/flying depths, not just Gaussian
        # outliers. Reject inconsistent samples; never replace/fill depth.
        median=cv2.medianBlur(d,3)
        valid&=(median>0)&(np.abs(d-median)<.006)
        if self.static_scene is not None:valid &= self.static_scene.foreground(frame)
        # Closing groups fragmented components only. It never invents depth samples.
        grouped = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, 8)
        candidates = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < self.min_points: continue
            mask = (labels == i) & valid
            cloud = points[mask]
            if len(cloud) < self.min_points: continue
            extent = np.quantile(cloud, .95, axis=0) - np.quantile(cloud, .05, axis=0)
            if np.max(extent) > self.size * 1.9 or np.linalg.norm(extent) < self.size * .45: continue
            candidates.append((cloud, mask))
        return candidates

    def _planes(self, cloud):
        rng = np.random.default_rng(self.seed)
        remaining = cloud
        planes = []
        threshold = .0018
        for _ in range(3):
            if len(remaining) < 45: break
            best = None; best_count = 0
            # Batched hypotheses keep Python work outside the inner point loop.
            samples = remaining[rng.integers(0, len(remaining), (100, 3))]
            normals = np.cross(samples[:, 1]-samples[:, 0], samples[:, 2]-samples[:, 0])
            norms = np.linalg.norm(normals, axis=1)
            normals = normals[norms > 1e-7] / norms[norms > 1e-7, None]
            origins = samples[norms > 1e-7, 0]
            for normal, origin in zip(normals, origins):
                if any(abs(normal @ p[0]) > .18 for p in planes): continue
                inliers = np.abs((remaining-origin) @ normal) < threshold
                count = int(inliers.sum())
                if count > best_count: best = inliers; best_count = count
            if best is None or best_count < max(45, len(cloud)*.055): break
            inlier_cloud = remaining[best]
            mean = inlier_cloud.mean(axis=0)
            _, _, vt = np.linalg.svd(inlier_cloud-mean, full_matrices=False)
            normal = vt[-1]
            if normal @ (self.calibration.T_B_C[:3, 3]-mean) < 0: normal = -normal
            if any(abs(normal @ p[0]) > .20 for p in planes): break
            planes.append((normal, float(mean @ normal), best_count, mean))
            remaining = remaining[~best]
        return planes

    def fit(self, cloud):
        if len(cloud) > 1800:
            cloud = cloud[np.linspace(0, len(cloud)-1, 1800).astype(int)]
        planes = self._planes(cloud)
        metrics = {'points': len(cloud), 'visible_planes': len(planes), 'size_prior_m': self.size}
        if len(planes) < 2: return Status.DEGENERATE, None, None, metrics
        n1, n2 = planes[0][0], planes[1][0]
        n2 = n2 - n1*(n1@n2); n2 /= np.linalg.norm(n2)
        rotation = np.column_stack([n1, n2, np.cross(n1, n2)])
        coords = cloud @ rotation
        half = self.size/2
        center_local = (np.quantile(coords, .01, axis=0)+np.quantile(coords, .99, axis=0))/2
        # Orthogonalizing normals changes their plane offsets. Reproject each
        # fitted plane's centroid; using the old normal's offset causes a large
        # center error when correlated stereo noise tilts a face normal.
        center_local[:2] = [p[3] @ rotation[:,j]-half for j,p in enumerate(planes[:2])]
        if len(planes) == 3:
            sign = np.sign(planes[2][0] @ rotation[:, 2])
            center_local[2] = planes[2][3] @ rotation[:,2]-sign*half
        center = rotation @ center_local
        # Trim gross depth outliers before the robust surface refinement.
        keep = np.abs(surface_residual(cloud, rotation, center, half)) < .006
        fit_cloud = cloud[keep]
        if len(fit_cloud) < self.min_points or keep.mean() < .72:
            metrics['inlier_fraction'] = float(keep.mean())
            return Status.DEGENERATE, None, None, metrics
        # With two visible planes the extent of the third axis supplies a weak
        # silhouette constraint. A partial patch must fail the size-support gate.
        span = np.quantile(coords, .98, axis=0)-np.quantile(coords, .02, axis=0)
        metrics['support_span_m'] = span.tolist()
        if np.min(span) < self.size*.88:
            return Status.DEGENERATE, None, None, {**metrics, 'reason': 'insufficient_size_support'}
        if len(planes) == 2 and span[2] < self.size*.80:
            return Status.DEGENERATE, None, None, {**metrics, 'reason': 'partial_silhouette'}
        def residual(x):
            r = rotation @ Rotation.from_rotvec(x[:3]).as_matrix()
            value = surface_residual(fit_cloud, r, x[3:], half)
            if len(planes) == 2:
                # Do not claim unobserved translation is fixed by plane residuals.
                value = np.r_[value, ((x[3:]-center) @ rotation[:, 2])*.5]
            return value
        fit = least_squares(residual, np.r_[np.zeros(3), center], loss='soft_l1', f_scale=.001,
                            max_nfev=18, diff_step=1e-5)
        rotation = rotation @ Rotation.from_rotvec(fit.x[:3]).as_matrix()
        center = fit.x[3:]
        error = surface_residual(fit_cloud, rotation, center, half)
        rms = float(np.sqrt(np.mean(error**2)))
        metrics.update(surface_rmse_m=rms, inlier_fraction=float(keep.mean()), evaluations=fit.nfev)
        if not fit.success or rms > .0025 or np.linalg.norm(fit.x[:3]) > .3:
            return Status.DEGENERATE, None, None, metrics
        pose = np.eye(4); pose[:3, :3] = rotation; pose[:3, 3] = center
        # Conservative heuristic uncertainty, NOT a calibrated statistical bound.
        sigma = max(.0015, rms*2); angular = max(.025, sigma/self.size)
        covariance = np.diag([sigma**2]*3+[angular**2]*3)
        if len(planes) == 2: covariance[:3, :3] += np.outer(rotation[:, 2], rotation[:, 2])*.002**2
        metrics['uncertainty_model'] = 'heuristic_floor_not_statistically_calibrated'
        return Status.TRACKING, pose, covariance, metrics

    def locate(self, frame):
        candidates = self.segment(frame)
        fitted = []
        for cloud, mask in candidates:
            status, pose, covariance, metrics = self.fit(cloud)
            fitted.append((status, pose, covariance, metrics, mask))
        valid = [f for f in fitted if f[0] == Status.TRACKING]
        if len(valid) > 1:
            return Status.AMBIGUOUS, None, None, {'candidates': len(valid)}, None
        if valid: return valid[0]
        if fitted: return fitted[0]
        return Status.LOST, None, None, {'reason': 'no_geometric_candidate'}, None

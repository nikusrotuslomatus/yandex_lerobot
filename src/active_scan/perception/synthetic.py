"""Independent analytic RGB-D test camera, never imported by the estimator.

Ray/box intersection supplies sensor images; truth is returned separately to
the evaluator. It intentionally does not reuse localization residuals/planes.
"""
import numpy as np
from ..core import optical_pose, look_at
from .types import Calibration, FramePacket


def test_calibration(width=424, height=240):
    k = np.array([[330., 0, (width-1)/2], [0, 330., (height-1)/2], [0, 0, 1]])
    return Calibration(width, height, k, optical_pose(look_at(np.array([.62, -.46, .48]), np.array([.26, 0, .07]))),
                       workspace=(.08,.46,-.22,.22,.003,.35), min_depth=.32)


def d455_fixture_calibration(width=848,height=480):
    """Approximate mode FOV for validation, NOT measured SDK intrinsics."""
    from dataclasses import replace
    from .realsense import profile_min_depth
    hfov=75. if (width,height)==(640,480) else 87.
    focal=width/(2*np.tan(np.deg2rad(hfov)/2))
    k=np.array([[focal,0,(width-1)/2],[0,focal,(height-1)/2],[0,0,1.]])
    return replace(test_calibration(width,height),K=k,min_depth=profile_min_depth(width,height),
                   profile=f'fixture_{width}x{height}_FOV_approximation_SDK_K_required')


def render(calibration, pose, sequence=0, timestamp=0., size=.06, noise=.0006,
           holes=.05, outliers=.003, occluders=(), include_cube=True, seed=0):
    c = calibration; rng = np.random.default_rng(seed)
    yy, xx = np.indices((c.height, c.width))
    ray_C = np.stack([(xx-c.K[0,2])/c.K[0,0], (yy-c.K[1,2])/c.K[1,1], np.ones_like(xx)], axis=-1)
    ray_B = ray_C @ c.T_B_C[:3,:3].T; origin = c.T_B_C[:3,3]
    with np.errstate(divide='ignore', invalid='ignore'):
        table = (c.table_z-origin[2])/ray_B[...,2]
    depth = np.where(table>0, table, np.inf)
    rgb = np.full((*depth.shape,3), 105, np.uint8)
    boxes = [(pose, np.full(3,size/2), [220,90,35])] if include_cube else []
    boxes += [(p, np.asarray(h), [45,45,48]) for p,h in occluders]
    for box_pose, half, color in boxes:
        o = (origin-box_pose[:3,3]) @ box_pose[:3,:3]
        directions = ray_B @ box_pose[:3,:3]
        with np.errstate(divide='ignore', invalid='ignore'):
            lo = (-half-o)/directions; hi = (half-o)/directions
        enter = np.minimum(lo,hi).max(axis=-1); leave = np.maximum(lo,hi).min(axis=-1)
        hit = (leave>=enter) & (enter>0) & (enter<depth)
        depth[hit]=enter[hit]; rgb[hit]=color
    valid = np.isfinite(depth) & (depth>=c.min_depth) & (depth<=c.max_depth)
    depth = np.where(valid, depth+rng.normal(0,noise,depth.shape),0)
    depth[rng.random(depth.shape)<holes]=0
    corrupt = valid & (rng.random(depth.shape)<outliers)
    depth[corrupt] += rng.uniform(-.05,.05,corrupt.sum())
    return FramePacket(sequence,timestamp,rgb,depth,c.calibration_id,'synthetic'), np.array(pose,copy=True)


def render_stereo(calibration,model,pose_at,sequence,timestamp,include_cube=True):
    """Render true optics/mount/exposure, then apply the sensor surrogate.

    pose_at is evaluator-only trajectory truth. The estimator receives just the
    returned FramePacket, carrying the nominal calibration ID.
    """
    clean,_=render(model.true_calibration(timestamp),pose_at(timestamp),sequence,max(0,timestamp),
                   noise=0,holes=0,outliers=0,include_cube=include_cube)
    depths=[];colors=[]
    for offset in (-model.profile.exposure_s/2,model.profile.exposure_s/2):
        end,_=render(model.true_calibration(timestamp+offset),pose_at(timestamp+offset),sequence,max(0,timestamp+offset),
                     noise=0,holes=0,outliers=0,include_cube=include_cube)
        depths.append(end.depth_m);colors.append(end.rgb)
    return model.apply(clean,timestamp,depths,colors)

"""Fit depth statistics using recorded raw frames and independent plane truth.

Does not qualify extrinsics, robot feedback or control. Reference-plane values
are metrology input, never estimated by fitting the same depth image.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.optimize import least_squares
from .calibration import rays
from .recording import Replay
from .sensor_noise import StereoNoiseProfile


def fit_noise_manifest(path):
    path=Path(path).resolve();data=json.loads(path.read_text(encoding='utf-8'))
    sessions=data['sessions']
    if len(sessions)<5:raise ValueError('Need >=3 training distances and >=2 held-out sessions')
    reports=[];calibration=None;hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest()}
    reference_uncertainty=float(data['reference_uncertainty_m'])
    if reference_uncertainty<=0 or not np.isfinite(reference_uncertainty) or not data.get('reference_evidence'):
        raise ValueError('Independent reference evidence and its uncertainty are required')
    for session in sessions:
        reader=Replay(path.parent/session['episode']);c=reader.calibration
        conditions=data.get('conditions',{})
        if c.provenance=='hardware' and not all(k in conditions for k in ('fps','firmware','depth_preset','emitter_enabled','exposure_policy')):
            raise ValueError('Hardware fit requires documented FPS/firmware/preset/emitter/exposure conditions')
        if calibration is None:calibration=c
        if c.calibration_id!=calibration.calibration_id:raise ValueError('All recordings must share serial/profile/calibration')
        if session['split'] not in ('train','holdout'):raise ValueError('Explicit train/holdout split required')
        x0,y0,x1,y1=session['roi']
        if not (0<=x0<x1<=c.width and 0<=y0<y1<=c.height):raise ValueError('Invalid ROI')
        normal=np.array(session['plane_C'][:3],float);distance=float(session['plane_C'][3])
        if not np.isclose(np.linalg.norm(normal),1,atol=1e-6) or distance<=0:raise ValueError('Plane n dot X = d must have unit n and d>0')
        denom=rays(c)[y0:y1,x0:x1]@normal
        if np.any(denom<=.1):raise ValueError('Invalid reference plane orientation')
        reference=distance/denom
        stack=[]
        for frame in reader:
            if c.provenance=='hardware' and frame.source not in ('realsense','realsense_bag'):
                raise ValueError('Hardware camera fit cannot use simulated sensor frames')
            if np.any(frame.robot_mask[y0:y1,x0:x1]):raise ValueError('Reference ROI contains robot')
            stack.append(np.array(frame.depth_m[y0:y1,x0:x1]))
        if len(stack)<30:raise ValueError('At least 30 frames per static-plane session required')
        stack=np.asarray(stack);valid=(stack>=c.min_depth)&(stack<=c.max_depth)
        stable=valid.mean(axis=0)>=.9
        if stable.sum()<100:raise ValueError('Insufficient stable reference pixels')
        # Require complete per-pixel tracks for correlation estimates; no hole fill.
        complete=valid.all(axis=0)
        if complete.sum()<50:raise ValueError('Insufficient complete tracks for temporal fit')
        tracks=stack[:,complete].astype(float);means=tracks.mean(axis=0);residual=tracks-means
        c0=float(np.mean(residual**2));c1=float(np.mean(residual[1:]*residual[:-1]));c2=float(np.mean(residual[2:]*residual[:-2]))
        rho=float(np.clip(c2/c1,0,.95)) if c1>0 and c2>0 else 0.
        correlated=min(c0,max(0,c1/rho)) if rho>.05 else 0.
        white=max(0,c0-correlated)
        z=float(np.median(reference[complete]));factor=c.K[0,0]*.095/z**2
        # Unidentifiable sub-quantization variance is represented as a floor.
        floor=c.depth_scale/np.sqrt(12)
        mean=np.ma.array(stack,mask=~valid).mean(axis=0).filled(np.nan)
        errors=(mean-reference)[stable]
        report={'split':session['split'],'frames':len(stack),'reference_z_m':z,
                'mean_bias_m':float(np.mean(errors)),'spatial_std_m':float(np.std(errors)),
                'temporal_std_m':max(float(np.sqrt(c0)),floor),'sub_quantization_floor_used':c0<floor**2,
                'disparity_white_px':max(float(np.sqrt(white)*factor),floor*factor),
                'disparity_correlated_px':float(np.sqrt(correlated)*factor),'temporal_correlation':rho,
                'missing_fraction':float(1-valid.mean()),'reference_uncertainty_m':reference_uncertainty}
        reports.append(report)
        index=reader.path/'index.json';hashes[str(index)]=hashlib.sha256(index.read_bytes()).hexdigest()
    train=[s for s in reports if s['split']=='train'];holdout=[s for s in reports if s['split']=='holdout']
    if len(train)<3 or len(holdout)<2 or np.ptp([s['reference_z_m'] for s in train])<.1:
        raise ValueError('Training distances need >=0.1 m span and independent holdout')
    x=np.array([s['reference_z_m'] for s in train]);y=np.array([s['mean_bias_m'] for s in train])
    fit=least_squares(lambda v:v[0]+v[1]*x-y,[0.,0.],loss='soft_l1',f_scale=.001)
    validation=[s['mean_bias_m']-fit.x[0]-fit.x[1]*s['reference_z_m'] for s in holdout]
    profile=StereoNoiseProfile(name='plane_fitted_camera_statistics',provenance='empirical_camera_statistics_other_terms_assumed',
        disparity_white_px=float(np.median([s['disparity_white_px'] for s in train])),
        disparity_correlated_px=float(np.median([s['disparity_correlated_px'] for s in train])),
        temporal_correlation=float(np.median([s['temporal_correlation'] for s in train])),
        fixed_pattern_m=float(np.median([s['spatial_std_m'] for s in train])),
        depth_offset_m=float(fit.x[0]),depth_scale_error=float(fit.x[1]),
        offset_session_std_m=0.,scale_session_std=0.,dropout_probability=float(np.median([s['missing_fraction'] for s in train])))
    return {'schema':1,'calibration_id':calibration.calibration_id,'serial':calibration.serial,
            'sensor_provenance':calibration.provenance,'noise_profile':profile.to_dict(),
            'qualified_for_control':False,'reference_evidence':data['reference_evidence'],'conditions':data.get('conditions',{}),
            'sessions':reports,'heldout_bias_residual_m':validation,'source_sha256':hashes,
            'unidentified_terms':['extrinsics','intrinsics','spatial_correlation_length','materials','exposure_motion',
                                  'transport_delay','robot_encoder_and_backlash','IMU'],'absolute_control_budget':None}

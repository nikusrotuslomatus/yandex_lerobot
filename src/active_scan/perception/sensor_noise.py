"""Physics-informed D455 *surrogate*, not a calibrated digital twin.

Stereo law and nominal 95-mm D450 baseline are sourced. Remaining parameters
are explicitly scenario assumptions until fitted to a specific serial/profile.
Truth perturbations and diagnostics stay outside FramePacket and the estimator.
"""
from dataclasses import dataclass,asdict,replace
import hashlib
import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from ..core import transform
from .types import FramePacket


@dataclass(frozen=True)
class StereoNoiseProfile:
    name: str='d455_engineering_nominal'
    provenance: str='engineering_assumption_not_device_calibration'
    baseline_m: float=.095
    disparity_white_px: float=.07
    disparity_correlated_px: float=.045
    correlation_length_px: float=2.
    temporal_correlation: float=.65
    temporal_reference_dt_s: float=1/30
    fixed_pattern_m: float=.00035
    depth_offset_m: float=0.
    offset_session_std_m: float=.0005
    depth_scale_error: float=0.
    scale_session_std: float=.0015
    thermal_drift_m_per_s: float=.000003
    thermal_time_constant_s: float=120.
    focal_relative_std: float=.0008
    principal_point_std_px: float=.20
    residual_radial_k1: float=.0005
    extrinsic_translation_std_m: float=.0006
    extrinsic_rotation_std_rad: float=.0009
    vibration_translation_m: float=.00008
    vibration_rotation_rad: float=.00015
    vibration_hz: float=18.
    lateral_std_px: float=.12
    rgb_alignment_px: float=.3
    rgb_read_noise_dn: float=1.5
    rgb_shot_noise_scale: float=.15
    dropout_probability: float=.025
    cluster_dropout_probability: float=.015
    outlier_probability: float=.003
    edge_dropout_probability: float=.25
    edge_flying_probability: float=.08
    edge_width_px: int=2
    texture_quality: float=.85
    specular_fraction: float=0.
    exposure_s: float=.004
    transport_delay_s: float=.018
    transport_jitter_s: float=.004
    clock_offset_s: float=0.
    clock_jitter_s: float=.0003
    clock_drift_ppm: float=10.
    reported_time_uncertainty_s: float=.010

    def __post_init__(self):
        signed={'depth_offset_m','depth_scale_error','thermal_drift_m_per_s','clock_offset_s','clock_drift_ppm','residual_radial_k1'}
        for key,value in asdict(self).items():
            if isinstance(value,(int,float)) and (not np.isfinite(value) or (key not in signed and value<0)):
                raise ValueError(f'Invalid noise parameter {key}')
        for key in ('dropout_probability','cluster_dropout_probability','outlier_probability','edge_dropout_probability',
                    'edge_flying_probability','texture_quality','specular_fraction'):
            if not 0<=getattr(self,key)<=1:raise ValueError(f'Invalid probability {key}')
        if not 0<=self.temporal_correlation<1:raise ValueError('Invalid temporal correlation')
        if not .01<self.baseline_m<.3 or self.temporal_reference_dt_s<=0 or abs(self.depth_scale_error)>.2:
            raise ValueError('Invalid stereo geometry/time/scale')
        if not isinstance(self.edge_width_px,int) or self.edge_width_px<0:raise ValueError('Invalid edge width')

    @property
    def profile_id(self):return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()

    def to_dict(self):return {**asdict(self),'profile_id':self.profile_id}

    @classmethod
    def from_dict(cls,data):
        data=dict(data);expected=data.pop('profile_id',None);result=cls(**data)
        if expected is not None and expected!=result.profile_id:raise ValueError('Noise profile checksum mismatch')
        return result


def scenario_profile(name):
    p=StereoNoiseProfile()
    if name=='nominal':return p
    if name=='adverse':return replace(p,name='d455_adverse_assumed',disparity_white_px=.14,
        disparity_correlated_px=.10,fixed_pattern_m=.001,offset_session_std_m=.0015,
        scale_session_std=.006,extrinsic_translation_std_m=.0015,extrinsic_rotation_std_rad=.003,
        texture_quality=.35,specular_fraction=.08,dropout_probability=.08,cluster_dropout_probability=.04,
        edge_dropout_probability=.6,edge_flying_probability=.18,exposure_s=.012)
    if name=='spec_limit':return replace(p,name='d455_2pct_scale_sensitivity_not_gaussian_spec',
        depth_scale_error=.02,depth_offset_m=.001,offset_session_std_m=0.,scale_session_std=0.,
        extrinsic_translation_std_m=.002,extrinsic_rotation_std_rad=.004)
    if name=='timing_fault':return replace(p,name='clock_and_transport_fault',clock_offset_s=-.06,
        transport_delay_s=.10,transport_jitter_s=.02,reported_time_uncertainty_s=.06)
    raise ValueError(f'Unknown scenario {name}')


def disparity_depth_sigma(z,fx,baseline,sigma_disparity):
    return np.asarray(z,dtype=float)**2/(fx*baseline)*sigma_disparity


def _field(rng,shape,length):
    value=rng.normal(size=shape).astype(np.float32)
    if length>0:value=cv2.GaussianBlur(value,(0,0),max(.1,length))
    std=float(value.std())
    return (value-value.mean())/max(std,1e-12)


def stereo_visibility(depth,fx,baseline,tolerance=.002):
    """One-way rectified right-view z-buffer; not a photometric stereo matcher."""
    if depth.ndim!=2 or not np.isfinite(fx+baseline) or fx<=0 or baseline<=0:raise ValueError('Invalid stereo visibility geometry')
    h,w=depth.shape;y,x=np.indices(depth.shape);valid=np.isfinite(depth)&(depth>0)
    right=np.rint(x-fx*baseline/np.where(valid,depth,1.)).astype(np.int32)
    valid&=(right>=0)&(right<w)
    index=y[valid]*w+right[valid];buffer=np.full(h*w,np.inf,np.float32)
    np.minimum.at(buffer,index,depth[valid])
    visible=np.zeros(depth.shape,bool)
    visible[valid]=depth[valid]<=buffer[index]+tolerance
    return visible


class StereoNoiseModel:
    def __init__(self,calibration,profile=None,seed=0):
        self.calibration=calibration;self.profile=profile or StereoNoiseProfile();self.rng=np.random.default_rng(seed)
        p=self.profile;r=self.rng;shape=(calibration.height,calibration.width)
        self.fixed=_field(r,shape,12)*p.fixed_pattern_m;self.start_time=None
        self.temporal=_field(r,shape,p.correlation_length_px);self.previous_time=None
        self.offset=p.depth_offset_m+r.normal(0,p.offset_session_std_m)
        self.scale=p.depth_scale_error+r.normal(0,p.scale_session_std)
        K=calibration.K.copy();K[0,0]*=1+r.normal(0,p.focal_relative_std);K[1,1]*=1+r.normal(0,p.focal_relative_std)
        K[:2,2]+=r.normal(0,p.principal_point_std_px,2)
        self.mount=transform(Rotation.from_rotvec(r.normal(0,p.extrinsic_rotation_std_rad,3)).as_matrix(),
                             r.normal(0,p.extrinsic_translation_std_m,3))
        self.true_base=replace(calibration,K=K,T_B_C=calibration.T_B_C@self.mount)
        self.phases=r.uniform(0,2*np.pi,6)
        y,x=np.indices(shape,dtype=np.float32);self.x=x;self.y=y
        self.flying_fixed=_field(r,shape,8)

    def true_calibration(self,timestamp):
        p=self.profile
        vibration=np.sin(2*np.pi*p.vibration_hz*timestamp+self.phases)
        delta=transform(Rotation.from_rotvec(vibration[3:]*p.vibration_rotation_rad).as_matrix(),
                        vibration[:3]*p.vibration_translation_m)
        return replace(self.true_base,T_B_C=self.true_base.T_B_C@delta)

    def apply(self,clean,timestamp,exposure_depths=(),exposure_rgbs=(),source='synthetic_stereo'):
        c=self.calibration;p=self.profile;r=self.rng;k=self.true_base.K
        if clean.depth_m.shape!=(c.height,c.width):raise ValueError('Noise profile image shape mismatch')
        if not np.isfinite(timestamp) or (self.previous_time is not None and timestamp<=self.previous_time):
            raise ValueError('Noise timestamps must increase; instantiate a new episode on reset')
        dt=p.temporal_reference_dt_s if self.previous_time is None else timestamp-self.previous_time
        if self.start_time is None:self.start_time=timestamp
        elapsed=timestamp-self.start_time
        self.previous_time=timestamp
        rho=p.temporal_correlation**(dt/p.temporal_reference_dt_s)
        self.temporal=rho*self.temporal+np.sqrt(1-rho*rho)*_field(r,clean.depth_m.shape,p.correlation_length_px)
        z=np.array(clean.depth_m,copy=True);valid=(z>=c.min_depth)&(z<=c.max_depth)
        # Residual lens/lateral error warps depth by nearest-neighbor, RGB linearly.
        nx=(self.x-k[0,2])/k[0,0];ny=(self.y-k[1,2])/k[1,1];rad=nx*nx+ny*ny
        dx=p.residual_radial_k1*rad*(self.x-k[0,2])+p.lateral_std_px*_field(r,z.shape,2)
        dy=p.residual_radial_k1*rad*(self.y-k[1,2])+p.lateral_std_px*_field(r,z.shape,2)
        z=cv2.remap(z,(self.x+dx).astype(np.float32),(self.y+dy).astype(np.float32),cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
        valid&=stereo_visibility(z,k[0,0],p.baseline_m)
        # Incidence-dependent stereo uncertainty from measured surface normals.
        xyz=np.stack([nx*z,ny*z,z],axis=-1)
        gy,gx=np.gradient(xyz,axis=(0,1));normal=np.cross(gx,gy)
        norm=np.linalg.norm(normal,axis=-1);direction=np.stack([nx,ny,np.ones_like(nx)],axis=-1)
        direction/=np.linalg.norm(direction,axis=-1,keepdims=True)
        cosine=np.abs(np.sum(normal*direction,axis=-1))/np.maximum(norm,1e-10)
        angle_factor=1/np.maximum(cosine,.25)
        texture_factor=1+2*(1-p.texture_quality)
        disparity=k[0,0]*p.baseline_m/np.maximum(z,1e-6)
        delta=(p.disparity_white_px*r.normal(size=z.shape)+p.disparity_correlated_px*self.temporal)*angle_factor*texture_factor
        noisy=k[0,0]*p.baseline_m/np.maximum(disparity+delta,1e-4)
        thermal=elapsed if p.thermal_time_constant_s==0 else p.thermal_time_constant_s*(-np.expm1(-elapsed/p.thermal_time_constant_s))
        noisy=noisy*(1+self.scale)+self.offset+self.fixed+p.thermal_drift_m_per_s*thermal
        # Depth discontinuities are handled separately from smooth-surface noise.
        high=cv2.dilate(z,np.ones((3,3),np.uint8));low=cv2.erode(z,np.ones((3,3),np.uint8))
        edge=(high-low)>.008
        if p.edge_width_px:edge=cv2.dilate(edge.astype(np.uint8),np.ones((2*p.edge_width_px+1,)*2,np.uint8)).astype(bool)
        moving_edge=np.zeros(z.shape,bool)
        for end in exposure_depths:
            if end.shape!=z.shape:raise ValueError('Exposure depth shape mismatch')
            moving_edge|=np.abs(end-z)>.006
        clustered=_field(r,z.shape,8)
        # Correlated patch loss and a fixed low-return/specular pattern.
        cut=np.quantile(clustered,1-p.cluster_dropout_probability) if p.cluster_dropout_probability else np.inf
        spec_cut=np.quantile(self.flying_fixed,1-p.specular_fraction) if p.specular_fraction else np.inf
        lost=(r.random(z.shape)<p.dropout_probability*(1+2*(1-p.texture_quality)))|(clustered>cut)|(self.flying_fixed>spec_cut)
        lost|=(edge&(r.random(z.shape)<p.edge_dropout_probability))|(moving_edge&(r.random(z.shape)<.55))
        flying=(edge|moving_edge)&(r.random(z.shape)<p.edge_flying_probability)
        mix=r.random(z.shape);noisy[flying]=(mix*high+(1-mix)*low)[flying]
        outliers=r.random(z.shape)<p.outlier_probability
        noisy[outliers]+=r.uniform(-.05,.05,outliers.sum())
        noisy=np.rint(noisy/c.depth_scale)*c.depth_scale
        valid&=~lost&(noisy>=c.min_depth)&(noisy<=c.max_depth)&(z>0)
        noisy[~valid]=0
        rgb=np.asarray(clean.rgb,dtype=float)
        if exposure_rgbs:rgb=np.mean([rgb,*exposure_rgbs],axis=0)
        rgb=cv2.remap(rgb.astype(np.float32),self.x+p.rgb_alignment_px,self.y,cv2.INTER_LINEAR)
        rgb+=r.normal(size=rgb.shape)*np.sqrt(p.rgb_read_noise_dn**2+p.rgb_shot_noise_scale*np.maximum(rgb,0))
        rgb=np.clip(np.rint(rgb),0,255).astype(np.uint8)
        reported=timestamp+elapsed*p.clock_drift_ppm*1e-6+p.clock_offset_s+r.normal(0,p.clock_jitter_s)
        arrival=timestamp+max(0,p.transport_delay_s+r.normal(0,p.transport_jitter_s))
        packet=FramePacket(clean.sequence,max(0,reported),rgb,noisy,c.calibration_id,source,
                           color_time_offset=0.,time_uncertainty=p.reported_time_uncertainty_s,
                           robot_mask=clean.robot_mask,exposure_s=p.exposure_s,arrival_time=arrival)
        diagnostics={'valid_fraction':float(valid.mean()),'edge_fraction':float(edge.mean()),
                     'moving_edge_fraction':float(moving_edge.mean()),'actual_scale_error':self.scale,
                     'actual_offset_m':self.offset,'actual_capture_time':timestamp,'arrival_time':arrival,
                     'noise_profile_id':p.profile_id}
        return packet,diagnostics

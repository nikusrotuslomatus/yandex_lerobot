"""Explicit error envelope; systematic terms never shrink with pixel count.

This is a conditional engineering bound, not a calibrated coverage probability.
Defaults are deliberately unqualified for physical control. A quality fit of a
known 60-mm cube cannot qualify camera-to-base, encoders or absolute depth.
"""
from dataclasses import dataclass,asdict
import json
import numpy as np
from .sensor_noise import disparity_depth_sigma


@dataclass(frozen=True)
class ErrorBudget:
    calibration_id: str
    qualified: bool=False
    evidence_sha256: str=''
    provenance: str='engineering_bounds_require_bench_validation'
    depth_scale_bound: float=.02
    depth_bias_bound_m: float=.001
    extrinsic_translation_bound_m: float=.002
    extrinsic_rotation_bound_rad: float=.003
    intrinsic_relative_bound: float=.003
    pixel_calibration_bound_px: float=.5
    disparity_std_px: float=.10
    baseline_m: float=.095
    correlation_length_px: float=3.
    max_effective_samples: int=20
    motion_speed_bound_m_s: float=.5
    motion_acceleration_bound_m_s2: float=15.
    unknown_exposure_bound_s: float=.033
    relative_motion_bound_m: float=.03
    tool_position_bound_m: float=.002

    def __post_init__(self):
        for key,value in asdict(self).items():
            if isinstance(value,(int,float)) and (not np.isfinite(value) or value<0):raise ValueError(f'Invalid error budget {key}')
        if self.baseline_m<=0 or self.max_effective_samples<1:raise ValueError('Invalid budget geometry')
        if self.qualified and (len(self.evidence_sha256)!=64 or any(x not in '0123456789abcdef' for x in self.evidence_sha256)):
            raise ValueError('Qualified budget requires SHA256 of independent metrology evidence')

    def to_dict(self):return asdict(self)

    @classmethod
    def load(cls,path,calibration):
        with open(path,encoding='utf-8') as f:result=cls(**json.load(f))
        if result.calibration_id!=calibration.calibration_id:raise ValueError('Error budget calibration mismatch')
        return result

    def evaluate(self,pose,mask,frame,calibration,metrics):
        if calibration.calibration_id!=self.calibration_id:raise ValueError('Error budget calibration mismatch')
        depths=frame.depth_m[mask] if mask is not None else np.array([])
        z=float(np.median(depths[depths>0])) if np.any(depths>0) else calibration.max_depth
        n=max(1,len(depths));effective=max(1,min(self.max_effective_samples,n/max(1,2*np.pi*self.correlation_length_px**2)))
        # No IID / sqrt(N-pixels) claim for spatially correlated depth.
        pixel_sigma=float(disparity_depth_sigma(z,calibration.K[0,0],self.baseline_m,self.disparity_std_px))
        random_sigma=max(pixel_sigma,metrics.get('surface_rmse_m',pixel_sigma))/np.sqrt(effective)
        if metrics.get('visible_planes',0)<3:random_sigma=max(random_sigma,.0008)
        distance=float(np.linalg.norm(pose[:3,3]-calibration.T_B_C[:3,3]))
        quantization=calibration.depth_scale/2
        systematic=(quantization+self.depth_bias_bound_m+self.depth_scale_bound*z+self.extrinsic_translation_bound_m+
                    self.extrinsic_rotation_bound_rad*distance+self.intrinsic_relative_bound*distance+
                    self.pixel_calibration_bound_px*distance/min(calibration.K[0,0],calibration.K[1,1]))
        exposure=frame.exposure_s or self.unknown_exposure_bound_s
        time=frame.time_uncertainty+exposure/2
        timing=self.motion_speed_bound_m_s*time+.5*self.motion_acceleration_bound_m_s2*time*time
        absolute=3*random_sigma+systematic+timing
        relative=quantization+3*random_sigma+timing+(self.depth_scale_bound+self.extrinsic_rotation_bound_rad+
                                      self.intrinsic_relative_bound)*self.relative_motion_bound_m
        angular=3*random_sigma/metrics.get('size_prior_m',.06)+self.extrinsic_rotation_bound_rad+self.pixel_calibration_bound_px/min(calibration.K[0,0],calibration.K[1,1])
        covariance=np.diag([(random_sigma**2+systematic**2/3)]*3+[(angular/3)**2]*3)
        info={'position_error_bound_m':float(absolute),'relative_error_bound_m':float(relative),
              'angular_error_bound_rad':float(angular),'systematic_position_bound_m':float(systematic),
              'timing_error_bound_m':float(timing),'random_position_sigma_m':float(random_sigma),
              'quantization_bound_m':float(quantization),
              'effective_samples':float(effective),'error_budget_qualified':self.qualified,
              'error_budget_provenance':self.provenance,'tool_position_bound_m':self.tool_position_bound_m,
              'uncertainty_model':'conditional_engineering_envelope_not_coverage_guarantee'}
        return covariance,info

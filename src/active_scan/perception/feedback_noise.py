"""Encoder and IMU uncertainty surrogates with explicit units/provenance.

IMU is diagnostic-only: the fixed-camera localizer does not consume inertial
odometry. Encoder parameters must be measured on the assembled transmission.
"""
from dataclasses import dataclass,asdict
import numpy as np


@dataclass(frozen=True)
class JointNoiseProfile:
    encoder_step_rad: float=2*np.pi/4096
    white_std_rad: float=.0008
    zero_std_rad: float=.002
    backlash_rad: float=.005
    drift_rad_s: float=.00001
    timestamp_offset_s: float=.002
    timestamp_jitter_s: float=.0005
    transport_delay_s: float=.006
    provenance: str='engineering_scenario_not_measured_SO101'

    def __post_init__(self):
        for k,v in asdict(self).items():
            if isinstance(v,(int,float)) and (not np.isfinite(v) or (k not in ('timestamp_offset_s','drift_rad_s') and v<0)):
                raise ValueError('Invalid encoder model')

    def angle_envelope(self):return 3*self.zero_std_rad+3*self.white_std_rad+self.backlash_rad/2+self.encoder_step_rad/2


class JointNoiseModel:
    def __init__(self,profile=None,joints=6,seed=0):
        self.profile=profile or JointNoiseProfile();self.rng=np.random.default_rng(seed);self.joints=joints
        self.zero=self.rng.normal(0,self.profile.zero_std_rad,joints)
        self.previous=None;self.direction=np.zeros(joints);self.start=None;self.last_time=None

    def measure(self,q_link,timestamp):
        p=self.profile;q=np.asarray(q_link,float)
        if q.shape!=(self.joints,) or not np.all(np.isfinite(q)) or not np.isfinite(timestamp):raise ValueError('Invalid true joints')
        if self.last_time is not None and timestamp<=self.last_time:raise ValueError('Encoder time must increase')
        if self.start is None:self.start=timestamp
        if self.previous is not None:
            delta=q-self.previous;change=np.abs(delta)>1e-7;self.direction[change]=np.sign(delta[change])
        measured=q+self.zero+self.direction*p.backlash_rad/2+p.drift_rad_s*(timestamp-self.start)+self.rng.normal(0,p.white_std_rad,self.joints)
        if p.encoder_step_rad:measured=np.rint(measured/p.encoder_step_rad)*p.encoder_step_rad
        self.previous=q.copy();self.last_time=timestamp
        return measured,timestamp+p.timestamp_offset_s+self.rng.normal(0,p.timestamp_jitter_s)


def linearized_fk_position_bound(jacobian,angle_bound_rad):
    jacobian=np.asarray(jacobian,float)
    if jacobian.ndim!=2 or jacobian.shape[0]!=3 or not np.all(np.isfinite(jacobian)) or angle_bound_rad<0:
        raise ValueError('Expected translational FK Jacobian (3,N)')
    return float(np.linalg.norm(np.sum(np.abs(jacobian)*angle_bound_rad,axis=1)))


@dataclass(frozen=True)
class ImuNoiseProfile:
    gyro_density_rad_s_sqrt_hz: float=.0003
    accel_density_m_s2_sqrt_hz: float=.002
    gyro_bias_rw_rad_s2_sqrt_hz: float=.00002
    accel_bias_rw_m_s3_sqrt_hz: float=.0002
    gyro_initial_bias_std_rad_s: float=.001
    accel_initial_bias_std_m_s2: float=.01
    provenance: str='assumed_diagnostic_only_fit_with_Allan_variance'


class ImuNoiseModel:
    def __init__(self,profile=None,seed=0):
        self.profile=profile or ImuNoiseProfile();self.rng=np.random.default_rng(seed);p=self.profile
        for k,v in asdict(p).items():
            if isinstance(v,(int,float)) and (not np.isfinite(v) or v<0):raise ValueError('Invalid IMU noise')
        self.gyro_bias=self.rng.normal(0,p.gyro_initial_bias_std_rad_s,3)
        self.accel_bias=self.rng.normal(0,p.accel_initial_bias_std_m_s2,3)

    def measure(self,angular_velocity,specific_force,dt):
        if not np.isfinite(dt) or dt<=0:raise ValueError('IMU dt must be positive')
        w=np.asarray(angular_velocity,float);a=np.asarray(specific_force,float)
        if w.shape!=(3,) or a.shape!=(3,) or not np.all(np.isfinite(np.r_[w,a])):raise ValueError('Invalid IMU truth')
        p=self.profile;r=self.rng
        self.gyro_bias+=p.gyro_bias_rw_rad_s2_sqrt_hz*np.sqrt(dt)*r.normal(size=3)
        self.accel_bias+=p.accel_bias_rw_m_s3_sqrt_hz*np.sqrt(dt)*r.normal(size=3)
        return (w+self.gyro_bias+p.gyro_density_rad_s_sqrt_hz/np.sqrt(dt)*r.normal(size=3),
                a+self.accel_bias+p.accel_density_m_s2_sqrt_hz/np.sqrt(dt)*r.normal(size=3))

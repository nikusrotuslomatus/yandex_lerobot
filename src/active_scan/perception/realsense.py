"""D455 native-depth acquisition. USB I/O stays outside the control thread."""
from time import monotonic
import numpy as np
from .sync import ClockMapper
from .types import FramePacket


def probe():
    import pyrealsense2 as rs
    return [{'name': d.get_info(rs.camera_info.name), 'serial': d.get_info(rs.camera_info.serial_number),
             'firmware': d.get_info(rs.camera_info.firmware_version)} for d in rs.context().query_devices()]


def profile_min_depth(width, height):
    # D400 datasheet March 2026, D455 Min-Z at supported rectified profiles.
    values = {(1280,720):.520, (848,480):.350, (640,480):.320, (640,360):.260,(480,270):.200,(424,240):.180}
    if (width,height) not in values: raise ValueError('Unqualified D455 profile; specify a validated profile')
    return values[width,height]


class D455Source:
    def __init__(self, calibration, fps=30, bag_path=None, record_bag=None):
        import pyrealsense2 as rs
        self.rs=rs; self.calibration=calibration; self.pipeline=rs.pipeline(); config=rs.config()
        c=calibration
        if bag_path:
            config.enable_device_from_file(str(bag_path), repeat_playback=False)
        else:
            if c.provenance != 'hardware' or c.serial == 'unassigned':
                raise ValueError('Live localization requires measured base-camera calibration and serial')
            config.enable_device(c.serial)
            config.enable_stream(rs.stream.depth,c.width,c.height,rs.format.z16,fps)
            config.enable_stream(rs.stream.color,640,480,rs.format.rgb8,30)
        if record_bag: config.enable_record_to_file(str(record_bag))
        self.started=False; self.is_playback=bool(bag_path)
        try:
            profile=self.pipeline.start(config); self.started=True
            device=profile.get_device()
            if 'D455' not in device.get_info(rs.camera_info.name):raise ValueError('This acquisition profile is qualified for D455 only')
            if bag_path: device.as_playback().set_real_time(False)
            dp=profile.get_stream(rs.stream.depth).as_video_stream_profile()
            intr=dp.get_intrinsics()
            k=np.array([[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1.]])
            serial=device.get_info(rs.camera_info.serial_number)
            scale=device.first_depth_sensor().get_depth_scale()
            if (intr.width,intr.height)!=(c.width,c.height) or not np.allclose(k,c.K,atol=.01,rtol=0):
                raise ValueError('Actual depth intrinsics/profile differ from calibration')
            if serial!=c.serial or not np.isclose(scale,c.depth_scale,rtol=1e-5):
                raise ValueError('Device serial or depth scale differs from calibration')
            if np.max(np.abs(intr.coeffs)) > 1e-8:
                raise ValueError('Nonzero depth distortion requires explicit rectification calibration')
            if c.min_depth < profile_min_depth(c.width,c.height):
                raise ValueError('Configured range violates D455 profile Min-Z')
            self.align=rs.align(rs.stream.depth); self.clock=ClockMapper(); self.sequence=0
        except Exception:
            self.close(); raise

    def read(self, timeout_ms=2000):
        raw=self.pipeline.wait_for_frames(timeout_ms)
        arrival=monotonic()
        frames=self.align.process(raw)
        depth=frames.get_depth_frame(); color=frames.get_color_frame()
        if not depth or not color: raise RuntimeError('Incomplete synchronized RGB-D frameset')
        if depth.get_frame_timestamp_domain()!=color.get_frame_timestamp_domain():
            raise RuntimeError('RGB and depth clock domains differ')
        timestamp,uncertainty,epoch=self.clock.map(depth.get_timestamp(),arrival)
        exposure=0.
        if hasattr(depth,'supports_frame_metadata') and depth.supports_frame_metadata(self.rs.frame_metadata_value.actual_exposure):
            exposure=float(depth.get_frame_metadata(self.rs.frame_metadata_value.actual_exposure))*1e-6
        packet=FramePacket(self.sequence,timestamp,np.asanyarray(color.get_data()),
                           np.asanyarray(depth.get_data()).astype(np.float32)*self.calibration.depth_scale,
                           self.calibration.calibration_id,'realsense_bag' if self.is_playback else 'realsense',epoch,
                           str(depth.get_frame_timestamp_domain()),
                           (color.get_timestamp()-depth.get_timestamp())/1000,uncertainty,
                           exposure_s=exposure,arrival_time=arrival)
        self.sequence+=1
        return packet

    def close(self):
        if self.started: self.pipeline.stop(); self.started=False

    def __enter__(self): return self
    def __exit__(self,*args): self.close()

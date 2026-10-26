"""Acquisition protocol tests; SDK mocks are not physical-device acceptance."""
from types import SimpleNamespace
import numpy as np
import pytest
from active_scan.perception.realsense import D455Source
from active_scan.perception.sync import ClockMapper
from active_scan.perception.synthetic import test_calibration as make_calibration


class SDKFrame:
    def __init__(self,data,timestamp=1000.,domain='hardware_clock'):
        self.data=data;self.timestamp=timestamp;self.domain=domain
    def get_data(self):return self.data
    def get_timestamp(self):return self.timestamp
    def get_frame_timestamp_domain(self):return self.domain


def source_fixture():
    source=D455Source.__new__(D455Source);source.calibration=make_calibration()
    c=source.calibration
    depth=SDKFrame(np.full((c.height,c.width),600,np.uint16))
    color=SDKFrame(np.full((c.height,c.width,3),123,np.uint8))
    frames=SimpleNamespace(get_depth_frame=lambda:depth,get_color_frame=lambda:color)
    source.pipeline=SimpleNamespace(wait_for_frames=lambda timeout:frames)
    source.align=SimpleNamespace(process=lambda raw:raw)
    source.clock=ClockMapper();source.sequence=0;source.is_playback=False
    return source,depth,color


def test_sdk_native_depth_scale_alignment_and_frame_ownership():
    source,depth,color=source_fixture();depth.data[0,0]=0
    frame=source.read()
    assert frame.source=='realsense' and frame.sequence==0
    assert frame.depth_m.shape==depth.data.shape and frame.rgb.shape[:2]==depth.data.shape
    assert frame.depth_m[10,10]==pytest.approx(.6) and frame.depth_m[0,0]==0
    depth.data[:]=0;color.data[:]=0
    assert frame.depth_m[10,10]>0 and frame.rgb[10,10,0]==123


def test_sdk_clock_domains_must_agree():
    source,depth,color=source_fixture();color.domain='system_time'
    with pytest.raises(RuntimeError,match='clock domains'):source.read()


def test_sdk_missing_color_is_not_silently_reused():
    source,depth,color=source_fixture()
    source.pipeline=SimpleNamespace(wait_for_frames=lambda timeout:SimpleNamespace(get_depth_frame=lambda:depth,get_color_frame=lambda:None))
    with pytest.raises(RuntimeError,match='Incomplete'):source.read()


def test_sdk_restart_changes_epoch_and_bag_is_not_live():
    source,depth,color=source_fixture();first=source.read()
    depth.timestamp=10.;color.timestamp=10.;source.is_playback=True
    second=source.read()
    assert second.epoch==first.epoch+1 and second.source=='realsense_bag'

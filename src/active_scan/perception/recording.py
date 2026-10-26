"""Portable, checksum-verified sensor-only recordings (no pickle, no truth)."""
from pathlib import Path
import hashlib
import json
import numpy as np
from .types import Calibration, FramePacket


class Recorder:
    def __init__(self, path, calibration):
        self.path = Path(path); self.path.mkdir(parents=True, exist_ok=False)
        self.calibration = calibration; self.frames = []
        self._flush()

    def _flush(self):
        data = {'schema': 1, 'calibration': self.calibration.to_dict(), 'frames': self.frames}
        temporary = self.path / 'index.tmp'
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(self.path/'index.json')

    def append(self, frame):
        if frame.calibration_id != self.calibration.calibration_id: raise ValueError('Calibration mismatch')
        if self.frames and (frame.epoch, frame.sequence) <= (self.frames[-1]['epoch'], self.frames[-1]['sequence']):
            raise ValueError('Recording order must increase')
        filename = f'frame_{len(self.frames):06d}.npz'
        path = self.path/filename
        np.savez_compressed(path, rgb=frame.rgb, depth_m=frame.depth_m, robot_mask=frame.robot_mask)
        meta = {k:getattr(frame,k) for k in ('sequence','capture_time','calibration_id','source','epoch','clock_domain','color_time_offset','time_uncertainty','exposure_s','arrival_time')}
        self.frames.append({**meta, 'file': filename, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        self._flush()


class Replay:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.index = json.loads((self.path/'index.json').read_text(encoding='utf-8'))
        if self.index['schema'] != 1: raise ValueError('Unsupported recording schema')
        self.calibration = Calibration.from_dict(self.index['calibration'])

    def __iter__(self):
        for item in self.index['frames']:
            meta = dict(item); path = (self.path/meta.pop('file')).resolve()
            if path.parent != self.path: raise ValueError('Recording path escapes episode')
            if hashlib.sha256(path.read_bytes()).hexdigest() != meta.pop('sha256'):
                raise ValueError('Frame checksum mismatch')
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != {'rgb','depth_m','robot_mask'}: raise ValueError('Unexpected sensor fields')
                yield FramePacket(**meta, rgb=data['rgb'], depth_m=data['depth_m'], robot_mask=data['robot_mask'])

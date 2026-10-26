from collections import deque
import numpy as np
from ..core import Failure


class ClockMapper:
    """Device milliseconds -> host monotonic time, with reset detection.

    Arrival-only synchronization cannot determine absolute transport delay.
    Its reported uncertainty must remain in the measurement error budget.
    """
    def __init__(self, capacity=64, transport_uncertainty=.01):
        self.samples = deque(maxlen=capacity); self.epoch = 0
        self.transport_uncertainty = transport_uncertainty

    def map(self, device_ms, host_arrival):
        d = float(device_ms)*.001; h = float(host_arrival)
        if not np.isfinite(d+h) or d < 0: raise ValueError('Invalid clock sample')
        if self.samples and (d <= self.samples[-1][0] or h <= self.samples[-1][1]):
            self.samples.clear(); self.epoch += 1
        self.samples.append((d, h)); a = np.asarray(self.samples)
        slope = 1.
        if len(a) >= 8 and np.ptp(a[:, 0]) > .1:
            slope = float(np.clip(np.polyfit(a[:, 0]-a[0, 0], a[:, 1]-a[0, 1], 1)[0], .995, 1.005))
        offsets = a[:, 1]-slope*a[:, 0]
        offset = float(np.quantile(offsets, .1))
        uncertainty = self.transport_uncertainty+float(np.ptp(offsets))
        return min(h, slope*d+offset), uncertainty, self.epoch


class FeedbackBuffer:
    def __init__(self, capacity=512, max_gap=.05, joints=5):
        self.values = deque(maxlen=capacity); self.max_gap = max_gap;self.joints=joints

    def add(self, timestamp, q):
        q = np.asarray(q, float)
        if q.shape != (self.joints,) or not np.all(np.isfinite(q)) or not np.isfinite(timestamp): raise ValueError('Invalid feedback')
        if self.values and timestamp <= self.values[-1][0]: raise ValueError('Non-monotonic feedback')
        self.values.append((float(timestamp), q.copy()))

    def at(self, timestamp):
        if not self.values: raise Failure('ROBOT_FEEDBACK_UNAVAILABLE')
        times = np.array([t for t, _ in self.values]); i = np.searchsorted(times, timestamp)
        if i < len(times) and abs(times[i]-timestamp) < 1e-9: return self.values[i][1].copy()
        if i == 0 or i == len(times) or times[i]-times[i-1] > self.max_gap:
            raise Failure('ROBOT_FEEDBACK_STALE')
        u = (timestamp-times[i-1])/(times[i]-times[i-1])
        # The SO-101 revolute joints have finite limits, not continuous wraparound.
        return (1-u)*self.values[i-1][1]+u*self.values[i][1]


class LatestFrameBuffer:
    """Bounded newest-only channel. Publish immutable FramePackets."""
    def __init__(self):
        import threading
        self._lock = threading.Lock(); self._frame = None; self.dropped = 0

    def publish(self, frame):
        with self._lock:
            if self._frame is not None: self.dropped += 1
            self._frame = frame

    def take(self):
        with self._lock:
            frame = self._frame; self._frame = None
            return frame

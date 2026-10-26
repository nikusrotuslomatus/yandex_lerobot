"""Bounded USB capture worker; localization and mapping never block robot I/O."""
from dataclasses import replace
import threading
from time import monotonic
from ..core import Failure
from ..grasp import CameraObserver
from .pipeline import PerceptionPipeline
from .sync import LatestFrameBuffer


class CaptureWorker:
    def __init__(self, source, mask_provider=None):
        self.source=source;self.mask_provider=mask_provider;self.buffer=LatestFrameBuffer()
        self.stop_event=threading.Event();self.ready=threading.Event();self.error=None
        self.thread=threading.Thread(target=self._run,name='rgbd-capture',daemon=True)

    def _run(self):
        try:
            while not self.stop_event.is_set():
                frame=self.source.read()
                if self.mask_provider is not None:
                    frame=replace(frame,robot_mask=self.mask_provider(frame.capture_time))
                self.buffer.publish(frame);self.ready.set()
        except Exception as exc:
            self.error=exc;self.ready.set()

    def start(self): self.thread.start();return self

    def take(self,timeout=.25):
        if not self.ready.wait(timeout): return None
        self.ready.clear()
        if self.error is not None: raise RuntimeError('RGB-D capture worker failed') from self.error
        return self.buffer.take()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=3.)
        if self.thread.is_alive(): raise RuntimeError('RGB-D capture worker did not stop within 3 seconds')


class LocalizationRuntime:
    """Capture -> latest-only buffer -> pose worker -> nonblocking observer.

    Source exposes .calibration/.read(). Robot control polls .camera_observer();
    no camera waits, TSDF, disk or network operations run in that callback.
    Mapping/recording consumers should use a separate bounded worker.
    """
    def __init__(self,source,mask_provider=None,static_scene=None,now=monotonic,error_budget=None):
        self.source=source;self.capture=CaptureWorker(source,mask_provider)
        if error_budget is None and source.calibration.provenance=='hardware':
            from .uncertainty import ErrorBudget
            error_budget=ErrorBudget(source.calibration.calibration_id)
        self.pipeline=PerceptionPipeline(source.calibration,static_scene=static_scene,error_budget=error_budget)
        self.now=now;self.mask_provider=mask_provider
        self.stop_event=threading.Event();self.error=None;self.processed=0
        self.thread=threading.Thread(target=self._run,name='cube-localization',daemon=True)

    def _run(self):
        try:
            while not self.stop_event.is_set():
                frame=self.capture.take()
                if frame is not None:
                    self.pipeline.process(frame,self.now());self.processed+=1
        except Exception as exc:
            self.error=exc

    def start(self):
        self.capture.start();self.thread.start();return self

    def camera_observer(self,require_hardware=True):
        observer=self.pipeline.camera_observer(require_hardware)
        def read():
            if self.stop_event.is_set() or self.error is not None or self.capture.error is not None:
                raise Failure('GRASP_EVIDENCE_UNAVAILABLE',{'reason':'perception_worker_unhealthy'})
            if require_hardware and self.mask_provider is None:
                raise Failure('GRASP_EVIDENCE_UNAVAILABLE',{'reason':'synchronized_robot_mask_required'})
            # Observer.observe performs the final age check in the robot's clock.
            return observer.read_pose()
        return CameraObserver(read,max_age=self.pipeline.max_age,motion_speed_bound=observer.motion_speed_bound,
                              max_position_error=observer.max_position_error)

    def close(self):
        self.stop_event.set();self.thread.join(timeout=3.)
        try:self.capture.close()
        finally:
            if hasattr(self.source,'close'):self.source.close()
        if self.thread.is_alive():raise RuntimeError('Pose worker did not stop within 3 seconds')

    def __enter__(self):return self.start()
    def __exit__(self,*args):self.close()

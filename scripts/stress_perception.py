"""Bounded asynchronous software soak, not a USB/physical-cell soak."""
import argparse
from dataclasses import replace
from pathlib import Path
from time import monotonic,sleep
import numpy as np
import psutil
from scipy.spatial.transform import Rotation
from active_scan.core import transform,write_json,Failure
from active_scan.perception.synthetic import test_calibration,render
from active_scan.perception.runtime import LocalizationRuntime


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    parser.add_argument('--seconds',type=float,default=30);args=parser.parse_args()
    if args.seconds<5:raise ValueError('Use at least 5 seconds')
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    c=test_calibration()
    pose=transform(Rotation.from_euler('xyz',[.2,.3,.1]).as_matrix(),[.26,.01,.08])
    template,_=render(c,pose,noise=.0009,holes=.12,outliers=.01)
    class Source:
        calibration=c
        def __init__(self):self.sequence=0;self.next=monotonic();self.closed=False
        def read(self):
            sleep(max(0,self.next-monotonic()));self.next=monotonic()+1/30
            self.sequence+=1
            return replace(template,sequence=self.sequence,capture_time=monotonic())
        def close(self):self.closed=True
    source=Source();process=psutil.Process();ages=[];gaps=[];rss=[];timings=[]
    checks=0;fresh=0;previous_sequence=-1;previous_ready=None;start=monotonic()
    with LocalizationRuntime(source) as runtime:
        observer=runtime.camera_observer(require_hardware=False)
        while monotonic()-start<args.seconds:
            elapsed=monotonic()-start
            if runtime.error is not None or runtime.capture.error is not None:raise RuntimeError('Worker failed during stress')
            if elapsed>2:
                checks+=1
                try:observer.observe(monotonic());fresh+=1
                except Failure:pass
            estimate=runtime.pipeline.latest
            if estimate is not None and estimate.sequence!=previous_sequence:
                ready=monotonic();previous_sequence=estimate.sequence
                ages.append((ready-estimate.timestamp)*1000)
                timings.append(estimate.metrics['processing_ms'])
                if previous_ready is not None:gaps.append((ready-previous_ready)*1000)
                previous_ready=ready
            if not rss or elapsed-rss[-1][0]>.25:rss.append([elapsed,process.memory_info().rss/1024**2])
            sleep(.005)
    stable=[v for t,v in rss if t>=2]
    growth=max(stable)-stable[0]
    def q(values):return dict(zip(('p50','p95','p99','max'),np.r_[np.percentile(values,[50,95,99]),max(values)].tolist()))
    gates={'fresh_observer_fraction':fresh/max(1,checks)>.95,'rss_growth_under_64MiB':growth<64,
           'threads_stopped':not runtime.thread.is_alive() and not runtime.capture.thread.is_alive(),'source_closed':source.closed}
    report={'source':'30Hz repeated synthetic image; actual host clocks/threads',
            'wall_seconds':monotonic()-start,'capture_frames':source.sequence,'processed_frames':runtime.processed,
            'dropped_frames':runtime.capture.buffer.dropped,'queue_capacity':1,'fresh_observer_fraction':fresh/max(1,checks),
            'capture_to_read_ms_mock_source':q(ages),'processing_ms':q(timings),'pose_update_gaps_ms':q(gaps),
            'rss_samples_seconds_MiB':rss,'rss_growth_after_warmup_MiB':growth,'software_gates':gates,
            'software_pass':all(gates.values()),'hardware_accuracy_validated':False,'robot_acceptance':False}
    write_json(out,report);print(__import__('json').dumps({k:v for k,v in report.items() if k!='rss_samples_seconds_MiB'},indent=2))
    if not report['software_pass']:raise SystemExit(1)


if __name__=='__main__':main()

"""Fixed-scene reference and table consistency from an explicitly empty cell."""
import numpy as np
from .calibration import deproject


class StaticScene:
    def __init__(self, calibration, depth, points_B, table_error_m):
        self.calibration=calibration;self.depth=np.array(depth,copy=True)
        self.depth.setflags(write=False);self.points_B=points_B;self.table_error_m=table_error_m

    @classmethod
    def capture_empty(cls, calibration, frames, *, cell_confirmed_empty=False):
        if not cell_confirmed_empty: raise ValueError('Empty-cell capture must be explicitly confirmed')
        frames=list(frames)
        if len(frames)<5: raise ValueError('At least five reference frames required')
        for frame in frames:
            if frame.calibration_id!=calibration.calibration_id or frame.depth_m.shape!=(calibration.height,calibration.width):
                raise ValueError('Reference calibration mismatch')
        stack=np.stack([f.depth_m for f in frames]);valid=(stack>=calibration.min_depth)&(stack<=calibration.max_depth)
        for i,f in enumerate(frames): valid[i]&=~f.robot_mask
        # Median only where at least 80% of actual measurements exist.
        masked=np.ma.array(stack,mask=~valid)
        depth=np.ma.median(masked,axis=0).filled(0).astype(np.float32)
        depth[valid.sum(axis=0)<np.ceil(.8*len(frames))]=0
        pc=deproject(depth,calibration)
        pb=pc @ calibration.T_B_C[:3,:3].T+calibration.T_B_C[:3,3]
        w=calibration.workspace
        region=(pb[:,0]>w[0])&(pb[:,0]<w[1])&(pb[:,1]>w[2])&(pb[:,1]<w[3])
        pb=pb[region]
        near_table=pb[np.abs(pb[:,2]-calibration.table_z)<.02]
        if len(near_table)<200: raise ValueError('Insufficient table evidence')
        error=float(np.median(near_table[:,2])-calibration.table_z)
        if abs(error)>.003: raise ValueError('Table/extrinsic mismatch; recalibrate camera')
        # Plane residual cannot detect translations in the table plane or yaw.
        return cls(calibration,depth,pb[::max(1,len(pb)//30000)],error)

    def foreground(self, frame, separation=.006):
        if frame.calibration_id!=self.calibration.calibration_id: raise ValueError('Reference calibration mismatch')
        return (self.depth>0)&(frame.depth_m>0)&(frame.depth_m<self.depth-separation)&~frame.robot_mask

    def check_drift(self,frame,limit_m=.004):
        """Detect observable fixed-background changes, not all camera motions.

        Pure translation parallel to an infinite plane / yaw around its normal
        is unobservable here. Independent calibrated anchors remain necessary.
        """
        if frame.calibration_id!=self.calibration.calibration_id:raise ValueError('Reference calibration mismatch')
        valid=(self.depth>0)&(frame.depth_m>0)&~frame.robot_mask
        difference=frame.depth_m-self.depth
        # Foreground objects may cover a minority of pixels. A median does not
        # selectively remove negative differences and hide a camera movement.
        if valid.sum()<500:return {'observable':False,'drift_detected':False,'reason':'insufficient_static_support'}
        median=float(np.median(difference[valid]))
        return {'observable':True,'drift_detected':abs(median)>limit_m,'background_median_delta_m':median}

    def save(self,path):
        np.savez_compressed(path,depth_m=self.depth,points_B=self.points_B,
                            calibration_id=self.calibration.calibration_id,table_error_m=self.table_error_m)

    @classmethod
    def load(cls,path,calibration):
        with np.load(path,allow_pickle=False) as data:
            if str(data['calibration_id'])!=calibration.calibration_id:raise ValueError('Scene calibration mismatch')
            depth=data['depth_m'];cloud=data['points_B'];error=float(data['table_error_m'])
            if depth.shape!=(calibration.height,calibration.width) or cloud.ndim!=2 or cloud.shape[1]!=3:
                raise ValueError('Invalid scene shapes')
            if not np.all(np.isfinite(depth)) or not np.all(np.isfinite(cloud)):raise ValueError('Invalid scene points')
            return cls(calibration,depth,cloud,error)

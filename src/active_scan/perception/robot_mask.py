"""Conservative robot silhouette from measured joints and collision geometry.

Only robot geometry is queried. Neither cube pose nor rendered segmentation
is consumed. Large masks reduce availability; they cannot fabricate evidence.
"""
from itertools import product
import cv2
import numpy as np
from ..core import inverse


class SimulatorRobotMask:
    """MuJoCo geometry adapter, also usable as an FK model on real joint data.

    Caller must synchronize measured joint positions to image capture time.
    The current physical robot model and gripper calibration still need bench
    validation. Do not share the simulator's scratch state across threads.
    """
    def __init__(self, simulator, calibration, padding_px=3,link_position_bound_m=0.):
        self.sim=simulator;self.calibration=calibration;self.padding=padding_px
        self.geometry=[]
        if not np.isfinite(link_position_bound_m) or link_position_bound_m<0:raise ValueError('Invalid robot position bound')
        self.link_position_bound_m=link_position_bound_m
        m=simulator.model
        for i in range(m.ngeom):
            if m.geom_group[i] not in (3,4): continue
            if int(m.geom_type[i])==7:  # mjGEOM_MESH; compiled vertices in geom frame
                mesh=int(m.geom_dataid[i]);start=m.mesh_vertadr[mesh];count=m.mesh_vertnum[mesh]
                vertices=m.mesh_vert[start:start+count]
                self.geometry.append((i,np.array(vertices,copy=True)))
                continue
            elif int(m.geom_type[i])==6:  # Exact box; a bounding sphere hides nearby cube faces.
                high=m.geom_size[i].copy();low=-high
            elif int(m.geom_type[i]) in (3,5):
                radius,halfheight=m.geom_size[i,:2]
                high=np.array([radius,radius,halfheight+(radius if int(m.geom_type[i])==3 else 0)])
                low=-high
            else:
                # Bounding sphere contains boxes, cylinders and capsules.
                radius=m.geom_rbound[i];low=np.full(3,-radius);high=np.full(3,radius)
            corners=np.array(list(product(*zip(low,high))))
            self.geometry.append((i,corners))

    def mask(self, q_measured, gripper_measured):
        c=self.calibration;d=self.sim.query_state(q_measured,gripper=gripper_measured)
        T_C_B=inverse(c.T_B_C)
        image=np.zeros((c.height,c.width),np.uint8)
        nearest=float('inf')
        for i,corners in self.geometry:
            world=corners @ d.geom_xmat[i].reshape(3,3).T+d.geom_xpos[i]
            camera=world @ T_C_B[:3,:3].T+T_C_B[:3,3]
            if np.any(camera[:,2]<=.01):
                if np.all(camera[:,2]<=.01): continue
                # Crossing the near plane cannot be projected safely.
                return np.ones(image.shape,bool)
            uv=camera @ c.K.T;uv=uv[:,:2]/uv[:,2,None]
            nearest=min(nearest,float(camera[:,2].min()))
            uv=np.clip(uv,-10000,10000).astype(np.int32)
            cv2.fillConvexPoly(image,cv2.convexHull(uv),1)
        extra=int(np.ceil(max(c.K[0,0],c.K[1,1])*self.link_position_bound_m/max(nearest,.05)))
        padding=self.padding+extra
        if padding>128:return np.ones(image.shape,bool)
        if padding:image=cv2.dilate(image,np.ones((2*padding+1,)*2,np.uint8))
        return image.astype(bool)

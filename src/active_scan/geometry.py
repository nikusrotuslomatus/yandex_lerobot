from dataclasses import dataclass
import itertools
import mujoco as mj
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from .core import transform, inverse, points, cube_surface, CoverageState, Failure


class Kinematics:
    def __init__(self, sim):
        self.sim = sim

    def forward(self, q):
        return self.sim.tcp_pose(self.sim.query_state(q))

    def jacobian(self, q):
        s = self.sim; d = s.query_state(q)
        jp = np.zeros((3, s.model.nv)); jr = jp.copy()
        mj.mj_jacSite(s.model, d, jp, jr, s.tcp)
        return np.vstack([jp[:, s.vadr], jr[:, s.vadr]])

    def solve_ik(self, target, seed, orientation=True):
        s = self.sim; cfg = s.cfg['planning']
        margin=cfg.get('ik_joint_margin',0.)
        lo=s.limits[:,0]+margin;hi=s.limits[:,1]-margin
        seed = np.clip(seed, lo + 1e-7, hi - 1e-7)
        def residual(q):
            T = self.forward(q)
            p = (T[:3, 3] - target[:3, 3]) * 10
            R = Rotation.from_matrix(target[:3, :3] @ T[:3, :3].T).as_rotvec()
            return np.r_[p, R if orientation else [], .0001*(q-seed)]
        def jacobian(q):
            T=self.forward(q);J=self.jacobian(q)
            if not orientation:return np.vstack([10*J[:3],.0001*np.eye(5)])
            phi=Rotation.from_matrix(target[:3,:3]@T[:3,:3].T).as_rotvec()
            x,y,z=phi;hat=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
            theta=np.linalg.norm(phi)
            coefficient=1/12 if theta<1e-5 else (1-.5*theta/np.tan(.5*theta))/theta**2
            right_inverse=np.eye(3)+.5*hat+coefficient*(hat@hat)
            return np.vstack([10*J[:3],-right_inverse@J[3:],.0001*np.eye(5)])
        fit = least_squares(residual, seed, jac=jacobian,bounds=(lo+1e-8, hi-1e-8), max_nfev=cfg['ik_max_evaluations'], ftol=1e-9, xtol=1e-9, gtol=1e-9)
        T = self.forward(fit.x)
        ep = np.linalg.norm(T[:3,3]-target[:3,3])
        er = np.linalg.norm(Rotation.from_matrix(T[:3,:3].T @ target[:3,:3]).as_rotvec())
        if ep > cfg['ik_position_tolerance'] or (orientation and er > cfg['ik_orientation_tolerance']):
            raise Failure('TARGET_POSE_UNREACHABLE', {'position_error':ep, 'orientation_error':er})
        return fit.x


class CollisionChecker:
    """Explicit geom-distance checks include collision-filtered pairs.

    Same-body and direct parent-child robot pairs are intentional assembly overlaps.
    All other collision geoms are checked, regardless of contype/conaffinity masks.
    Visual-only meshes are outside this conservative collision representation.
    """
    def __init__(self, sim):
        self.s = sim; m = sim.model
        self.ids = [i for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]]
        self.pairs = []
        for a,b in itertools.combinations(self.ids, 2):
            ba,bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
            if ba == bb:
                continue
            if ba and bb and (m.body_parentid[ba] == bb or m.body_parentid[bb] == ba):
                continue
            if (a == sim.table and m.geom(b).name == 'base_collision') or (b == sim.table and m.geom(a).name == 'base_collision'):
                continue
            self.pairs.append((a,b))
        self.fingers = {i for i in self.ids if m.geom_bodyid[i] in (sim.grip,sim.moving)}
        self.radius = m.geom_rbound.copy()
        self.pair_array=np.asarray(self.pairs,dtype=int)
        self.radius_sums=self.radius[self.pair_array[:,0]]+self.radius[self.pair_array[:,1]]

    def check(self, d=None, mode='FREE', clearance=None):
        s = self.s; m = s.model; d = d or s.data
        margin = s.cfg['planning']['collision_margin'] if clearance is None else clearance
        tol = s.cfg['planning']['max_contact_penetration']
        bad=[]; min_dist=1.0
        self.last_slack=1.
        delta=d.geom_xpos[self.pair_array[:,0]]-d.geom_xpos[self.pair_array[:,1]]
        bounds=np.linalg.norm(delta,axis=1)-self.radius_sums
        if s.cfg['planning'].get('aabb_broadphase',False):
            # Rotated local AABBs are conservative world bounds. In particular,
            # the wide, thin table no longer looks like a huge enclosing sphere.
            rotation=d.geom_xmat.reshape(-1,3,3);aabb=m.geom_aabb
            centers=d.geom_xpos+np.einsum('nij,nj->ni',rotation,aabb[:,:3])
            extents=np.einsum('nij,nj->ni',np.abs(rotation),aabb[:,3:])
            a,b=self.pair_array.T
            separation=np.maximum(np.abs(centers[a]-centers[b])-extents[a]-extents[b],0)
            bounds=np.maximum(bounds,np.linalg.norm(separation,axis=1))
        near=bounds<=max(margin,.01)
        if np.any(~near):
            min_dist=min(min_dist,float(np.min(bounds[~near])))
            self.last_slack=min(self.last_slack,float(np.min(bounds[~near]))-margin)
        for a,b in self.pair_array[near]:
            contact_cube = s.cubegeom in (a,b)
            other = b if a == s.cubegeom else a
            allowed = contact_cube and ((other in self.fingers and mode in ('GRASP','HELD','PLACE')) or (other == s.table and mode in ('TABLE','GRASP','PLACE')))
            dist = mj.mj_geomDistance(m, d, a, b, max(.02,margin), None)
            self.last_slack=min(self.last_slack,dist+tol if allowed else dist-margin)
            if not allowed:
                min_dist=min(min_dist,dist)
            if dist < (-tol if allowed else margin):
                bad.append({'a':m.geom(a).name,'b':m.geom(b).name,'distance':dist,'allowed_contact':allowed})
        return bad, min_dist

    def grasp_intersections(self, d):
        s=self.s
        return {s.model.geom(i).name: mj.mj_geomDistance(s.model,d,s.cubegeom,i,.10,None)
                for i in self.ids if s.model.geom_bodyid[i] in (s.grip,s.moving)}


class VisibilityEvaluator:
    def __init__(self, sim):
        self.s = sim; cfg=sim.cfg
        self.local, self.normals, self.faces = cube_surface(cfg['scene']['cube_size'],cfg['visibility']['grid'])
        self.groups = np.array([1,1,1,0,0,0],dtype=np.uint8)

    def project(self, world):
        p=points(inverse(self.s.camera_pose()),world)
        uv=p[:,:2]/np.maximum(p[:,2:3],1e-12)
        uv=uv @ self.s.K[:2,:2].T+self.s.K[:2,2]
        return uv,p[:,2]

    def evaluate(self, d=None):
        s=self.s; d=d or s.data; cfg=s.cfg['visibility']; cam=s.cfg['camera']
        T=s.cube_pose(d); p=points(T,self.local); n=self.normals @ T[:3,:3].T
        origin=s.data.cam_xpos[s.camera].copy()
        vectors=p-origin; lengths=np.linalg.norm(vectors,axis=1)
        directions=-vectors/lengths[:,None]
        mu=np.einsum('ij,ij->i',n,directions)
        uv,z=self.project(p)
        valid=(mu>0)&(z>max(cam['near'],cfg['min_distance']))&(z<min(cam['far'],cfg['max_distance']))&(uv[:,0]>=0)&(uv[:,0]<cam['width'])&(uv[:,1]>=0)&(uv[:,1]<cam['height'])
        quality=np.zeros(len(p)); occluded=np.ones(len(p),bool)
        idx=np.flatnonzero(valid)
        if len(idx):
            ids=np.empty(len(idx),np.int32); dist=np.empty(len(idx))
            ray=np.ascontiguousarray(vectors[idx]/lengths[idx,None])
            mj.mj_multiRay(s.model,d,origin,ray.ravel(),self.groups,True,-1,ids,dist,None,len(idx),cam['far'])
            visible=(ids==s.cubegeom)&(np.abs(dist-lengths[idx])<cfg['ray_tolerance'])
            occluded[idx]=~visible
            pixel_area=(s.cfg['scene']['cube_size']/s.cfg['visibility']['grid'])**2*s.K[0,0]*s.K[1,1]*mu[idx]/z[idx]**2
            scale=np.clip(pixel_area/cfg['min_pixels_per_element'],0,1)
            quality[idx]=visible*mu[idx]**cfg['angle_power']*scale
        return quality,directions @ T[:3,:3],occluded

    def coverage(self):
        c=self.s.cfg['visibility']
        return CoverageState(len(self.local),c['quality_threshold'],c['diversity_deg'])

    def frame_metrics(self,d=None,evaluation=None,segmentation=None):
        """Whole-image suitability, independent of cumulative surface coverage.

        Planning uses a projected-area quadrature over ray-tested surface cells.
        Capture uses the actual segmentation pixel count. Neither is a claim
        about downstream reconstruction accuracy.
        """
        s=self.s;d=d or s.data;T=s.cube_pose(d);C=inverse(s.camera_pose())@T
        a=s.cfg['scene']['cube_size']/2
        corners=np.array(list(itertools.product([-a,a],repeat=3)))
        uv,z=self.project(points(T,corners));extent=np.ptp(uv,axis=0)
        silhouette=float(ConvexHull(uv).volume) if np.all(z>0) else 0.
        quality,directions,occluded=evaluation or self.evaluate(d)
        pc=points(C,self.local);nc=self.normals@C[:3,:3].T
        front=np.maximum(-np.sum(nc*pc,axis=1),0)
        projected=(s.cfg['scene']['cube_size']/s.cfg['visibility']['grid'])**2*s.K[0,0]*s.K[1,1]*front/np.maximum(pc[:,2],1e-9)**3
        pixels=float(np.sum(projected[~occluded])) if segmentation is None else float(np.count_nonzero((segmentation[:,:,0]==s.cubegeom)&(segmentation[:,:,1]==int(mj.mjtObj.mjOBJ_GEOM))))
        cam=s.cfg['camera'];padding=float(min(uv[:,0].min(),uv[:,1].min(),cam['width']-uv[:,0].max(),cam['height']-uv[:,1].max()))
        return {'distance_m':float(np.linalg.norm(C[:3,3])),'bbox_short_px':float(extent.min()),
                'bbox_width_px':float(extent[0]),'bbox_height_px':float(extent[1]),
                'silhouette_pixels':silhouette,'visible_pixels':pixels,
                'visible_fraction':float(np.clip(pixels/max(silhouette,1),0,1)),
                'border_padding_px':padding,'pixel_measurement':'segmentation' if segmentation is not None else 'ray_area_estimate'}

    def frame_rejections(self,metrics,planning=False):
        cfg=self.s.cfg.get('frame_quality',{})
        if not cfg.get('enabled',False):return []
        safety=cfg.get('planning_pixel_safety',1.) if planning else 1.
        checks={'distance':metrics['distance_m']<=cfg['max_distance_m'],
                'object_scale':metrics['bbox_short_px']>=cfg['min_bbox_short_px'],
                'visible_pixels':metrics['visible_pixels']>=cfg['min_visible_pixels']*safety,
                'occlusion':metrics['visible_fraction']>=cfg['min_visible_fraction'],
                'border':metrics['border_padding_px']>=cfg['border_padding_px']}
        return [key for key,passed in checks.items() if not passed]


@dataclass
class GraspCandidate:
    grasp_id: str
    T_G_O: np.ndarray
    gripper: float
    contact_axis: int
    required_opening: float
    clearance: float


def stable_cube_rotations():
    result=[]
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1,1),repeat=3):
            R=np.eye(3)[:,perm]*signs
            if np.linalg.det(R)>.5:
                result.append(R)
    return result


class GraspPlanner:
    def __init__(self, sim, collisions):
        self.s=sim; self.collisions=collisions

    def aperture_audit(self):
        s=self.s; cfg=s.cfg['planning']; rows=[]
        q=np.array(s.cfg['robot']['home'])
        # Rays at three lateral offsets and four depths, in the gripper body frame.
        groups=np.array([0,0,0,1,1,0],np.uint8)
        for g in np.linspace(*s.model.jnt_range[s.gj],cfg['aperture_samples']):
            d=s.query_state(q,gripper=g)
            R=d.xmat[s.grip].reshape(3,3); p=d.xpos[s.grip]
            for z in cfg['aperture_z']:
                for y in cfg['aperture_y']:
                    origin=p+R@np.array([.025,y,z]); hit=np.empty(1,np.int32)
                    minus=mj.mj_ray(s.model,d,origin,-R[:,0],groups,True,-1,hit)
                    minus_id=int(hit[0]); plus=mj.mj_ray(s.model,d,origin,R[:,0],groups,True,-1,hit)
                    bodies={int(s.model.geom_bodyid[minus_id]),int(s.model.geom_bodyid[int(hit[0])])} if minus_id>=0 and hit[0]>=0 else set()
                    if minus>=0 and plus>=0 and bodies=={s.grip,s.moving}:
                        rows.append({'g':g,'z':z,'y':y,'gap':minus+plus,'left_geom':s.model.geom(minus_id).name,'right_geom':s.model.geom(int(hit[0])).name})
        return rows

    def geometric_candidates(self,all_fits=False):
        """Whole-box clearance before assigning a hypothesized rigid attachment.

        Enumerate depth/centre/aperture; reject any penetration with the original
        collision geoms. This is a fit test, never a force-closure certificate.
        """
        s=self.s; q=np.array(s.cfg['robot']['home']); cfg=s.cfg['planning']
        d=s.query_state(q)
        B=transform(d.xmat[s.grip].reshape(3,3),d.xpos[s.grip])
        T_B_G=inverse(B)@s.tcp_pose(d)
        candidates=[]; checked=0; best_pen=-1.
        for z,x,g in itertools.product(cfg['grasp_depths'],cfg['grasp_x'],np.linspace(*cfg['fit_gripper_range'],cfg['fit_gripper_samples'])):
            # Axis X is the closing axis at zero; tapered jaws motivate a Y tilt.
            for tilt in cfg['fit_tilts']:
                TO=transform(Rotation.from_euler('y',tilt).as_matrix(),[x,0,z])
                relative=inverse(T_B_G)@TO
                d=s.query_state(q,relative,gripper=g)
                dist=self.collisions.grasp_intersections(d)
                depth=min(dist.values()); best_pen=max(best_pen,depth); checked+=1
                # Require proximity to BOTH fingers in addition to no penetration.
                fixed=min(v for k,v in dist.items() if s.model.geom(k).bodyid[0]==s.grip)
                moving=min(v for k,v in dist.items() if s.model.geom(k).bodyid[0]==s.moving)
                if depth>=-cfg['fit_penetration_tolerance'] and max(fixed,moving)<cfg['fit_finger_proximity']:
                    candidates.append((abs(fixed)+abs(moving),g,relative,z,x,tilt,depth))
        candidates.sort(key=lambda v:v[0])
        if not candidates:
            return [],{'checked':checked,'best_min_distance':best_pen,'failure':'NO_CENTERED_GEOMETRIC_GRASP'}
        best=candidates[0]; _,g,relative,z,x,tilt,clearance=best
        result=[]
        # Preserve physical face identity: change the cube-to-gripper rotation.
        # These are geometric hypotheses. Reachable place/regrasp paths are separate.
        fits=[best]
        if all_fits:
            seen={(best[3],best[5])}
            for candidate in candidates:
                key=(candidate[3],candidate[5])
                if key not in seen:fits.append(candidate);seen.add(key)
        for fit_id,fit in enumerate(fits):
            _,fg,frelative,*_=fit
            for axis in (0,1,2):
                R=np.roll(np.eye(3),axis,axis=1)
                rel=frelative@transform(R)
                name=f'axis_{axis}' if not all_fits else f'fit_{fit_id}_axis_{axis}'
                result.append(GraspCandidate(name,rel,fg,axis,s.cfg['scene']['cube_size'],fit[-1]))
        return result,{'checked':checked,'valid_fits':len(candidates),'best_min_distance':best_pen,'selected':{'g':g,'gripper_body_cube_center':[x,0,z],'tilt':tilt,'clearance':clearance},'status':'geometric_fit_only'}

"""Reachable joint-space views, bounded greedy selection and quintic motion."""
from dataclasses import dataclass
from collections import Counter
import time
import heapq
import numpy as np
from scipy.stats import qmc
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from .core import Failure, inverse, transform
from .geometry import Kinematics, stable_cube_rotations


@dataclass
class ViewCandidate:
    q: np.ndarray
    T_W_G: np.ndarray
    T_W_O: np.ndarray
    T_C_O: np.ndarray
    quality: np.ndarray
    directions: np.ndarray
    clearance: float
    limit_margin: float
    frame_quality: dict | None = None


class ViewPlanner:
    def __init__(self, sim, collisions, visibility):
        self.s,self.collisions,self.v=sim,collisions,visibility

    def generate_candidates(self, grasp, seed=None):
        s=self.s;cfg=s.cfg['planning'];n=cfg['candidates'];seed=s.cfg['seed'] if seed is None else seed
        lo=s.limits[:,0]+cfg['joint_margin'];hi=s.limits[:,1]-cfg['joint_margin']
        sampler=qmc.Sobol(5,scramble=True,seed=seed)
        qs=qmc.scale(sampler.random_base2(int(np.ceil(np.log2(n))))[:n],lo,hi)
        # Add structured wrist motions around the valid home seed.
        structured=[]
        for flex in np.linspace(lo[3],hi[3],7):
            for roll in np.linspace(lo[4],hi[4],17):
                q=np.array(s.cfg['robot']['home']);q[3:]=[flex,roll];structured.append(q)
        qs=np.vstack([qs,structured]);accepted=[];reasons=Counter();start=time.perf_counter()
        for i,q in enumerate(qs):
            d=s.query_state(q,grasp.T_G_O,gripper=grasp.gripper)
            T=s.cube_pose(d)
            if not cfg['min_cube_height']<T[2,3]<cfg['max_cube_height']:
                reasons['height']+=1;continue
            # Require the whole cube's eight corners inside image (not just its centre).
            corners=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])*s.cfg['scene']['cube_size']/2
            from .core import points
            uv,z=self.v.project(points(T,corners));cam=s.cfg['camera']
            if np.any(z<cam['near']) or np.any(uv<0) or np.any(uv[:,0]>=cam['width']) or np.any(uv[:,1]>=cam['height']):
                reasons['frustum']+=1;continue
            frame_cfg=s.cfg.get('frame_quality',{})
            if frame_cfg.get('enabled',False):
                C=inverse(s.camera_pose())@T
                if np.linalg.norm(C[:3,3])>frame_cfg['max_distance_m']:
                    reasons['frame_distance']+=1;continue
                if np.min(np.ptp(uv,axis=0))<frame_cfg['min_bbox_short_px']:
                    reasons['frame_object_scale']+=1;continue
                padding=min(uv[:,0].min(),uv[:,1].min(),cam['width']-uv[:,0].max(),cam['height']-uv[:,1].max())
                if padding<frame_cfg['border_padding_px']:
                    reasons['frame_border']+=1;continue
            bad,clearance=self.collisions.check(d,'HELD')
            if bad:
                reasons['collision']+=1;continue
            evaluation=self.v.evaluate(d);quality,directions,_=evaluation
            frame=self.v.frame_metrics(d,evaluation)
            rejected=self.v.frame_rejections(frame,planning=True)
            if rejected:
                reasons['frame_'+rejected[0]]+=1;continue
            if not np.any(quality>=s.cfg['visibility']['quality_threshold']):
                reasons['uninformative']+=1;continue
            accepted.append(ViewCandidate(q.copy(),s.tcp_pose(d),T,inverse(s.camera_pose())@T,quality,directions,clearance,float(np.min(np.minimum(q-lo,hi-q))),frame))
        return accepted,{'sampled':len(qs),'accepted':len(accepted),'rejected':dict(reasons),'planning_seconds':time.perf_counter()-start,'seed':seed}

    def validate_candidate(self,q,grasp):
        """Recheck a library pose against the actual grasp, without global sampling."""
        s=self.s;cfg=s.cfg['planning'];d=s.query_state(q,grasp.T_G_O,gripper=grasp.gripper)
        T=s.cube_pose(d)
        if not cfg['min_cube_height']<T[2,3]<cfg['max_cube_height']:return None
        bad,clearance=self.collisions.check(d,'HELD')
        if bad:return None
        evaluation=self.v.evaluate(d);frame=self.v.frame_metrics(d,evaluation)
        if self.v.frame_rejections(frame,planning=True):return None
        if not np.any(evaluation[0]>=s.cfg['visibility']['quality_threshold']):return None
        return ViewCandidate(np.asarray(q).copy(),s.tcp_pose(d),T,inverse(s.camera_pose())@T,
                             evaluation[0],evaluation[1],clearance,0.,frame)

    def select_views(self, candidates, initial=None, start_q=None, max_views=None,priority=None):
        if not candidates:return []
        cfg=self.s.cfg['planning'];threshold=self.s.cfg['visibility']['quality_threshold']
        coverage=np.zeros_like(candidates[0].quality) if initial is None else initial.copy()
        weights=np.ones_like(coverage) if priority is None else np.asarray(priority)
        selected=[];q=np.array(self.s.cfg['robot']['home'] if start_q is None else start_q)
        remaining=list(candidates)
        for _ in range(cfg['max_views'] if max_views is None else max_views):
            gains=np.array([np.mean(weights*((c.quality>=threshold)&(coverage<threshold)))+.1*np.mean(weights*(np.maximum(coverage,c.quality)-coverage)) for c in remaining])
            costs=np.array([motion_duration(c.q-q,self.s.cfg) for c in remaining])
            scores=gains-cfg['movement_weight']*costs
            if cfg.get('time_normalized_gain',False):scores=gains/(costs+self.s.cfg['motion']['settle_time'])
            eligible=gains>=cfg['min_gain']
            if not np.any(eligible):break
            scores=np.where(eligible,scores,-np.inf)
            i=int(np.argmax(scores))
            c=remaining.pop(i);selected.append(c);q=c.q
            coverage=np.maximum(coverage,c.quality)
            if priority is not None and np.sum(weights*(coverage>=threshold))/np.sum(weights)>=cfg.get('first_stage_priority_coverage',.98):break
            if cfg.get('time_normalized_gain',False) and np.mean(coverage>=threshold)>=cfg['target_coverage']:break
        return selected

    def order_views(self, selected, start_q):
        todo=list(selected);ordered=[];q=start_q
        while todo:
            idx=int(np.argmin([motion_duration(v.q-q,self.s.cfg) for v in todo]))
            c=todo.pop(idx);ordered.append(c);q=c.q
        return ordered


def motion_duration(delta,cfg):
    reserve=cfg['motion'].get('reference_limit_fraction',1.)
    v=np.asarray(cfg['motion']['velocity'])*reserve;a=np.asarray(cfg['motion']['acceleration'])*reserve
    # max s'=1.875, max |s''|=10/sqrt(3) for the quintic smoothstep.
    return max(cfg['motion']['min_duration'],float(np.max(1.875*np.abs(delta)/v)),float(np.sqrt(np.max((10/np.sqrt(3))*np.abs(delta)/a))))


@dataclass
class Trajectory:
    start: np.ndarray
    goal: np.ndarray
    duration: float

    def sample(self,t):
        u=np.clip(t/self.duration,0,1)
        delta=self.goal-self.start
        return (self.start+(10*u**3-15*u**4+6*u**5)*delta,
                (30*u**2-60*u**3+30*u**4)*delta/self.duration,
                (60*u-180*u**2+120*u**3)*delta/self.duration**2)


class SplineTrajectory:
    """One timed C2 path through IK nodes; only its two ends stop.

    IK nodes are collision-checking geometry, not twelve separate servo moves.
    A global quintic time law supplies zero end velocity/acceleration. Duration
    includes curvature acceleration, not just the endpoint joint displacement.
    """
    def __init__(self,waypoints,cfg):
        values=np.asarray(waypoints,float)
        keep=np.r_[True,np.max(np.abs(np.diff(values,axis=0)),axis=1)>1e-10]
        self.waypoints=values[keep]
        if len(self.waypoints)<2:raise Failure('EMPTY_CARTESIAN_PATH')
        lengths=np.linalg.norm(np.diff(self.waypoints,axis=0),axis=1)
        knots=np.r_[0,np.cumsum(lengths)];knots/=knots[-1]
        self.curve=CubicSpline(knots,self.waypoints,axis=0,bc_type='natural')
        self.start=self.waypoints[0].copy();self.goal=self.waypoints[-1].copy()
        self.duration=1.
        _,vel,acc=self.sample(np.linspace(0,1,2001))
        vmax=np.max(np.abs(vel),axis=0)
        amax=np.max(np.abs(acc),axis=0)
        motion=cfg['motion']
        reserve=motion.get('reference_limit_fraction',1.)
        self.duration=max(motion['min_duration'],float(np.max(vmax/(np.asarray(motion['velocity'])*reserve))),
                          float(np.sqrt(np.max(amax/(np.asarray(motion['acceleration'])*reserve)))))*motion.get('spline_timing_safety',1.02)

    def sample(self,t):
        u=np.clip(t/self.duration,0,1);s=10*u**3-15*u**4+6*u**5
        ds=30*u**2-60*u**3+30*u**4;dds=60*u-180*u**2+120*u**3
        first=self.curve(s,1);second=self.curve(s,2)
        return self.curve(s),first*(ds/self.duration)[...,None],(second*(ds**2)[...,None]+first*dds[...,None])/self.duration**2


class BrakeTrajectory:
    """Monotone velocity decay from measured motion to a stationary endpoint."""
    def __init__(self,q,velocity,deceleration,min_duration=.04):
        self.start=np.asarray(q).copy();self.velocity=np.asarray(velocity).copy()
        self.duration=max(min_duration,float(np.max(2*np.abs(self.velocity)/np.asarray(deceleration))))
        self.goal=self.start+self.velocity*self.duration/3

    def sample(self,t):
        u=np.clip(t/self.duration,0,1)
        return (self.start+self.velocity*self.duration*(u-u*u+u**3/3),
                self.velocity*(1-u)**2,-2*self.velocity*(1-u)/self.duration)


class MotionPlanner:
    def __init__(self,sim,collisions):
        self.s=sim;self.cc=collisions;self.k=Kinematics(sim)
        self._last_route=None
        self.lever_bound=float(np.sum(np.linalg.norm(sim.model.body_pos,axis=1))+np.max(sim.model.geom_rbound))

    def _context(self,T_G_O,gripper):
        return {'relative':np.eye(4) if T_G_O is None else np.asarray(T_G_O),
                'cube':self.s.cube_pose() if T_G_O is None else np.eye(4),
                'gripper':float(self.s.data.qpos[self.s.gq]) if gripper is None else gripper}

    def _change_bound(self,state,prior,joint_change):
        bound=2*self.lever_bound*(joint_change+abs(state['gripper']-prior['gripper']))
        for name in ('relative','cube'):
            delta=state[name]-np.array(prior[name])
            bound+=2*(np.linalg.norm(delta[:3,3])+self.s.cfg['scene']['cube_size']*np.linalg.norm(delta[:3,:3]))
        return bound

    def plan_joint_motion(self,start,goal,T_G_O=None,gripper=None,mode='FREE'):
        s=self.s
        if np.any(goal<s.limits[:,0]) or np.any(goal>s.limits[:,1]):raise Failure('JOINT_LIMIT_VIOLATION')
        t=Trajectory(np.array(start),np.array(goal),motion_duration(np.array(goal)-start,s.cfg))
        library=getattr(self,'library',None)
        state={'start':np.asarray(start),'goal':np.asarray(goal),**self._context(T_G_O,gripper)}
        key='line_'+mode+'_'+str(T_G_O is not None)+'_'+str(np.round(np.r_[start,goal],2).tolist())
        prior=library.get(key,required=False) if library is not None else None
        if prior:
            # A conservative displacement bound preserves the checked sampled
            # collision margin under tiny changes of endpoints/attachment.
            # Larger pose changes always trigger the complete geometric check.
            dq=max(np.sum(np.abs(state['start']-prior['start'])),np.sum(np.abs(state['goal']-prior['goal'])))
            bound=self._change_bound(state,prior,dq)
            if bound<prior['slack']*.5:
                library.certificate_hits+=1
                return t
        if library is not None and library.enabled:library.certificate_rechecks+=1
        count=max(3,int(np.ceil(np.max(np.abs(t.goal-t.start))/s.cfg['motion']['path_joint_step']))+1)
        checks=np.linspace(0,1,count);max_delta=float(np.max(np.abs(t.goal-t.start)))
        clearances=[];slack=1.
        for u in checks:
            q=t.start+u*(t.goal-t.start) # same locus as monotone quintic
            d=s.query_state(q,T_G_O,gripper=gripper)
            bad,clearance=self.cc.check(d,mode);clearances.append(clearance)
            slack=min(slack,getattr(self.cc,'last_slack',0.))
            if bad:raise Failure('COLLISION_PATH',{'fraction':u,'contacts':bad})
        # Refine intervals bordering a near-obstacle sample. This is a discrete
        # safety check at documented resolution, not a continuous-time proof.
        for i in range(count-1):
            if min(clearances[i:i+2])>=s.cfg['motion']['path_near_clearance']:continue
            subdivisions=max(2,int(np.ceil((checks[i+1]-checks[i])*max_delta/s.cfg['motion']['path_joint_step_near'])))
            for u in np.linspace(checks[i],checks[i+1],subdivisions+1)[1:-1]:
                d=s.query_state(t.start+u*(t.goal-t.start),T_G_O,gripper=gripper)
                bad,_=self.cc.check(d,mode)
                slack=min(slack,getattr(self.cc,'last_slack',0.))
                if bad:raise Failure('COLLISION_PATH',{'fraction':u,'contacts':bad})
        if library is not None:library.put(key,{**state,'slack':slack})
        return t

    def cartesian(self,start,target,T_G_O=None,gripper=None,mode='GRASP',steps=12,orientation=True):
        origin=self.k.forward(start);q=np.array(start);result=[]
        library=getattr(self,'library',None)
        key='cartesian_'+mode+'_'+str(orientation)+'_'+str(np.round(np.r_[origin[:3,3],target[:3,3]],2).tolist())
        hint=library.get(key,required=False) if library is not None else None
        if hint and self.s.cfg['motion'].get('blend_cartesian',False):
            nodes=np.array(hint['waypoints']);end=self.k.forward(nodes[-1])
            origin_error=np.linalg.norm(origin[:3,3]-np.array(hint['origin'])[:3,3])
            position_error=np.linalg.norm(end[:3,3]-target[:3,3])
            angle=np.linalg.norm(Rotation.from_matrix(end[:3,:3].T@target[:3,:3]).as_rotvec())
            if origin_error<.0001 and position_error<.0001 and (not orientation or angle<.001):
                try:return [self.plan_spline(np.vstack([start,nodes[1:]]),T_G_O,gripper,mode)]
                except Failure:pass
        rotations=Slerp([0,1],Rotation.from_matrix([origin[:3,:3],target[:3,:3]]))
        for u in np.linspace(0,1,steps+1)[1:]:
            T=target.copy();T[:3,3]=(1-u)*origin[:3,3]+u*target[:3,3]
            T[:3,:3]=rotations(u).as_matrix()
            goal=self.k.solve_ik(T,q,orientation=orientation)
            result.append(self.plan_joint_motion(q,goal,T_G_O,gripper,mode));q=goal
        if self.s.cfg['motion'].get('blend_cartesian',False):
            tr=self.plan_spline([start]+[tr.goal for tr in result],T_G_O,gripper,mode)
            if library is not None:library.put(key,{'waypoints':tr.waypoints,'origin':origin})
            return [tr]
        return result

    def plan_spline(self,waypoints,T_G_O=None,gripper=None,mode='HELD'):
        s=self.s;tr=SplineTrajectory(waypoints,s.cfg)
        library=getattr(self,'library',None);context=self._context(T_G_O,gripper)
        key='spline_'+mode+'_'+str(T_G_O is not None)+'_'+str(np.round(tr.waypoints,2).tolist())
        prior=library.get(key,required=False) if library is not None else None
        path_length=np.sum(np.max(np.abs(np.diff(tr.waypoints,axis=0)),axis=1))
        count=len(prior['samples']) if prior else max(65,int(np.ceil(path_length/s.cfg['motion']['path_joint_step_near']))+1)
        while True:
            times=np.linspace(0,tr.duration,count);values=tr.sample(times)[0]
            if np.max(np.abs(np.diff(values,axis=0)))<=s.cfg['motion']['path_joint_step_near']*1.001:break
            count*=2
        if np.any(values<s.limits[:,0]) or np.any(values>s.limits[:,1]):raise Failure('JOINT_LIMIT_VIOLATION')
        if prior and np.shape(prior['samples'])==values.shape:
            dq=float(np.max(np.sum(np.abs(values-np.asarray(prior['samples'])),axis=1)))
            if self._change_bound(context,prior,dq)<prior['slack']*.5:
                library.certificate_hits+=1;return tr
        if library is not None and library.enabled:library.certificate_rechecks+=1
        slack=1.
        for t,q in zip(times,values):
            if np.any(q<s.limits[:,0]) or np.any(q>s.limits[:,1]):raise Failure('JOINT_LIMIT_VIOLATION')
            bad,_=self.cc.check(s.query_state(q,T_G_O,gripper=gripper),mode)
            slack=min(slack,getattr(self.cc,'last_slack',0.))
            if bad:raise Failure('COLLISION_PATH',{'fraction':t/tr.duration,'contacts':bad})
        if library is not None:library.put(key,{**context,'samples':values,'slack':slack})
        return tr

    def reverse(self,tr,T_G_O=None,gripper=None,mode='PLACE'):
        if isinstance(tr,SplineTrajectory):return self.plan_spline(tr.waypoints[::-1],T_G_O,gripper,mode)
        return self.plan_joint_motion(tr.goal,tr.start,T_G_O,gripper,mode)

    def via_home(self,start,goal,T_G_O=None,gripper=None,mode='HELD'):
        # Pick feasibility and subsequent execution often request exactly the
        # same route without advancing the simulator. Reuse that checked result.
        signature=(np.asarray(start).tobytes(),np.asarray(goal).tobytes(),
                   None if T_G_O is None else np.asarray(T_G_O).tobytes(),gripper,mode,
                   self.s.cube_pose().tobytes())
        if self._last_route is not None and self._last_route[0]==signature:return self._last_route[1]
        library=getattr(self,'library',None)
        key='route_'+mode+'_'+str(np.round(goal,2).tolist())+'_'+str(T_G_O is not None)
        hint=library.get(key,required=False) if library is not None else None
        if hint:
            nodes=[np.asarray(start),*[np.array(q) for q in hint['interior']],np.asarray(goal)]
            try:
                route=[self.plan_joint_motion(a,b,T_G_O,gripper,mode) for a,b in zip(nodes,nodes[1:])]
                if len(nodes)>2 and self.s.cfg['motion'].get('blend_joint_waypoints',False):
                    try:route=[self.plan_spline(nodes,T_G_O,gripper,mode)]
                    except Failure:pass
                self._last_route=(signature,route);return route
            except Failure:pass
        route=self._via_home(start,goal,T_G_O,gripper,mode)
        self._last_route=(signature,route)
        if library is not None:
            nodes=list(route[0].waypoints) if len(route)==1 and isinstance(route[0],SplineTrajectory) else [route[0].start]+[tr.goal for tr in route]
            library.put(key,{'interior':nodes[1:-1]})
        return route

    def _via_home(self,start,goal,T_G_O=None,gripper=None,mode='HELD'):
        try:return [self.plan_joint_motion(start,goal,T_G_O,gripper,mode)]
        except Failure as direct:
            if self.s.cfg['motion'].get('safe_waypoint_graph',False):
                nodes=[np.asarray(start)]
                for value in self.s.cfg['motion']['safe_configurations']:
                    value=np.asarray(value)
                    if all(np.max(np.abs(value-n))>1e-6 for n in nodes):nodes.append(value)
                nodes.append(np.asarray(goal));last=len(nodes)-1
                queue=[(0.,0)];distance={0:0.};parent={};edges={}
                while queue:
                    cost,u=heapq.heappop(queue)
                    if cost>distance[u]+1e-9:continue
                    if u==last:
                        indices=[last]
                        while indices[-1]!=0:indices.append(parent[indices[-1]])
                        indices.reverse()
                        if self.s.cfg['motion'].get('blend_joint_waypoints',False):
                            try:return [self.plan_spline([nodes[i] for i in indices],T_G_O,gripper,mode)]
                            except Failure:pass
                        return [edges[(a,b)] for a,b in zip(indices,indices[1:])]
                    for v in range(len(nodes)):
                        if v==u:continue
                        if (u,v) not in edges:
                            try:
                                edge=self.plan_joint_motion(nodes[u],nodes[v],T_G_O,gripper,mode)
                                edges[(u,v)]=edge;edges[(v,u)]=Trajectory(edge.goal,edge.start,edge.duration)
                            except Failure:edges[(u,v)]=edges[(v,u)]=None
                        edge=edges[(u,v)]
                        if edge is None:continue
                        new=cost+edge.duration
                        if new<distance.get(v,float('inf')):
                            distance[v]=new;parent[v]=u;heapq.heappush(queue,(new,v))
                raise direct
            for home in self.s.cfg['motion']['safe_configurations']:
                home=np.array(home)
                try:
                    segments=[self.plan_joint_motion(start,home,T_G_O,gripper,mode),self.plan_joint_motion(home,goal,T_G_O,gripper,mode)]
                    if self.s.cfg['motion'].get('blend_joint_waypoints',False):
                        try:return [self.plan_spline([start,home,goal],T_G_O,gripper,mode)]
                        except Failure:pass
                    return segments
                except Failure:continue
            raise direct


class RegraspPlanner:
    def __init__(self,sim,motion):
        self.s=sim;self.motion=motion;self.k=motion.k

    def pick_solutions(self,cube_pose,grasps,seed,exclude_axis=None):
        """Enumerate 24 equivalent cube orientations, preserving physical face labels.

        The attachment hypothesis rotates with each candidate. IK must actually
        achieve that pose; no assumption of a general six-dimensional tool pose.
        """
        from .geometry import GraspCandidate
        s=self.s;cfg=s.cfg['planning'];rng=np.random.default_rng(s.cfg['seed'])
        starts=[seed,np.array(s.cfg['robot']['home'])]
        starts.extend(rng.uniform(s.limits[:,0],s.limits[:,1],(cfg['ik_starts']-2,5)))
        self.search_stats=Counter()
        library=getattr(self,'library',None);key='pick_'+str(exclude_axis)
        hint=library.get(key) if library is not None else None
        if hint:
            try:
                grasp=GraspCandidate(hint['id'],np.array(hint['relative']),hint['gripper'],hint['axis'],hint['opening'],hint['clearance'])
                target=cube_pose@inverse(grasp.T_G_O)
                q=np.array(hint['q']);achieved=self.k.forward(q)
                if np.linalg.norm(achieved[:3,3]-target[:3,3])>=.0001 or np.linalg.norm(Rotation.from_matrix(achieved[:3,:3].T@target[:3,:3]).as_rotvec())>=.001:
                    q=self.k.solve_ik(target,q)
                bad,_=self.motion.cc.check(s.query_state(q,gripper=s.cfg['robot']['open'],cube_pose=cube_pose),'GRASP')
                if bad:raise Failure('COLLISION_PATH')
                pre=target.copy();pre[:3,3]+=target[:3,2]*cfg['pregrasp_distance']+target[:3,0]*cfg['pregrasp_lateral']
                pq=np.array(hint['pre']);achieved=self.k.forward(pq)
                if np.linalg.norm(achieved[:3,3]-pre[:3,3])>=.0001 or np.linalg.norm(Rotation.from_matrix(achieved[:3,:3].T@pre[:3,:3]).as_rotvec())>=.001:
                    pq=self.k.solve_ik(pre,pq)
                approach=self.motion.cartesian(pq,target,gripper=s.cfg['robot']['open'],mode='GRASP')
                self.motion.via_home(seed,pq,gripper=s.cfg['robot']['open'],mode='TABLE')
                self.search_stats['library_hit']+=1
                yield grasp,pq,q,approach
                return
            except Failure as error:
                self.search_stats['library_rejected']+=1
                if not library.prepare:raise Failure('LIBRARY_GRASP_UNREACHABLE',{'reason':error.code})
        bases=[g for g in grasps if g.contact_axis==0]
        if exclude_axis is not None:
            bases.sort(key=lambda g:np.linalg.norm(Rotation.from_matrix(g.T_G_O[:3,:3]).as_rotvec()))
        for base in bases:
            for idx,R in enumerate(stable_cube_rotations()):
                rel=base.T_G_O@transform(R)
                axis=int(np.argmax(np.abs(rel[0,:3])))
                if axis==exclude_axis:continue
                target=cube_pose@inverse(rel)
                for initial in starts:
                    self.search_stats['ik_attempts']+=1
                    try:
                        q=self.k.solve_ik(target,initial)
                        d=s.query_state(q,gripper=s.cfg['robot']['open'],cube_pose=cube_pose)
                        bad,_=self.motion.cc.check(d,'GRASP')
                        if bad:self.search_stats['grasp_collision']+=1;continue
                        pre=target.copy();pre[:3,3]+=target[:3,2]*cfg['pregrasp_distance']+target[:3,0]*cfg['pregrasp_lateral']
                        pq=self.k.solve_ik(pre,q)
                        approach=self.motion.cartesian(pq,target,gripper=s.cfg['robot']['open'],mode='GRASP')
                        self.motion.via_home(seed,pq,gripper=s.cfg['robot']['open'],mode='TABLE')
                        grasp=GraspCandidate(f'{base.grasp_id}_rotation_{idx}',rel,base.gripper,axis,base.required_opening,base.clearance)
                        if library is not None:library.put(key,{'id':grasp.grasp_id,'relative':rel,'gripper':grasp.gripper,'axis':axis,'opening':grasp.required_opening,'clearance':grasp.clearance,'q':q,'pre':pq})
                        yield grasp,pq,q,approach
                        break
                    except Failure as e:self.search_stats[e.code]+=1;continue

    def placement(self,q,relative,return_path=None,flip=False):
        s=self.s; current=s.tcp_pose(s.query_state(q,relative));cube=current@relative
        if flip:
            return self.flip_placement(q,relative)
        # A verified lift path can be traversed backwards to its original support
        # pose. Every return segment is rechecked with the measured attachment.
        if return_path:
            try:
                liftq=return_path[-1].goal
                route=self.motion.via_home(q,liftq,relative,mode='HELD')
                lower=[self.motion.reverse(t,relative,mode='PLACE') for t in reversed(return_path)]
                d=s.query_state(return_path[0].start,relative)
                return s.cube_pose(d),liftq,route,lower
            except Failure:pass
        # Try all six support faces at the configured placement position.
        errors=[]
        for R in sorted(stable_cube_rotations(),key=lambda R:np.linalg.norm(R-cube[:3,:3])):
            pose=transform(R,np.array(s.cfg['scene']['cube_position']))
            target=pose@inverse(relative);pre=target.copy();pre[2,3]+=s.cfg['planning']['lift_distance']
            try:
                preq=self.k.solve_ik(pre,q)
                route=self.motion.via_home(q,preq,relative,mode='HELD')
                lower=self.motion.cartesian(preq,target,relative,mode='PLACE')
                return pose,preq,route,lower
            except Failure as e:errors.append(e.code)
        raise Failure('FLIP_KINEMATICALLY_INFEASIBLE',{'attempts':len(errors),'reasons':dict(Counter(errors))})

    def flip_placement(self,current_q,relative):
        """Place on a different physical face, leaving yaw free for the 5-DOF arm.

        Cube centre (3 constraints) + support normal (2 independent constraints)
        replace an overconstrained arbitrary 6-D tool pose. The object rotates
        only when the executor moves the arm; no live object pose is assigned.
        """
        s=self.s;cfg=s.cfg['regrasp'];g=s.state().gripper_position
        rng=np.random.default_rng(s.cfg['seed']);failures=Counter()
        starts=[current_q,np.array(s.cfg['robot']['home']),*rng.uniform(s.limits[:,0],s.limits[:,1],(cfg['ik_starts'],5))]
        library=getattr(self,'library',None)
        hint=library.get('flip_support') if library is not None else None
        if hint:starts.insert(0,np.array(hint['q']))
        positions=cfg['placement_positions'];faces=cfg['support_faces']
        if hint and library.enabled and not library.prepare:
            starts=[np.array(hint['q'])];positions=[hint['position']];faces=[hint['face']]
        initial=s.cube_pose()
        for pos in positions:
            for face in faces:
                if face//2==getattr(self,'support_axis_before_pick',2):continue
                n=np.zeros(3);n[face//2]=1 if face%2==0 else -1
                def residual(q):
                    T=s.cube_pose(s.query_state(q,relative,gripper=g))
                    return np.r_[10*(T[:3,3]-pos),T[:3,:3]@n-[0,0,-1]]
                for seed in starts:
                    fit=least_squares(residual,np.clip(seed,s.limits[:,0]+1e-7,s.limits[:,1]-1e-7),bounds=(s.limits[:,0]+1e-8,s.limits[:,1]-1e-8),max_nfev=s.cfg['planning']['ik_max_evaluations'])
                    if np.linalg.norm(residual(fit.x))>cfg['support_residual_tolerance']:
                        failures['support_ik']+=1;continue
                    d=s.query_state(fit.x,relative,gripper=g);pose=s.cube_pose(d)
                    bad,_=self.motion.cc.check(d,'PLACE')
                    if bad:failures['place_collision']+=1;continue
                    try:
                        raised=s.tcp_pose(d);raised[2,3]+=s.cfg['planning']['lift_distance']
                        lift=self.motion.cartesian(fit.x,raised,relative,g,'PLACE',orientation=False)
                        upper=lift[-1].goal
                        route=self.motion.via_home(current_q,upper,relative,g,'HELD')
                        lower=[self.motion.reverse(t,relative,g,'PLACE') for t in reversed(lift)]
                        self._placement_support_face=face
                        if library is not None:library.put('flip_support',{'q':fit.x,'face':face,'position':pos})
                        return pose,upper,route,lower
                    except Failure as e:failures[e.code]+=1
        raise Failure('FLIP_KINEMATICALLY_INFEASIBLE',{'search_reasons':dict(failures)})

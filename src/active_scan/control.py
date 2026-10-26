"""Measured-state execution, bounded settling and fail-closed task state machine."""
from dataclasses import asdict
from pathlib import Path
import json
import platform
import sys
import time
import importlib.metadata
import hashlib
import numpy as np
import mujoco as mj
from PIL import Image
from scipy.spatial.transform import Rotation
from .core import ROOT,Failure,inverse,write_json
from .geometry import CollisionChecker,VisibilityEvaluator,GraspPlanner
from .planning import MotionPlanner,ViewPlanner,RegraspPlanner,BrakeTrajectory
from .library import PlanningLibrary
from .grasp import verify_test_lift


class EventLog:
    def __init__(self,out,sim):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True);self.sim=sim
        if (self.out/'events.jsonl').exists():
            raise FileExistsError(f'Existing episode preserved: {self.out}. Choose a new --out directory.')
        self.events=open(self.out/'events.jsonl','w',encoding='utf-8')
        paths=list((ROOT/'src').rglob('*.py'))+list((ROOT/'configs').glob('*.yaml'))
        write_json(self.out/'implementation.json',{'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}})
        self.samples=[];self.contact_samples=[];self.trajectories=[];self.force_samples=[];self.dynamics_samples=[]

    def event(self,kind,**payload):
        obj={'time':float(self.sim.data.time),'kind':kind,**payload}
        self.events.write(json.dumps(obj,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x))+'\n');self.events.flush()

    def close(self):
        self.events.close()
        write_json(self.out/'trajectories.json',self.trajectories)
        write_json(self.out/'contacts.json',self.contact_samples)
        if self.samples:
            np.savez_compressed(self.out/'tracking.npz',samples=np.array(self.samples))
        if self.force_samples:np.savez_compressed(self.out/'actuator_forces.npz',samples=np.array(self.force_samples))
        if self.dynamics_samples:np.savez_compressed(self.out/'joint_dynamics.npz',samples=np.array(self.dynamics_samples))


class Executor:
    def __init__(self,sim,checker,log):
        self.s,self.cc,self.log=sim,checker,log
        self.errors=[];self.length=0.;self.moves=0;self.forbidden=0;self.settling=[]
        self.expected_relative=None
        self.settled_time=None;self.settled_goal=None
        self.velocity_max=np.zeros(5);self.acceleration_max=np.zeros(5)
        self.force_max=np.zeros(sim.model.nu);self.saturated=np.zeros(sim.model.nu);self.ticks=0
        self.mode='TABLE';self._braking=False;self.stop_result=None

    def tick(self,mode,reference=None):
        self.mode=mode
        s=self.s;before=s.data.qvel[s.vadr].copy();s.step()
        speed=s.data.qvel[s.vadr].copy();acc=(speed-before)/s.model.opt.timestep
        forces=s.data.actuator_force.copy()
        self.velocity_max=np.maximum(self.velocity_max,np.abs(speed));self.acceleration_max=np.maximum(self.acceleration_max,np.abs(acc))
        self.force_max=np.maximum(self.force_max,np.abs(forces));self.ticks+=1
        self.saturated+=np.abs(forces)>=np.max(np.abs(s.model.actuator_forcerange),axis=1)*.999
        self.log.dynamics_samples.append(np.r_[s.data.time,speed,acc,forces])
        if not self._braking and s.level!='A' and s.cfg['motion'].get('enforce_measured_velocity',False) and np.any(np.abs(speed)>np.asarray(s.cfg['motion']['velocity'])+1e-6):
            raise Failure('MEASURED_SPEED_LIMIT',{'measured':speed,'limits':s.cfg['motion']['velocity']})
        bad,_=self.cc.check(mode=mode)
        if bad:
            self.forbidden+=1;raise Failure('COLLISION_EXECUTION',{'contacts':bad})
        if not self._braking and mode=='HELD' and self.expected_relative is not None:
            error=inverse(self.expected_relative)@(inverse(s.tcp_pose())@s.cube_pose())
            shift=float(np.linalg.norm(error[:3,3]));angle=float(np.linalg.norm(Rotation.from_matrix(error[:3,:3]).as_rotvec()))
            if shift>s.cfg['physics']['grip_translation_tolerance'] or angle>s.cfg['physics']['grip_rotation_tolerance']:
                raise Failure('GRASP_SLIP',{'translation':shift,'rotation':angle})
        # Record actual contact forces rather than interpreting close commands as grasp success.
        for i in range(s.data.ncon):
            contact=s.data.contact[i];force=np.zeros(6);mj.mj_contactForce(s.model,s.data,i,force)
            self.log.contact_samples.append({'time':float(s.data.time),'geoms':[s.model.geom(int(contact.geom1)).name,s.model.geom(int(contact.geom2)).name],'distance':float(contact.dist),'force_contact_frame':force.tolist()})
        if reference is not None:
            e=reference-s.data.qpos[s.qadr];self.errors.extend(e)
            self.log.samples.append(np.r_[s.data.time,reference,s.data.qpos[s.qadr],s.data.qvel[s.vadr],s.cube_pose()[:3,3]])
            self.log.force_samples.append(np.r_[s.data.time,s.data.actuator_force.copy()])
            if np.max(np.abs(e))>s.cfg['motion']['tracking_abort']:
                raise Failure('TRACKING_ERROR',{'max_error':float(np.max(np.abs(e)))})

    def wait(self,seconds,mode):
        for _ in range(int(np.ceil(seconds/self.s.model.opt.timestep))):self.tick(mode)

    def controlled_stop(self,reason):
        s=self.s
        if not s.cfg['motion'].get('controlled_stop',False):s.stop();return
        start=s.data.time;initial=s.state().dq_arm
        if np.max(np.abs(initial))<.01:
            s.stop();self.stop_result={'method':'already_stationary','duration':0.,'final_velocity':initial}
            self.log.event('stop_result',trigger=reason,**self.stop_result);return
        try:
            tr=BrakeTrajectory(s.state().q_arm,initial,s.cfg['motion']['stop_deceleration'])
            if np.any(tr.goal<s.limits[:,0]) or np.any(tr.goal>s.limits[:,1]):raise Failure('BRAKE_JOINT_LIMIT')
            relative=s.T_G_O if reason!='GRASP_SLIP' else None
            count=max(5,int(np.ceil(np.max(np.abs(tr.goal-tr.start))/.0025))+1)
            for t in np.linspace(0,tr.duration,count):
                bad,_=self.cc.check(s.query_state(tr.sample(t)[0],relative),self.mode)
                if bad:raise Failure('BRAKE_PATH_BLOCKED')
            self._braking=True
            for i in range(int(np.ceil(tr.duration/s.model.opt.timestep))):
                q,v,_=tr.sample((i+1)*s.model.opt.timestep);s.set_joint_reference(q,v);self.tick(self.mode,q)
            s.set_joint_targets(tr.goal)
            deadline=s.data.time+.3
            while np.max(np.abs(s.state().dq_arm))>.05 and s.data.time<deadline:self.tick(self.mode,tr.goal)
            if np.max(np.abs(s.state().dq_arm))>.05:raise Failure('BRAKE_NOT_SETTLED')
            self.stop_result={'method':'controlled_brake','duration':float(s.data.time-start),'initial_velocity':initial,'final_velocity':s.state().dq_arm}
        except Failure as failure:
            s.stop();self.stop_result={'method':'hold_fallback','reason':failure.code,'duration':float(s.data.time-start)}
        finally:
            self._braking=False
        self.log.event('stop_result',trigger=reason,**self.stop_result)

    def gripper(self,target,mode):
        s=self.s;cfg=s.cfg['motion'];s.set_gripper_target(target)
        contact_close=s.level=='C' and mode=='GRASP' and cfg.get('contact_close',False)
        if not cfg.get('adaptive_waits',False) or (s.level=='C' and not contact_close):
            self.wait(cfg['gripper_duration'],mode);return
        start=s.data.time;stable=0.
        while s.data.time-start<cfg.get('gripper_timeout',cfg['gripper_duration']):
            self.tick(mode)
            stopped=abs(s.data.qvel[s.gv])<cfg['gripper_speed_tolerance']
            reached=abs(s.data.qpos[s.gq]-target)<cfg['gripper_position_tolerance']
            contact_bodies=set()
            for c in s.data.contact:
                if s.cubegeom in (c.geom1,c.geom2):
                    other=c.geom2 if c.geom1==s.cubegeom else c.geom1
                    contact_bodies.add(int(s.model.geom_bodyid[other]))
            closed_on_object=mode=='GRASP' and s.grip in contact_bodies and s.moving in contact_bodies
            feedback=s.cfg.get('grasp',{}).get('closure_feedback','simulator_contacts')
            blocked=not reached and s.data.qpos[s.gq]>target+cfg['gripper_position_tolerance'] and s.data.qpos[s.gq]<s.cfg['robot']['open']-.03
            closure=(blocked if feedback=='position_stall' else closed_on_object)
            okay=stopped and (closure if contact_close else (reached or closed_on_object))
            stable=stable+s.model.opt.timestep if okay else 0.
            if stable>=cfg['gripper_stable_time']:
                if contact_close:self.log.event('closure_feedback_verified',feedback=feedback,gripper=float(s.data.qpos[s.gq]),simulated_bilateral_contact=closed_on_object)
                return
        raise Failure('GRIPPER_TIMEOUT',{'target':target,'measured':float(s.data.qpos[s.gq]),'speed':float(s.data.qvel[s.gv]),'contact_bodies':sorted(contact_bodies)})

    def execute(self,trajectories,mode='HELD'):
        s=self.s;dt=s.model.opt.timestep;ratio=max(1,round(s.cfg['motion']['control_dt']/dt))
        for tr in trajectories:
            initial_offset=s.command_reference.copy()-tr.start
            self.log.trajectories.append({'start':tr.start,'goal':tr.goal,'duration':tr.duration,'mode':mode,
                                          'waypoints':getattr(tr,'waypoints',None),'kind':type(tr).__name__,
                                          'initial_command_offset':initial_offset})
            self.moves+=1;self.length+=float(np.linalg.norm(tr.goal-tr.start))
            ref=tr.start
            for i in range(int(np.ceil(tr.duration/dt))):
                if s.cfg['motion'].get('continuous_reference',False):
                    # Low-level setpoint interpolation between outer-loop updates.
                    # Dynamics, force limits and measured states remain unchanged.
                    elapsed=min((i+1)*dt,tr.duration)
                    ref,velocity,_=tr.sample(elapsed)
                    reconcile=s.cfg['motion'].get('reference_reconcile_time',0.)
                    if reconcile:
                        # The measured start differs from the previous held command
                        # under gravity. Remove that offset smoothly, without a
                        # one-step servo target jump at every new trajectory.
                        u=min(elapsed/min(reconcile,tr.duration),1.)
                        ref=ref+initial_offset*(1-10*u**3+15*u**4-6*u**5)
                    s.set_joint_reference(ref,velocity)
                elif i%ratio==0:
                    ref=tr.sample(min((i+ratio)*dt,tr.duration))[0];s.set_joint_targets(ref)
                self.tick(mode,ref)
            s.set_joint_targets(tr.goal)
        if trajectories:self.settle(trajectories[-1].goal,mode)

    def settle(self,goal,mode):
        s=self.s;cfg=s.cfg['motion'];start=s.data.time;stable=0.
        if self.settled_time==s.data.time and self.settled_goal is not None and np.max(np.abs(self.settled_goal-goal))<1e-9:return
        while s.data.time-start<cfg['settle_timeout']:
            self.tick(mode,goal)
            state=s.state()
            okay=(np.max(np.abs(state.q_arm-goal))<cfg['tracking_tolerance'] and np.max(np.abs(state.dq_arm))<cfg['speed_tolerance'] and np.linalg.norm(state.cube_velocity[:3])<cfg['cube_speed_tolerance'] and np.linalg.norm(state.cube_velocity[3:])<cfg['cube_angular_speed_tolerance'])
            stable=stable+s.model.opt.timestep if okay else 0.
            if stable>=cfg['settle_time']:
                self.settling.append(float(s.data.time-start));self.settled_time=s.data.time;self.settled_goal=np.array(goal).copy();return
        raise Failure('SETTLE_TIMEOUT',{'joint_error':float(np.max(np.abs(s.state().q_arm-goal))),'cube_velocity':s.state().cube_velocity})


class TaskPlanner:
    """One FSM for A/B/C. Failed operations retain all successful earlier evidence."""
    def __init__(self,sim,out):
        self.s=sim;self.log=EventLog(out,sim);self.cc=CollisionChecker(sim);self.v=VisibilityEvaluator(sim)
        self.motion=MotionPlanner(sim,self.cc);self.views=ViewPlanner(sim,self.cc,self.v)
        self.regrasp=RegraspPlanner(sim,self.motion);self.gp=GraspPlanner(sim,self.cc)
        self.library=PlanningLibrary(sim);self.regrasp.library=self.library;self.motion.library=self.library
        self.executor=Executor(sim,self.cc,self.log);self.coverage=self.v.coverage()
        self.fsm='INITIALIZE';self.captures=0;self.regrasps=0;self.stages={};self.selected_grasps=[]
        self.clock=time.perf_counter();self.planning_seconds=0.
        self.planning_breakdown={}
        self.planning_calls=[]

    def plan(self,kind,operation):
        start=time.perf_counter()
        try:return operation()
        finally:
            elapsed=time.perf_counter()-start;self.planning_seconds+=elapsed
            self.planning_breakdown[kind]=self.planning_breakdown.get(kind,0.)+elapsed
            self.planning_calls.append({'kind':kind,'seconds':elapsed,'simulation_time':float(self.s.data.time)})

    def transition(self,state,**details):
        self.log.event('transition',source=self.fsm,target=state,**details);self.fsm=state

    def move(self,target,mode='HELD'):
        s=self.s
        plans=self.plan('joint_paths',lambda:self.motion.via_home(s.state().q_arm,target,s.T_G_O,s.state().gripper_position,mode))
        self.executor.execute(plans,mode)

    def pick(self,grasps,repeated=False):
        s=self.s;self.transition('PLAN_REGRASP' if repeated else 'PLAN_TASK')
        cube=s.cube_pose();seed=s.state().q_arm
        self.table_pose_before_pick=cube.copy()
        self.regrasp.support_axis_before_pick=int(np.argmax(np.abs(cube[2,:3])))
        previous_axis=self.selected_grasps[-1]['contact_axis'] if self.selected_grasps else None
        solutions=self.regrasp.pick_solutions(cube,grasps,seed,exclude_axis=previous_axis if repeated else None)
        solution=self.plan('grasp_search',lambda:next(solutions,None))
        self.log.event('grasp_search',counts=dict(self.regrasp.search_stats))
        if solution is None:raise Failure('SECOND_GRASP_UNREACHABLE' if repeated else 'INITIAL_GRASP_UNREACHABLE',dict(self.regrasp.search_stats))
        grasp,pre,q,approach=solution
        self.approach_path=approach
        self.selected_grasps.append({'grasp_id':grasp.grasp_id,'contact_axis':grasp.contact_axis,'predicted_T_G_O':grasp.T_G_O,'pregrasp_q':pre,'grasp_q':q})
        self.log.event('selected_grasp',**self.selected_grasps[-1])
        self.transition('MOVE_PREGRASP');self.move(pre,'TABLE')
        self.transition('APPROACH');self.executor.execute(approach,'GRASP')
        self.transition('CLOSE_GRIPPER')
        self.executor.gripper(s.cfg['robot']['closed'] if s.level=='C' else grasp.gripper,'GRASP')
        # Proximity is only a precondition for the explicit ideal attachment.
        position_error=np.linalg.norm(s.tcp_pose()[:3,3]-s.cube_pose()[:3,3])
        if position_error>s.cfg['physics']['grasp_alignment_tolerance']:raise Failure('GRASP_ALIGNMENT_FAILED',{'position_error':position_error})
        s.attach(grasp.grasp_id)
        self.transition('VERIFY_GRASP')
        baseline=s.cube_pose().copy();relative=inverse(s.tcp_pose())@baseline
        probe_path=[]
        if s.level=='C' and s.cfg.get('grasp',{}).get('test_lift',False):
            before=s.observe_object();tool_before=s.tcp_pose()
            test_target=tool_before.copy();test_target[2,3]+=s.cfg['grasp']['test_lift_distance']
            probe_path=self.plan('test_lift_path',lambda:self.motion.cartesian(s.state().q_arm,test_target,s.T_G_O,s.state().gripper_position,'GRASP',orientation=True))
            self.executor.execute(probe_path,'GRASP')
            self.executor.wait(s.cfg['grasp']['test_hold_seconds'],'GRASP')
            evidence=verify_test_lift(before,s.observe_object(),tool_before,s.tcp_pose(),s.cfg['grasp']['test_min_lift'],s.cfg['grasp']['test_max_slip'])
            self.log.event('test_lift_verified',**evidence)
            self.executor.expected_relative=inverse(s.tcp_pose())@s.cube_pose()
        self.transition('LIFT')
        target=s.tcp_pose();target[2,3]+=s.cfg['planning']['lift_distance']-(s.cfg['grasp']['test_lift_distance'] if probe_path else 0.)
        trajectory=self.plan('cartesian_paths',lambda:self.motion.cartesian(s.state().q_arm,target,s.T_G_O,s.state().gripper_position,'PLACE',orientation=False))
        self.executor.execute(trajectory,'PLACE')
        self.return_path=probe_path+trajectory
        hold=s.cfg['motion'].get('ideal_hold_duration',s.cfg['physics']['hold_duration']) if s.level!='C' else s.cfg['physics']['hold_duration']
        self.executor.wait(hold,'HELD')
        moved=s.cube_pose()[2,3]-baseline[2,3]
        slip=inverse(relative)@(inverse(s.tcp_pose())@s.cube_pose())
        ang=np.linalg.norm(Rotation.from_matrix(slip[:3,:3]).as_rotvec())
        contacts=self.cc.grasp_intersections(s.data)
        finger_bodies={int(s.model.geom(k).bodyid[0]) for k,distance in contacts.items() if distance<s.cfg['physics']['contact_distance_tolerance']}
        valid=bool(moved>=s.cfg['physics']['min_lift'] and np.linalg.norm(slip[:3,3])<s.cfg['physics']['grip_translation_tolerance'] and ang<s.cfg['physics']['grip_rotation_tolerance'])
        if s.level=='C':valid=bool(valid and s.grip in finger_bodies and s.moving in finger_bodies and not s.data.eq_active.any())
        self.log.event('grasp_verification',lift=moved,slip_translation=slip[:3,3],slip_rotation=ang,bilateral_contact=(len(finger_bodies)==2),success=valid)
        if not valid:raise Failure('PHYSICAL_GRASP_UNSTABLE' if s.level=='C' else 'IDEAL_GRASP_FAILED')
        self.executor.expected_relative=inverse(s.tcp_pose())@s.cube_pose()
        self.stages['regrasp' if repeated else 'pick_and_lift']=True
        if repeated:self.regrasps+=1
        return grasp

    def capture(self,grasp):
        s=self.s;self.transition('SETTLE');self.executor.settle(s.command_reference.copy(),'HELD')
        self.transition('CAPTURE');state=s.state();rgb,depth,seg=s.capture()
        quality,directions,occ=self.v.evaluate()
        frame_quality=self.v.frame_metrics(evaluation=(quality,directions,occ),segmentation=seg)
        rejections=self.v.frame_rejections(frame_quality)
        if rejections:
            self.log.event('capture_rejected',reasons=rejections,frame_quality=frame_quality)
            raise Failure('FRAME_QUALITY_REJECTED',{'reasons':rejections,'frame_quality':frame_quality})
        if not np.any(quality>=s.cfg['visibility']['quality_threshold']):raise Failure('INSUFFICIENT_VISIBILITY')
        self.coverage.update(quality,directions)
        prefix=self.log.out/f'frame_{self.captures:03d}'
        Image.fromarray(rgb).save(str(prefix)+'.png')
        np.savez_compressed(str(prefix)+'.npz',depth=depth,segmentation=seg,quality=quality,occluded=occ,directions_object=directions)
        write_json(str(prefix)+'.json',{'timestamp':state.timestamp,'simulation_level':s.level,'grasp_id':grasp.grasp_id,'q':state.q_arm,'dq_arm':state.dq_arm,'cube_velocity':state.cube_velocity,'T_W_C':state.T_world_camera,'T_W_O':state.T_world_cube,'T_C_O':inverse(state.T_world_camera)@state.T_world_cube,'T_W_G':state.T_world_gripper,'K':s.K,'frame_quality':frame_quality,'coverage':self.coverage.metrics(self.v.faces)})
        self.captures+=1;self.transition('UPDATE_COVERAGE',coverage=self.coverage.metrics(self.v.faces)['binary_coverage'])
        self.last_frame_quality=frame_quality

    def observe(self,grasp):
        s=self.s;self.transition('PLAN_VIEW_SEQUENCE')
        # Recompute the actual attachment after measured pick, including IK/tracking offsets.
        from .geometry import GraspCandidate
        g=GraspCandidate(grasp.grasp_id,inverse(s.tcp_pose())@s.cube_pose(),s.state().gripper_position,grasp.contact_axis,grasp.required_opening,grasp.clearance)
        candidates,stats=self.plan('view_search',lambda:self.library.views(g,self.views))
        name=f'view_pool_{len(self.selected_grasps)}'
        np.savez_compressed(self.log.out/f'{name}.npz',q=np.array([c.q for c in candidates]),quality=np.array([c.quality for c in candidates]))
        potential=np.max([c.quality for c in candidates],axis=0) if candidates else np.zeros(len(self.v.local))
        self.log.event('view_pool_potential',binary=float(np.mean(potential>=s.cfg['visibility']['quality_threshold'])),per_face={str(i):float(np.mean(potential[self.v.faces==i]>=s.cfg['visibility']['quality_threshold'])) for i in range(6)})
        priority=None;deferred_face=None
        if len(self.selected_grasps)==1 and s.cfg.get('regrasp',{}).get('flip_before_second_grasp',False):
            normal=g.T_G_O[2,:3];axis=int(np.argmax(np.abs(normal)));deferred_face=2*axis+(0 if normal[axis]>0 else 1)
            priority=(self.v.faces!=deferred_face).astype(float)
        selected=self.plan('view_selection',lambda:self.views.order_views(self.views.select_views(candidates,self.coverage.quality,s.state().q_arm,priority=priority),s.state().q_arm))
        if stats.get('library_hit'):
            # Nominal library scores never count as captured coverage.
            checked=[]
            for candidate in selected:
                actual=self.plan('view_validation',lambda:self.views.validate_candidate(candidate.q,g))
                if actual is not None:checked.append(actual)
            selected=checked
        self.log.event('view_plan',statistics=stats,targets=[c.q for c in selected],deferred_until_flip_face=deferred_face)
        executed=0;errors=[]
        for candidate in selected:
            try:
                self.transition('MOVE_TO_VIEW');self.move(candidate.q,'HELD');self.capture(g);executed+=1
            except Failure as e:
                if e.code in ('COLLISION_PATH','FRAME_QUALITY_REJECTED'):
                    self.log.event('view_path_rejected',reason=e.code,details=e.details);errors.append(e.code);continue
                raise
        if not executed:raise Failure('NO_EXECUTABLE_VIEWS',{'reasons':errors})
        self.stages['view_sequence']=True

    def place(self,flip=False):
        s=self.s;self.transition('MOVE_PREPLACE')
        before=s.cube_pose().copy()
        pose,preq,route,lower=self.plan('placement_search',lambda:self.regrasp.placement(s.state().q_arm,s.T_G_O,getattr(self,'return_path',None),flip=flip))
        self.log.event('placement_plan',target_cube_pose=pose,flip=flip)
        self.executor.execute(route,'HELD');self.transition('REORIENT')
        self.transition('PLACE');self.executor.execute(lower,'PLACE')
        support_distance=mj.mj_geomDistance(s.model,s.data,s.cubegeom,s.table,.10,None)
        if support_distance>s.cfg['physics']['placement_height_tolerance']:
            raise Failure('PLACEMENT_SUPPORT_MISSING',{'table_gap':support_distance})
        self.transition('RELEASE');s.release();self.executor.expected_relative=None
        if s.cfg['motion'].get('adaptive_waits',False):
            self.executor.gripper(s.cfg['robot']['open'],'PLACE');self.executor.settle(s.state().q_arm,'PLACE')
        else:
            s.set_gripper_target(s.cfg['robot']['open']);self.executor.wait(s.cfg['motion']['placement_timeout'],'PLACE')
        self.transition('VERIFY_PLACEMENT')
        current=s.cube_pose();vel=s.state().cube_velocity
        support=np.max(np.abs(current[2,:3]))
        valid=(abs(current[2,3]-s.cfg['scene']['cube_size']/2)<s.cfg['physics']['placement_height_tolerance'] and np.arccos(np.clip(support,-1,1))<s.cfg['physics']['placement_angle_tolerance'] and np.linalg.norm(vel[:3])<s.cfg['motion']['cube_speed_tolerance'] and np.linalg.norm(vel[3:])<s.cfg['motion']['cube_angular_speed_tolerance'])
        if not valid:raise Failure('UNSTABLE_PLACEMENT',{'pose':current,'velocity':vel})
        if flip:
            old_down=int(np.argmax(np.abs(self.table_pose_before_pick[2,:3])))
            new_down=int(np.argmax(np.abs(current[2,:3])))
            if old_down==new_down:raise Failure('FLIP_SUPPORT_UNCHANGED')
            self.stages['flip']=True
            self.log.event('flip_verified',before=self.table_pose_before_pick,after=current,old_support_axis=old_down,new_support_axis=new_down)
        self.stages['placement']=True;self.transition('RETREAT')
        plans=[];start=s.state().q_arm
        if flip:
            target=s.tcp_pose();target[2,3]+=s.cfg['regrasp']['retreat_height']
            plans=self.plan('retreat_paths',lambda:self.motion.cartesian(start,target,gripper=s.cfg['robot']['open'],mode='GRASP',orientation=False))
            self.executor.execute(plans,'GRASP');return
        for previous in reversed(self.approach_path):
            if np.max(np.abs(start-previous.goal))<s.cfg['motion']['tracking_tolerance']:
                plans.append(self.motion.reverse(previous,gripper=s.cfg['robot']['open'],mode='GRASP'))
            else:plans.append(self.motion.plan_joint_motion(start,previous.start,gripper=s.cfg['robot']['open'],mode='GRASP'))
            start=previous.start
        self.executor.execute(plans,'GRASP')

    def run(self,force_regrasp=True,pick_only=False):
        s=self.s;success=False;failure=None
        write_json(self.log.out/'config.json',s.cfg);write_json(self.log.out/'initial_state.json',asdict(s.state()))
        write_json(self.log.out/'versions.json',{'python':sys.version,'platform':platform.platform(),'packages':{p:importlib.metadata.version(p) for p in ('mujoco','numpy','scipy','PyYAML','Pillow')},'model':json.loads((Path(__file__).resolve().parents[2]/'third_party/robotstudio_so101/SOURCE.json').read_text())})
        try:
            if self.library.enabled and not self.library.prepare:self.library.get('READY')
            self.transition('READ_STATE');grasps,fit=self.plan('grasp_geometry',lambda:self.library.grasps(self.gp));self.log.event('gripper_compatibility',audit=fit)
            if not grasps:raise Failure('GRIPPER_OPENING_INSUFFICIENT',fit)
            first=self.pick(grasps)
            if not pick_only:
                self.observe(first)
                needs_second=force_regrasp or self.coverage.metrics(self.v.faces)['binary_coverage']<s.cfg['planning']['target_coverage']
                self.place(flip=needs_second and s.cfg.get('regrasp',{}).get('flip_before_second_grasp',False))
                if needs_second:
                    second=self.pick(grasps,True);self.observe(second);self.place()
                self.transition('FINISH');self.move(np.array(s.cfg['robot']['home']),'TABLE')
                if self.coverage.metrics(self.v.faces)['binary_coverage']<s.cfg['planning']['target_coverage']:
                    raise Failure('COVERAGE_TARGET_UNREACHABLE')
            success=True;self.transition('DONE')
            if self.library.enabled and self.library.prepare:self.library.put('READY',{'simulation_level':s.level,'captures':self.captures,'coverage':self.coverage.metrics(self.v.faces)})
        except Failure as e:
            self.executor.controlled_stop(e.code);failure={'code':e.code,'details':e.details};self.transition('RECOVERY',failure=failure)
            self.log.event('recovery_state',q=s.state().q_arm,cube_pose=s.cube_pose())
            self.transition('FAILED')
        finally:
            errors=np.array(self.executor.errors)
            result={'success':success,'failure':failure,'fsm':self.fsm,'simulation_level':s.level,'task_scope':'pick_and_hold' if pick_only else 'full_episode','stages':self.stages,
                    'captures':self.captures,'regrasps':self.regrasps,'coverage':self.coverage.metrics(self.v.faces),
                    'simulated_seconds':float(s.data.time),'wall_seconds':time.perf_counter()-self.clock,'planning_seconds':self.planning_seconds,
                    'planning_breakdown':self.planning_breakdown,'sim_plus_planning_seconds':float(s.data.time)+self.planning_seconds,
                    'planning_calls':self.planning_calls,'max_planning_call_seconds':max((c['seconds'] for c in self.planning_calls),default=0.),
                    'planning_library':self.library.stats(),
                    'motion_segments':self.executor.moves,'joint_path_length':self.executor.length,
                    'tracking_max':float(np.max(np.abs(errors))) if len(errors) else None,'tracking_rms':float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                    'measured_velocity_max':self.executor.velocity_max,'measured_acceleration_max':self.executor.acceleration_max,
                    'actuator_force_max':self.executor.force_max,'force_saturation_fraction':self.executor.saturated/max(1,self.executor.ticks),
                    'stop_result':self.executor.stop_result,
                    'forbidden_collision_steps':self.executor.forbidden,'settling_seconds':self.executor.settling,
                    'weld_active_final':bool(s.data.eq_active[s.weld]),'selected_grasps':self.selected_grasps}
            write_json(self.log.out/'summary.json',result)
            np.savez_compressed(self.log.out/'coverage.npz',quality=self.coverage.quality,observations=self.coverage.counts,faces=self.v.faces,local_points=self.v.local)
            # Always save a diagnostic frame, clearly distinct from accepted capture frames.
            Image.fromarray(s.capture()[0]).save(self.log.out/'final_diagnostic.png')
            self.log.close();s.close()
        return result

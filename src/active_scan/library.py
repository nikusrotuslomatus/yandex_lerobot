"""Versioned setup-time planning data; every executed view/path is checked live."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .core import ROOT,Failure,write_json


class PlanningLibrary:
    def __init__(self,sim):
        self.s=sim;cfg=sim.cfg.get('library',{})
        self.enabled=cfg.get('enabled',False);self.prepare=cfg.get('prepare',False)
        relevant={k:v for k,v in sim.cfg.items() if k!='library'}
        digest=hashlib.sha256(sim.xml.encode()+json.dumps(relevant,sort_keys=True).encode())
        # Include compiled collision assets, not only XML filenames. Replacing an
        # STL at the same path must also invalidate prior geometric checks.
        for name in ('geom_size','geom_rbound','mesh_vert','mesh_face','body_mass','body_inertia','jnt_range'):
            digest.update(np.asarray(getattr(sim.model,name)).tobytes())
        for name in ('geometry.py','planning.py','library.py'):
            digest.update(Path(__file__).with_name(name).read_bytes())
        self.fingerprint=digest.hexdigest()
        self.root=ROOT/cfg.get('directory','.planning_library')/self.fingerprint
        self.hits=0;self.misses=0;self.build_seconds=0.
        self.certificate_hits=0;self.certificate_rechecks=0

    def path(self,key,suffix='.json'):
        return self.root/(hashlib.sha256(key.encode()).hexdigest()+suffix)

    def get(self,key,required=True):
        if not self.enabled:return None
        path=self.path(key)
        if path.exists():
            self.hits+=1;return json.loads(path.read_text(encoding='utf-8'))
        self.misses+=1
        if required and not self.prepare:raise Failure('LIBRARY_NOT_PREPARED',{'key':key,'fingerprint':self.fingerprint})
        return None

    def put(self,key,value):
        if not self.enabled or not self.prepare:return
        self.root.mkdir(parents=True,exist_ok=True)
        write_json(self.path(key),value)

    def grasps(self,planner):
        if not self.enabled:return planner.geometric_candidates(all_fits=True)
        from .geometry import GraspCandidate
        from dataclasses import asdict
        stored=self.get('geometric_grasps')
        if stored:
            return [GraspCandidate(**{**g,'T_G_O':np.array(g['T_G_O'])}) for g in stored['grasps']],stored['audit']
        start=time.perf_counter();grasps,audit=planner.geometric_candidates(all_fits=True)
        self.build_seconds+=time.perf_counter()-start
        self.put('geometric_grasps',{'grasps':[asdict(g) for g in grasps],'audit':audit})
        return grasps,audit

    def views(self,grasp,planner):
        from .planning import ViewCandidate
        key='views_'+grasp.grasp_id;path=self.path(key,'.npz')
        if self.enabled and path.exists():
            self.hits+=1
            with np.load(path,allow_pickle=False) as values:
                # Nominal quality is used only to shortlist. All shortlisted poses
                # are re-evaluated with the measured attachment before execution.
                rows=[ViewCandidate(q.copy(),None,None,None,quality.copy(),None,0.,0.)
                      for q,quality in zip(values['q'],values['quality'])]
            return rows,{'library_hit':True,'sampled':0,'accepted':len(rows),'fingerprint':self.fingerprint}
        if self.enabled:
            self.misses+=1
            if not self.prepare:raise Failure('LIBRARY_NOT_PREPARED',{'key':key})
        start=time.perf_counter();rows,stats=planner.generate_candidates(grasp)
        self.build_seconds+=time.perf_counter()-start
        if self.enabled:
            self.root.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,q=np.array([r.q for r in rows]),quality=np.array([r.quality for r in rows]))
        return rows,{**stats,'library_hit':False}

    def stats(self):
        return {'enabled':self.enabled,'preparation_mode':self.prepare,'fingerprint':self.fingerprint,
                'hits':self.hits,'misses':self.misses,'build_seconds':self.build_seconds,
                'certificate_hits':self.certificate_hits,'certificate_rechecks':self.certificate_rechecks}

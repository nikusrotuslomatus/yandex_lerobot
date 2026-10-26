import argparse
from .core import load_config,write_json
from .simulation import SimulatorAdapter
from .control import TaskPlanner


def main():
    p=argparse.ArgumentParser(description='SO101 active scan A=geometry, B=ideal weld, C=contact only')
    p.add_argument('--config');p.add_argument('--level',choices=['A','B','C'],default='B');p.add_argument('--out',default='results/episode_B');p.add_argument('--seed',type=int);p.add_argument('--candidates',type=int);p.add_argument('--single-grasp',action='store_true')
    p.add_argument('--pick-only',action='store_true')
    a=p.parse_args();cfg=load_config(a.config)
    if a.seed is not None:cfg['seed']=a.seed
    if a.candidates is not None:cfg['planning']['candidates']=a.candidates
    result=TaskPlanner(SimulatorAdapter(cfg,a.level),a.out).run(force_regrasp=not a.single_grasp,pick_only=a.pick_only)
    print({k:result[k] for k in ('success','failure','captures','coverage','simulation_level')},flush=True)


if __name__=='__main__':main()

"""Build nominal planning data before production, using an explicitly ideal B run."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from active_scan.core import load_config
from active_scan.simulation import SimulatorAdapter
from active_scan.control import TaskPlanner


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/factory.json')
    p.add_argument('--out',required=True);args=p.parse_args()
    cfg=load_config(args.config);cfg.setdefault('library',{}).update(enabled=True,prepare=True)
    result=TaskPlanner(SimulatorAdapter(cfg,'B'),args.out).run()
    print({key:result[key] for key in ['success','failure','planning_seconds','planning_library']})
    if not result['success']:raise SystemExit(1)


if __name__=='__main__':main()

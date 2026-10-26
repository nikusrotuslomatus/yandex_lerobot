"""Check saved results independently of the planner's success flag."""
from pathlib import Path
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from PIL import Image
from active_scan.core import ROOT,write_json,inverse,CoverageState


def validate_episode(path):
    out=Path(path);cfg=json.loads((out/'config.json').read_text());summary=json.loads((out/'summary.json').read_text())
    frames=sorted(out.glob('frame_*.json'));assert len(frames)==summary['captures']
    coverage=CoverageState(6*cfg['visibility']['grid']**2,cfg['visibility']['quality_threshold'],cfg['visibility']['diversity_deg'])
    previous=-1.;camera=None
    for p in frames:
        meta=json.loads(p.read_text());arr=np.load(p.with_suffix('.npz'))
        rgb=Image.open(p.with_suffix('.png'));rgb.verify()
        assert Image.open(p.with_suffix('.png')).size==(cfg['camera']['width'],cfg['camera']['height'])
        assert arr['depth'].shape==(cfg['camera']['height'],cfg['camera']['width'])
        assert arr['segmentation'].shape==(*arr['depth'].shape,2)
        assert np.all(np.isfinite(arr['depth'])) and np.all(arr['depth']>0)
        assert meta['timestamp']>previous;previous=meta['timestamp']
        a,b,c=map(np.array,[meta['T_W_C'],meta['T_W_O'],meta['T_C_O']])
        np.testing.assert_allclose(inverse(a)@b,c,atol=1e-12)
        for T in [a,b,c]:
            np.testing.assert_allclose(T[3],[0,0,0,1]);np.testing.assert_allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-10)
        if camera is not None:np.testing.assert_array_equal(a,camera)
        camera=a
        if 'dq_arm' in meta:
            assert np.max(np.abs(meta['dq_arm']))<cfg['motion']['speed_tolerance']
            vel=np.array(meta['cube_velocity'])
            assert np.linalg.norm(vel[:3])<cfg['motion']['cube_speed_tolerance']
            assert np.linalg.norm(vel[3:])<cfg['motion']['cube_angular_speed_tolerance']
        if 'directions_object' in arr:directions=arr['directions_object']
        else:
            from active_scan.core import cube_surface
            local,_,_=cube_surface(cfg['scene']['cube_size'],cfg['visibility']['grid'])
            cam_local=(inverse(b)@np.r_[a[:3,3],1])[:3]
            directions=cam_local-local;directions/=np.linalg.norm(directions,axis=1)[:,None]
        coverage.update(arr['quality'],directions)
    stored=np.load(out/'coverage.npz')
    np.testing.assert_array_equal(coverage.quality,stored['quality'])
    np.testing.assert_array_equal(coverage.counts,stored['observations'])
    metrics=coverage.metrics(stored['faces'])
    assert abs(metrics['binary_coverage']-summary['coverage']['binary_coverage'])<1e-12
    assert abs(metrics['multiview_coverage']-summary['coverage']['multiview_coverage'])<1e-12
    if summary['success']:
        assert summary['fsm']=='DONE' and summary['forbidden_collision_steps']==0
        assert metrics['binary_coverage']>=cfg['planning']['target_coverage']
        assert summary['stages']['placement'] and summary['stages']['regrasp']
        assert summary['selected_grasps'][0]['contact_axis']!=summary['selected_grasps'][1]['contact_axis']
        assert not summary['weld_active_final']
    if summary['simulation_level']=='C':assert not summary['weld_active_final']
    return {'path':str(out.relative_to(ROOT)),'captures':len(frames),'success':summary['success'],'coverage_recomputed':metrics['binary_coverage'],'status':'verified'}


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--prefix',default='acceptance_');parser.add_argument('--sweep',type=Path,default=ROOT/'results/physical_sweep');parser.add_argument('--out',type=Path,default=ROOT/'results/ARTIFACT_VALIDATION.json')
    args=parser.parse_args()
    source=ROOT/'third_party/robotstudio_so101';manifest=json.loads((source/'SOURCE.json').read_text())
    for name,expected in manifest['sha256'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==expected,name
    rows=[validate_episode(ROOT/'results'/f'{args.prefix}{n}') for n in 'ABC']
    sweep=json.loads((args.sweep/'summary.json').read_text())
    assert len(sweep['cases'])==6 and not any(c['weld_active_final'] for c in sweep['cases'])
    write_json(args.out,{'episodes':rows,'model_hashes_checked':len(manifest['sha256']),'physical_cases_checked':6})
    print(rows)


if __name__=='__main__':main()

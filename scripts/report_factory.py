"""Consolidate production-loop regression evidence without hiding failed contact tests."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import mujoco as mj
from active_scan.core import ROOT,load_config,write_json
from active_scan.simulation import SimulatorAdapter
from active_scan.geometry import VisibilityEvaluator


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    base=read(ROOT/'results/efficient_smooth_B/summary.json')
    cfg=load_config(ROOT/'configs/factory.json');oldcfg=load_config(ROOT/'configs/efficient.json')
    for field in ['velocity','acceleration','reference_limit_fraction']:
        assert cfg['motion'][field]==oldcfg['motion'][field]
    assert cfg['physics']==oldcfg['physics']
    for field in ['force_limit','gripper_force_limit']:assert cfg['robot'][field]==oldcfg['robot'][field]
    benchmark=read(ROOT/'results/factory_acceptance/BENCHMARK.json')
    sim=SimulatorAdapter(cfg,'B');v=VisibilityEvaluator(sim);summaries=[]
    for folder in sorted((ROOT/'results/factory_acceptance').glob('run_*')):
        s=read(folder/'summary.json');assert s['success'] and s['fsm']=='DONE'
        assert s['stages']['flip'] and s['regrasps']==1 and not s['weld_active_final']
        assert s['forbidden_collision_steps']==0
        assert np.all(np.asarray(s['measured_velocity_max'])<=cfg['motion']['velocity'])
        assert np.all(np.asarray(s['actuator_force_max'])<=np.max(np.abs(sim.model.actuator_forcerange),axis=1)+1e-10)
        quality=np.zeros(len(v.local))
        for i,f in enumerate(sorted(folder.glob('frame_*.json'))):
            frame=read(f);assert not v.frame_rejections(frame['frame_quality'])
            arrays=np.load(folder/f'frame_{i:03d}.npz');quality=np.maximum(quality,arrays['quality'])
            assert int(np.count_nonzero((arrays['segmentation'][:,:,0]==sim.cubegeom)&(arrays['segmentation'][:,:,1]==int(mj.mjtObj.mjOBJ_GEOM))))==frame['frame_quality']['visible_pixels']
        np.testing.assert_array_equal(quality,np.load(folder/'coverage.npz')['quality'])
        summaries.append(s)
    assert len(summaries)==6 and benchmark['successes']==6
    nominal=summaries[0];times=[s['planning_seconds'] for s in summaries]
    contact=read(ROOT/'results/factory_contact/summary.json')
    assert not contact['success'] and contact['failure']['code']=='TEST_LIFT_FAILED' and contact['captures']==0
    brake=read(ROOT/'results/factory_stop_probe/summary.json')
    assert brake['stop']['method']=='controlled_brake' and np.max(np.abs(brake['final_speed']))<.05
    result={'status':'verified','simulation_level':'B','hardware_validated':False,'contact_cycle_validated':False,
            'acceleration_profile_changed':False,'force_caps_changed':False,
            'baseline_planning_seconds':base['planning_seconds'],'nominal_planning_seconds':nominal['planning_seconds'],
            'nominal_planning_speedup':base['planning_seconds']/nominal['planning_seconds'],
            'planning_min_seconds':min(times),'planning_median_seconds':float(np.median(times)),
            'planning_max_seconds':max(times),'largest_planning_operation_seconds':max(s['max_planning_call_seconds'] for s in summaries),
            'successful_scenarios':6,'total_scenarios':6,'nominal_motion_seconds':nominal['simulated_seconds'],
            'nominal_serial_motion_plus_planning_seconds':nominal['sim_plus_planning_seconds'],
            'nominal_tracking_max':nominal['tracking_max'],'nominal_tracking_rms':nominal['tracking_rms'],
            'nominal_arm_force_saturation_fraction':nominal['force_saturation_fraction'][:5],
            'nominal_coverage':nominal['coverage'],'cold_preparation':read(ROOT/'results/factory_library/summary.json')['planning_seconds'],
            'contact_failure':contact['failure'],'stop_probe':brake,'tests_passed':48,
            'timing_note':'Wall-clock planning measurements depend on machine load; six simulations do not qualify hardware reliability.',
            'camera_note':'CameraObserver integration contract tested; real-image pose estimation and hardware driver are not implemented.'}
    write_json(ROOT/'results/FACTORY_ACCEPTANCE.json',result);sim.close()
    print(json.dumps({k:result[k] for k in ['status','nominal_planning_speedup','planning_min_seconds','planning_median_seconds','planning_max_seconds','largest_planning_operation_seconds','successful_scenarios']}))


if __name__=='__main__':main()

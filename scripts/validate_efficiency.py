"""Audit accepted pixels, coverage, flip, measured speeds and before/after timing."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import mujoco as mj
from PIL import Image,ImageDraw,ImageFont
from active_scan.core import ROOT,write_json
from active_scan.simulation import SimulatorAdapter
from active_scan.geometry import VisibilityEvaluator


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--episode',type=Path,default=ROOT/'results/efficient_smooth_B')
    parser.add_argument('--out',type=Path,default=ROOT/'results/EFFICIENCY_COMPARISON.json')
    args=parser.parse_args();p=args.episode
    cfg=read(p/'config.json');summary=read(p/'summary.json')
    sim=SimulatorAdapter(cfg,'B');visibility=VisibilityEvaluator(sim)
    assert summary['success'] and summary['stages']['flip'] and summary['regrasps']==1
    assert not summary['weld_active_final'] and summary['forbidden_collision_steps']==0
    frames=[read(f) for f in sorted(p.glob('frame_*.json'))]
    assert len(frames)==summary['captures']
    coverage=np.zeros_like(visibility.local[:,0]);qualities=[]
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',20)
    rows=(len(frames)+2)//3
    sheet=Image.new('RGB',(1440,400*rows),'#101721');draw=ImageDraw.Draw(sheet)
    for i,frame in enumerate(frames):
        quality=frame['frame_quality'];assert not visibility.frame_rejections(quality)
        arrays=np.load(p/f'frame_{i:03d}.npz');seg=arrays['segmentation']
        count=np.count_nonzero((seg[:,:,0]==sim.cubegeom)&(seg[:,:,1]==int(mj.mjtObj.mjOBJ_GEOM)))
        assert count==quality['visible_pixels']
        coverage=np.maximum(coverage,arrays['quality']);qualities.append(quality)
        assert np.max(np.abs(frame['dq_arm']))<cfg['motion']['speed_tolerance']
        x=(i%3)*480;y=(i//3)*400
        sheet.paste(Image.open(p/f'frame_{i:03d}.png').resize((480,360)),(x,y))
        draw.text((x+10,y+364),f'{i+1}: {count} px | {quality["distance_m"]:.2f} м | видно {quality["visible_fraction"]:.0%}',font=font,fill='#6ee6bd')
    np.testing.assert_array_equal(coverage,np.load(p/'coverage.npz')['quality'])
    actual_coverage=float(np.mean(coverage>=cfg['visibility']['quality_threshold']))
    assert actual_coverage==summary['coverage']['binary_coverage']>=cfg['planning']['target_coverage']
    dynamics=np.load(p/'joint_dynamics.npz')['samples'];vel=np.max(np.abs(dynamics[:,1:6]),axis=0)
    np.testing.assert_allclose(vel,summary['measured_velocity_max'])
    assert np.all(vel<=cfg['motion']['velocity'])
    forces=np.max(np.abs(dynamics[:,11:]),axis=0)
    assert np.all(forces<=np.max(np.abs(sim.model.actuator_forcerange),axis=1)+1e-10)
    events=[json.loads(s) for s in (p/'events.jsonl').read_text().splitlines()]
    flip=[e for e in events if e['kind']=='flip_verified'];assert len(flip)==1
    assert flip[0]['old_support_axis']!=flip[0]['new_support_axis']
    assert summary['selected_grasps'][0]['contact_axis']!=summary['selected_grasps'][1]['contact_axis']
    baseline=ROOT/'results/acceptance_B';old=read(baseline/'summary.json');oldframes=read(ROOT/'results/old_frame_quality_audit.json')
    new={k:summary[k] for k in ['simulated_seconds','planning_seconds','planning_breakdown','sim_plus_planning_seconds','wall_seconds','tracking_max','captures','coverage','measured_velocity_max','measured_acceleration_max','actuator_force_max','force_saturation_fraction']}
    new.update(first_capture_seconds=frames[0]['timestamp'],min_visible_pixels=min(q['visible_pixels'] for q in qualities),
               min_visible_fraction=min(q['visible_fraction'] for q in qualities),max_camera_distance_m=max(q['distance_m'] for q in qualities),
               min_bbox_short_px=min(q['bbox_short_px'] for q in qualities))
    result={'status':'verified','episode':str(p),'simulation_level':'B','contact_grasp_validated':False,
            'baseline':{'episode':str(baseline),'simulated_seconds':old['simulated_seconds'],'first_capture_seconds':oldframes[0]['time'],
                        'captures':old['captures'],'min_visible_pixels':min(f['visible_pixels'] for f in oldframes)},
            'improved':new,'cycle_speedup':old['simulated_seconds']/summary['simulated_seconds'],
            'first_image_speedup':oldframes[0]['time']/frames[0]['timestamp'],
            'quality_definition_changed':'Minimum pixels per surface cell raised from 1 to 16; old/new coverage is not directly comparable.',
            'clock_note':'Simulation time excludes planning. sim_plus_planning is a serial estimate, not measured hardware takt.',
            'flip_event':flip[0],'accepted_frame_quality':qualities}
    sheet.save(p/'accepted_frames.png');write_json(args.out,result)
    sim.close();print(json.dumps(result,indent=2))


if __name__=='__main__':main()

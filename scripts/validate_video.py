"""Add six seekable chapters and verify that the full-process video is decodable."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from active_scan.core import ROOT, write_json
from active_scan.simulation import SimulatorAdapter
from render_full_process import STEPS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=Path, default=ROOT/'results/full_process_v3')
    args = parser.parse_args()
    out = args.path
    meta = json.loads((out/'video_metadata.json').read_text())
    summary = json.loads((out/'episode/summary.json').read_text())
    states = np.load(out/'recorded_states.npz')
    timeline = states['timeline']
    assert summary['success'] and summary['fsm'] == 'DONE'
    assert all(meta['reproduction_checks'].values())
    assert states['qpos'].shape[0] == meta['recorded_frames']
    assert np.all(np.diff(timeline[:, 0]) >= 0)
    assert timeline[0, 0] == 0
    assert abs(timeline[-1, 0]-summary['simulated_seconds']) < 1e-8
    assert timeline[-1, 1] == summary['captures'] > 0
    assert timeline[-1, 2] == summary['coverage']['binary_coverage']
    assert set(timeline[:, 3]) == set(range(6))
    assert states['eq_active'].any() and not states['eq_active'][-1].any()
    sim = SimulatorAdapter(json.loads((out/'episode/config.json').read_text()), 'B')
    final_cube = states['qpos'][-1, sim.cq:sim.cq+3]
    assert abs(final_cube[2]-.03) < sim.cfg['physics']['placement_height_tolerance']
    assert np.linalg.norm(states['qvel'][-1, sim.cv:sim.cv+3]) < sim.cfg['motion']['cube_speed_tolerance']
    sim.close()
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        raise RuntimeError('ffmpeg and ffprobe must be on PATH')
    total = meta['recorded_frames'] + meta['static_final_frames']
    duration = total/meta['fps']
    chapters = []
    for i, title in enumerate(STEPS):
        idx = int(np.flatnonzero(timeline[:, 3] == i)[0])
        chapters.append({'title': title, 'start_seconds': idx/meta['fps']})
    ffmeta = [';FFMETADATA1', 'title=SO-101: полный цикл сканирования',
              'comment=MuJoCo level B: explicit ideal weld attachment; not validated contact grasp.']
    for i, chapter in enumerate(chapters):
        end = chapters[i+1]['start_seconds'] if i+1 < len(chapters) else duration
        chapter['end_seconds'] = end
        ffmeta.extend(['[CHAPTER]', 'TIMEBASE=1/1000', f'START={int(chapter["start_seconds"]*1000)}',
                       f'END={int(end*1000)}', f'title={chapter["title"]}'])
    (out/'chapters.ffmeta').write_text('\n'.join(ffmeta)+'\n', encoding='utf-8')
    video = out/'full_process.mp4'
    chaptered = out/'full_process_chaptered.mp4'
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
                    '-i', str(video), '-i', str(out/'chapters.ffmeta'), '-map', '0:v:0',
                    '-map_metadata', '1', '-map_chapters', '1', '-c:v', 'copy',
                    '-movflags', '+faststart', str(chaptered)], check=True)
    chaptered.replace(video)
    probe = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-select_streams', 'v:0',
                       '-count_frames', '-show_streams', '-show_format', '-show_chapters',
                       '-of', 'json', str(video)], text=True, encoding='utf-8'))
    stream = probe['streams'][0]
    assert stream['codec_name'] == 'h264'
    assert (stream['width'],stream['height']) == tuple(meta['size'])
    assert int(stream['nb_read_frames']) == total
    assert abs(float(probe['format']['duration'])-duration) < .05
    assert len(probe['chapters']) == 6
    # Decode the entire stream; a readable header alone is not a complete media check.
    subprocess.run([ffmpeg, '-v', 'error', '-xerror', '-nostdin', '-i', str(video), '-map', '0:v:0',
                    '-f', 'null', '-'], stdout=subprocess.DEVNULL, check=True)
    for idx in [1,3,5]:
        ch = chapters[idx]
        when = ch['start_seconds'] + min(2.,(ch['end_seconds']-ch['start_seconds'])/2)
        subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-n', '-ss', str(when),
                        '-i', str(video), '-frames:v', '1', str(out/f'qa_phase_{idx+1}.png')], check=True)
    result = {'status': 'verified', 'duration_seconds': float(probe['format']['duration']),
              'frames_decoded': total, 'size': meta['size'], 'fps': meta['fps'],
              'simulation_level': 'B', 'captures': summary['captures'], 'coverage': summary['coverage']['binary_coverage'],
              'final_cube_position': final_cube, 'final_weld_active': False,
              'all_source_reproduction_checks_passed': True, 'chapters': chapters,
              'file_sha256': hashlib.sha256(video.read_bytes()).hexdigest(),
              'file_bytes': video.stat().st_size}
    write_json(out/'VIDEO_VALIDATION.json', result)
    print(json.dumps(result, default=lambda x:x.tolist(), ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()

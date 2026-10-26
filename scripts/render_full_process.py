"""Record a complete real MuJoCo B episode; render actual states, never interpolate poses.

The overview camera and headlight are presentation-only. The fixed scan camera,
dynamics, task planner and object constraints remain those of the source episode.
Requires the ffmpeg executable on PATH; no additional Python dependencies.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from active_scan.core import ROOT, write_json
from active_scan.simulation import SimulatorAdapter
from active_scan.control import TaskPlanner


STEPS = ['Первый захват', 'Съёмка · 1', 'Переворот', 'Перехват', 'Съёмка · 2', 'Возврат на стол']
DETAILS = {
    'INITIALIZE': 'Исходное положение', 'READ_STATE': 'Исходное положение',
    'PLAN_TASK': 'Планирование первого захвата', 'MOVE_PREGRASP': 'Подход к кубику',
    'APPROACH': 'Точный подход открытого гриппера', 'CLOSE_GRIPPER': 'Закрытие гриппера',
    'VERIFY_GRASP': 'Проверка захвата', 'LIFT': 'Подъём кубика над столом',
    'PLAN_VIEW_SEQUENCE': 'Выбор съёмочных ракурсов', 'MOVE_TO_VIEW': 'Перемещение к новому ракурсу',
    'SETTLE': 'Остановка и стабилизация', 'CAPTURE': 'Сохранение снимка',
    'UPDATE_COVERAGE': 'Обновление покрытия поверхности', 'MOVE_PREPLACE': 'Возврат к зоне укладки',
    'REORIENT': 'Подготовка к укладке', 'PLACE': 'Плавное опускание кубика',
    'RELEASE': 'Раскрытие пальцев', 'VERIFY_PLACEMENT': 'Проверка устойчивости на столе',
    'RETREAT': 'Отвод гриппера от кубика', 'PLAN_REGRASP': 'Выбор другой пары граней',
    'FINISH': 'Кубик на столе · возвращение руки домой', 'DONE': 'Цикл завершён · кубик снова на столе',
    'RECOVERY': 'Остановка по ошибке', 'FAILED': 'Эпизод не завершён',
}


def font(size, bold=False):
    path = Path('C:/Windows/Fonts') / ('segoeuib.ttf' if bold else 'segoeui.ttf')
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.truetype('DejaVuSans.ttf', size)


class VideoRecorder:
    width, height = 1600, 900

    def __init__(self, sim, out, fps, speed, duration, expected_captures):
        self.s, self.out, self.fps, self.speed = sim, Path(out), fps, speed
        self.duration, self.expected_captures = duration, expected_captures
        self.task = None
        self.fonts = {n: font(n) for n in [17, 19, 21, 24, 28, 34]}
        self.bold = {n: font(n, True) for n in [20, 24, 28, 32, 36]}
        sim.model.vis.global_.offwidth = 1088
        sim.model.vis.global_.offheight = 704
        # Only rendering settings: no change to masses, contacts, controls or cameras.
        sim.model.vis.headlight.ambient[:] = .35
        sim.model.vis.headlight.diffuse[:] = .6
        self.overview = mj.Renderer(sim.model, height=704, width=1088)
        self.scan = mj.Renderer(sim.model, height=336, width=448)
        self.options = mj.MjvOption()
        self.options.geomgroup[3:] = False
        self.camera = mj.MjvCamera()
        self.camera.type = mj.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [.15, 0, .13]
        self.camera.distance = 1.05
        self.camera.azimuth = -45
        self.camera.elevation = -30
        self.process = None
        self.frames = 0
        self.rows, self.positions, self.velocities, self.constraints = [], [], [], []
        self.next_time = 0.
        self.last_capture = 0
        self.flash_time = -100.
        self.last_progress = -10.

    def phase(self):
        t = self.task
        if t.fsm in ('FINISH', 'DONE'):
            return 5
        if t.fsm == 'PLAN_REGRASP':
            return 3
        if t.fsm in ('MOVE_PREPLACE', 'REORIENT', 'PLACE', 'RELEASE', 'VERIFY_PLACEMENT', 'RETREAT'):
            return 2 if len(t.selected_grasps) < 2 else 5
        if t.fsm in ('PLAN_VIEW_SEQUENCE', 'MOVE_TO_VIEW', 'SETTLE', 'CAPTURE', 'UPDATE_COVERAGE'):
            return 1 if len(t.selected_grasps) < 2 else 4
        return 0 if len(t.selected_grasps) < 2 else 3

    def compose(self):
        s, t = self.s, self.task
        self.overview.update_scene(s.data, camera=self.camera, scene_option=self.options)
        left = self.overview.render().copy()
        self.scan.update_scene(s.data, camera='scan_camera', scene_option=self.options)
        right = self.scan.render().copy()
        canvas = Image.new('RGB', (self.width, self.height), '#101721')
        canvas.paste(Image.fromarray(left), (16, 94))
        canvas.paste(Image.fromarray(right), (1136, 128))
        d = ImageDraw.Draw(canvas)
        white, muted, mint = '#edf3fa', '#a4b2c5', '#6ee6bd'
        d.text((26, 14), 'SO-101  /  ПОЛНЫЙ ЦИКЛ СКАНИРОВАНИЯ', font=self.bold[28], fill=white)
        d.text((28, 53), DETAILS.get(t.fsm, t.fsm), font=self.fonts[24], fill=mint)
        d.text((1136, 32), f'Время симуляции   {s.data.time:05.1f} / {self.duration:.1f} с', font=self.fonts[19], fill=white)
        d.text((1136, 62), f'Планирование отдельно: {t.planning_seconds:.2f} с',font=self.fonts[17],fill=muted)
        d.text((1136, 95), 'НЕПОДВИЖНАЯ КАМЕРА', font=self.bold[20], fill=white)
        d.text((32, 110), 'Общий вид сцены', font=self.fonts[19], fill=white,
               stroke_width=1, stroke_fill='#161b22')
        if t.captures != self.last_capture:
            self.last_capture = t.captures
            self.flash_time = s.data.time
        if s.data.time - self.flash_time < .65:
            d.rectangle((1134, 126, 1586, 466), outline=mint, width=4)
            d.rectangle((1136, 430, 1584, 464), fill='#153a30')
            d.text((1150, 433), f'Кадр {t.captures:02d} сохранён', font=self.bold[20], fill=mint)
        d.text((1136, 478), f'Снимки   {t.captures:02d} / {self.expected_captures:02d}', font=self.bold[28], fill=white)
        grasp = min(2, max(1, len(t.selected_grasps)))
        axis = ('X', 'Y', 'Z')[t.selected_grasps[-1]['contact_axis']] if t.selected_grasps else '—'
        d.text((1136, 520), f'Захват {grasp} / 2    ·    контактные грани: {axis}', font=self.fonts[21], fill=muted)
        coverage = t.coverage.metrics(t.v.faces)
        d.text((1136, 562), 'Покрытие поверхности', font=self.fonts[24], fill=white)
        d.text((1430, 558), f'{coverage["binary_coverage"]*100:5.1f}%', font=self.bold[28], fill=mint)
        d.rounded_rectangle((1136, 604, 1584, 616), radius=6, fill='#263344')
        progress = coverage['binary_coverage']
        if progress > 0:
            d.rounded_rectangle((1136, 604, 1136 + 448*progress, 616), radius=6, fill=mint)
        for i, (face, stats) in enumerate(coverage['per_face'].items()):
            col, row = i % 3, i // 3
            x, y = 1136 + 151*col, 642 + 62*row
            d.text((x, y), f'{face}   {stats["binary"]*100:.0f}%', font=self.fonts[19], fill=muted)
            d.rectangle((x, y+30, x+130, y+35), fill='#263344')
            if stats['binary'] > 0:
                d.rectangle((x, y+30, x+130*stats['binary'], y+35), fill=mint)
        if hasattr(t,'last_frame_quality'):
            quality=t.last_frame_quality
            d.text((1136, 754), f'Последний кадр: {quality["visible_pixels"]:,.0f} px куба', font=self.fonts[17], fill=mint)
            d.text((1136, 777), f'{quality["distance_m"]:.2f} м  ·  видно {quality["visible_fraction"]*100:.0f}% силуэта', font=self.fonts[17], fill=muted)
        else:
            d.text((1136, 772), 'Куб: 60 мм   ·   5 суставов + гриппер', font=self.fonts[19], fill=muted)
        active = self.phase()
        for i, name in enumerate(STEPS):
            x, y = 16 + 264*i, 812
            fill = '#1b4b3c' if i < active else '#182333'
            if i == active:
                fill = '#27624d'
            d.rounded_rectangle((x, y, x+252, y+34), radius=7, fill=fill)
            d.text((x+10, y+3), f'{i+1}. {name}', font=self.fonts[19], fill=white if i <= active else muted)
        d.text((24, 866), 'РЕЖИМ B: динамика суставов + идеальная weld-связь куба. Контактный захват не подтверждён.',
               font=self.fonts[19], fill='#e8c78d')
        d.text((1430, 866), f'Скорость ×{self.speed:g}', font=self.fonts[19], fill=muted)
        return canvas

    def start(self):
        exe = shutil.which('ffmpeg')
        if not exe:
            raise RuntimeError('ffmpeg executable is required on PATH')
        self.error_log = open(self.out/'ffmpeg.log', 'wb')
        self.process = subprocess.Popen([
            exe, '-hide_banner', '-loglevel', 'warning', '-nostdin', '-n',
            '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{self.width}x{self.height}',
            '-framerate', str(self.fps), '-i', '-', '-an', '-c:v', 'libx264',
            '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            str(self.out/'full_process.mp4'),
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.error_log)

    def record(self, force=False):
        if not force and self.s.data.time + 1e-9 < self.next_time:
            return
        picture = self.compose()
        if self.frames == 0:
            picture.save(self.out/'poster.png')
        self.process.stdin.write(picture.tobytes())
        self.frames += 1
        self.rows.append([float(self.s.data.time), float(self.task.captures),
                          float(self.task.coverage.metrics(self.task.v.faces)['binary_coverage']), self.phase()])
        self.positions.append(self.s.data.qpos.copy())
        self.velocities.append(self.s.data.qvel.copy())
        self.constraints.append(self.s.data.eq_active.copy())
        self.next_time = self.frames * self.speed / self.fps
        if self.s.data.time - self.last_progress >= 10:
            print(f'Rendered sim t={self.s.data.time:.1f}s; frames={self.frames}; state={self.task.fsm}', flush=True)
            self.last_progress = self.s.data.time

    def finish(self):
        self.record(force=True)
        last = self.compose()
        last.save(self.out/'final_frame.png')
        # Two seconds of a clearly finished, static final state; no synthetic motion.
        for _ in range(self.fps*2):
            self.process.stdin.write(last.tobytes())
        self.process.stdin.close()
        code = self.process.wait(timeout=90)
        self.error_log.close()
        if code:
            raise RuntimeError(f'ffmpeg exited with code {code}; see ffmpeg.log')
        np.savez_compressed(self.out/'recorded_states.npz', timeline=np.array(self.rows),
                            qpos=np.array(self.positions), qvel=np.array(self.velocities),
                            eq_active=np.array(self.constraints))
        self.overview.close()
        self.scan.close()
        return {'video': str(self.out/'full_process.mp4'), 'recorded_frames': self.frames,
                'static_final_frames': self.fps*2, 'fps': self.fps, 'speed': self.speed,
                'simulation_duration': float(self.s.data.time), 'size': [self.width, self.height],
                'simulation_level': 'B', 'artificial_attachment': 'explicit equality weld',
                'recording': 'actual post-step states; no pose interpolation or object teleportation'}


class RecordingSimulator(SimulatorAdapter):
    recorder = None

    def step(self):
        super().step()
        if self.recorder is not None:
            self.recorder.record()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT/'results/efficient_smooth_B')
    parser.add_argument('--out', type=Path, default=ROOT/'results/full_process_v3')
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()
    if args.fps < 1 or args.speed <= 0:
        parser.error('fps and speed must be positive')
    if (args.out/'full_process.mp4').exists():
        raise FileExistsError('Choose a new output directory; existing video is preserved')
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((args.source/'config.json').read_text())
    reference = json.loads((args.source/'summary.json').read_text())
    sim = RecordingSimulator(cfg, 'B')
    task = TaskPlanner(sim, args.out/'episode')
    recorder = VideoRecorder(sim, args.out, args.fps, args.speed,
                             reference['simulated_seconds'], reference['captures'])
    recorder.task = task
    if args.preview_only:
        recorder.compose().save(args.out/'preview.png')
        recorder.overview.close();recorder.scan.close();task.log.close();sim.close()
        print(args.out/'preview.png')
        return
    recorder.start()
    sim.recorder = recorder
    recorder.record(force=True)
    result = task.run()
    metadata = recorder.finish()
    checks = {key: bool(np.isclose(result[key], reference[key], atol=1e-10))
              for key in ['simulated_seconds', 'tracking_max', 'tracking_rms', 'captures', 'regrasps']}
    checks['binary_coverage'] = result['coverage']['binary_coverage'] == reference['coverage']['binary_coverage']
    checks['success'] = result['success'] is True
    metadata['source_episode'] = str(args.source)
    metadata['reproduction_checks'] = checks
    write_json(args.out/'video_metadata.json', metadata)
    if not all(checks.values()):
        raise RuntimeError(f'Recorded episode differs from source: {checks}')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()

# D455: реализация и проверка локализации

Обновление 26 октября: добавлена физически мотивированная модель стереошума,
ошибки камеры/энкодеров и отдельные gates по uncertainty. Текущая проверка
и ограничения описаны в [D455_NOISE_MODEL.md](D455_NOISE_MODEL.md).
Прежний `perception_final` ниже — Gaussian baseline, не текущая квалификация.
Hardware runtime/replay теперь используют консервативный error budget;
неопределённость может запретить fusion и grasp evidence даже при TRACKING.

Реализован локальный RGB-D контур для **неподвижной камеры, известного куба
60 мм и калиброванного рабочего места**. Код: `src/active_scan/perception/`.
Это локализация объекта и реконструкция в координатах объекта; мобильная
одометрия камеры и loop closure неподвижной камеры здесь не нужны.

## Что работает

1. `realsense.py`: SDK D455, native Z16 → метры, RGB выравнивается на сетку
   depth. Проверяются серийный номер, K, разрешение, единицы, часы и Min-Z
   профиля. Неподдержанная ненулевая дисторсия отклоняется; автоматическая
   ректификация такого профиля пока не реализована. Доступны USB и `.bag`.
2. `calibration.py`: camera-to-base из известных **в координатах базы** точек,
   PnP RANSAC + уточнение по inliers. Это не автоматическая ChArUco-калибровка.
   В тестах есть независимые контрольные точки, исключённые из подгонки.
3. `localizer.py`: depth ROI → удаление стола и маски робота → компоненты →
   ортогональные плоскости RANSAC → робастная подгонка поверхности куба.
   Нет аргумента с истинной позой, FK объекта или идеальной сегментацией.
   Проверяются число точек, две/три наблюдаемые плоскости, полнота размеров,
   остаточная ошибка и несколько конкурирующих объектов.
4. `pipeline.py`: два последовательных измерения для инициализации, контроль
   возраста/порядка кадров/эпох часов, 24 симметрии куба, потеря и повторный
   поиск. `PREDICTED` — удержанная старая поза с растущей неопределённостью;
   она **не является измерением** и запрещена для grasp evidence и TSDF.
5. `robot_mask.py`: проекция геометрической модели робота из измеренных
   суставов и гриппера. Нужны суставы, синхронизированные с экспозицией;
   командные уставки вместо обратной связи не подходят. FK-модель требует
   проверки на стенде. Используется отдельный scratch state, без cube truth.
6. `scene.py`: медианная карта явно пустой рабочей зоны, карта точек базы,
   проверка высоты стола и выделение foreground. Нужны ≥5 исходных кадров;
   недостающие глубины не заполняются. Сдвиг в плоскости стола и yaw камеры
   эта проверка не обнаруживает: для этого нужна повторная калибровка/якорь.
7. `mapping.py`: object-centric Open3D TSDF, ограниченная библиотека из
   максимум 60 ключевых кадров, оригинальные измеренные глубины, удаление
   flying pixels и смешанных пикселей силуэта. Матрица интеграции —
   `T_C_O = inverse(T_B_C) @ T_B_O`, выход — PLY.
8. `runtime.py`: отдельные потоки захвата и оценки позы, буфер только
   последнего кадра. Контроллер читает готовый результат, не ждёт USB,
   реконструкцию, диск или облако. Ошибка worker и остановка блокируют observer.
9. `recording.py`: запись, хеши, воспроизведение без pickle. Файлы эпизода
   содержат только измерения; evaluator truth хранится отдельно.

По умолчанию всё считается на CPU локально. ROS и облако не нужны для этого
модуля; сетевой сервис в контур управления не добавлен.

## Контракт с управлением

`pipeline.camera_observer()` реализует существующий `CameraObserver`.
`LocalizationRuntime.camera_observer()` дополнительно проверяет состояние
потоков; для hardware режима требует зарегистрированную маску робота.
Синтетические кадры и bag не проходят hardware-проверку источника.
`verify_test_lift()` также отклоняет смену object identity / grasp epoch.

`MeasuredSceneProvider.snapshot()` строит `SceneState` из измерения, отдельно
переданной обратной связи и FK гриппера. Timestamp суставов должен совпадать
с timestamp изображения в пределах 5 мс. Снимок относится к экспозиции;
это не прогноз текущей позы быстро движущегося объекта. Для планирования
по текущему состоянию нужно использовать остановку/достаточно свежий снимок
или отдельную модель компенсации задержки с бюджетом ошибки.

Пример подключения к будущему аппаратному адаптеру:

```python
from active_scan.perception.calibration import load_calibration
from active_scan.perception.realsense import D455Source
from active_scan.perception.runtime import LocalizationRuntime
from active_scan.perception.sync import FeedbackBuffer
from active_scan.perception.robot_mask import SimulatorRobotMask

cal = load_calibration('calibration_d455.json')
feedback = FeedbackBuffer(joints=6)  # 5 измеренных суставов + гриппер
masker = SimulatorRobotMask(dedicated_fk_model, cal)

def mask_at_exposure(timestamp):
    q = feedback.at(timestamp)  # clock: host monotonic; без экстраполяции
    return masker.mask(q[:5], q[5])

with LocalizationRuntime(D455Source(cal), mask_provider=mask_at_exposure) as rt:
    robot.object_observer = rt.camera_observer()
    # robot driver добавляет измеренную обратную связь в feedback;
    # controller вызывает observe(now) в той же шкале monotonic.
```

Аппаратный драйвер SO-101 здесь не создан. Старый `TaskPlanner` остаётся
симуляционным и по умолчанию использует ground truth; подключение только
camera observer не заменяет все его обращения к симулятору. Полный цикл
робота с визуальной обратной связью на железе **не подтверждён**.
Интеграционный тест действительно подключает camera observer к
`SimulatorAdapter.observe_object()`, получая позу из RGB-D.

## Симметрия, перехват и точность

Поза немаркированного куба геометрически определяется с точностью до 24
вращений. Внутри непрерывного слежения выбирается ближайшая ветвь. После
потери/смены grasp epoch карта автоматически не объединяется с прежней:
невозможно доказать соответствие физических граней. `symmetry_ambiguous`
всегда сохраняет этот факт. Текстура/метки для идентификации граней пока
не реализованы. Покрытие всех шести **материальных** граней не заявляется.

Ковариация — консервативная эвристика с нижним пределом, не статистически
откалиброванный интервал. Размер 60 мм является prior в локализации и
фильтре fusion; близость PLY к этому размеру не является независимой
метрологической проверкой. Синтетическая точность не переносится на D455:
не смоделированы все систематические ошибки глубины, материалы, свет,
экспозиция, вибрации и транспортная задержка. Порог скольжения 4 мм требует
самостоятельной проверки точности и синхронизации на реальном стенде.

## Установка и команды

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements-perception.lock.txt
uv pip install --python .venv\Scripts\python.exe --no-deps --no-build-isolation -e .
.venv\Scripts\active-perception.exe probe
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\validate_perception.py --out results\my_perception_check --frames 240 --mujoco --record
```

Те же команды доступны через `python -m active_scan.perception.cli`.
Выходные каталоги новых проверок не должны существовать; старые результаты
не перезаписываются. NumPy/SciPy/SDK/OpenCV/Open3D зафиксированы в lock.
Open3D **0.19.0** проверен тестом интеграции метрической глубины. На этом
Windows-хосте 0.20.0 воспроизводимо давал пустую карту даже для плоскости;
есть соответствующий [upstream issue #7592](https://github.com/isl-org/Open3D/issues/7592).

Калибровка из JSON:

```powershell
active-perception calibrate --input correspondences.json --out calibration_d455.json
active-perception record --calibration calibration_d455.json --seconds 15 --out results\d455_episode
active-perception replay results\d455_episode --out results\d455_replay --map
active-perception scene results\empty_cell_episode --confirm-empty-cell --out empty_scene.npz
active-perception track --calibration calibration_d455.json --scene empty_scene.npz --seconds 15 --out results\live_tracking.json
```

В `correspondences.json`: `points_B` (N×3 в метрах, ≥6 неколлинеарных точек),
`pixels` (N×2, координаты в **ректифицированной depth-сетке**), `profile`:
width, height, K, depth_scale, min_depth, max_depth, serial, profile,
provenance=`hardware`, table_z, workspace. Не подставлять RGB intrinsics
вместо depth K. Измерение положения доски относительно базы обязательно:
PnP не создаёт это измерение. `T_B_C` вычисляется программой, K/scale/serial
берутся у реального устройства. Сохранённая quality JSON не подтверждает
абсолютную точность. Нужны независимые контрольные точки.

Для native bag: `record --bag-output capture.bag ...`, затем
`record --bag-input capture.bag --calibration ... --out ...`.
NPZ-запись ограничена newest-only буфером и считает потерянные кадры;
нативная bag полезна, если требуется исходный поток. `track` выполняет
только локализацию и пишет отчёт, команды роботу не отправляет.

Min-Z проверяется по профилю (March 2026 D400 datasheet): 848×480 — 0.350 м,
640×480 — 0.320 м, 640×360 — 0.260 м, 1280×720 — 0.520 м.
Это axial Z ближайшей поверхности, а не расстояние до центра куба.
SDK-проверка не заменяет подтверждения качества глубины на выбранном режиме.
[Источник: D400 datasheet](https://www.realsenseai.com/wp-content/uploads/2026/03/RealSense-D400-Series-Datasheet-Mar-2026.pdf).

## Что проверяется

- Независимый аналитический ray-box renderer: шум, пропуски, выбросы,
  полный подъём/вращение/возврат куба как trajectory fixture. Он не вызывает
  функции estimator для генерации изображения.
- MuJoCo renderer: только RGB/depth, без segmentation IDs. В тесте accessor
  истинной позы специально подменяется исключением на время sensing.
  Камера и свободные позы куба — тестовая постановка, не новый цикл захвата.
- Старые/переставленные кадры, skew RGB-depth, неверная калибровка, USB epoch,
  одиночная плоскость, два куба, неправильный размер, маленький видимый patch.
- Контрольные точки PnP, единицы глубины, immutable sensor buffers,
  checksum corruption, повторное воспроизведение, метаданные происхождения.
- Захват не подтверждается, если движется только инструмент; 0/2/8 мм
  смещения проверяются по независимым изображениям на нескольких seeds.
- TSDF действительно непустой, ограничен объектом, не принимает prediction,
  несовпадающий кадр или другую identity epoch; PLY проверяется численно.
- Newest-only очередь, исключение USB worker, работа pose worker отдельно
  от контроллера, остановка и запрет устаревшего результата после остановки.

Итоговые численные результаты и hashes: `results/perception_final/summary.json`.
Тесты: `results/perception_final/pytest.xml`. Первая неуспешная проверка
сохранена отдельно (`perception_validation_20261026`) для истории дефектов.
Порог software campaign: ≥85% измерений, p95 ошибки <3 мм и <3° по симметрии.
Эти пороги оценивают software fixtures, а не допуск захвата на производстве.
Время в отчёте — CPU localization, без USB, экспозиции, рендера, записи и TSDF.

Подключённая D455 в этом сеансе не обнаружена. Реальные bag, калибровка,
точность, 30–60-минутный аппаратный soak и серия 100 физических циклов
не проверены. `hardware_accuracy_validated=false`, `robot_acceptance=false`.

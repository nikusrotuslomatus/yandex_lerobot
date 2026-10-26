# Исторический README до публикации

Снимок прежнего описания. Текущий статус и аппаратные ограничения: [README](../README.md).

# SO-101: планирование наблюдений куба в MuJoCo

Работающая Python-система по [исходному ТЗ](../docs/TECHNICAL_HANDOUT.md).
Модель имеет **5 суставов руки + отдельный гриппер**. Куб — 60 мм,
камера неподвижна, состояние известно точно. Обучаемые политики и ROS не используются.

**Текущий результат B:** захват → съёмка → переворот на бок → освобождение →
повторный захват → съёмка → возврат куба на стол → home. Движения занимают
38,158 с, первый снимок — на 7,814 с; отдельно измерено 28,158 с планирования.
Получено 6 кадров и 99,537% геометрического покрытия, включая 97,222% исходной +Z.
Все кадры проходят проверку масштаба и перекрытия гриппером.
Уровень B использует идеальную weld-связь; **контактный захват C не подтверждён**.

[Новое видео ×1](../results/full_process_v3/full_process.mp4) ·
[Кадры](../results/efficient_smooth_B/accepted_frames.png) ·
[Изменения, замеры и ограничения](../docs/IMPROVEMENTS.md).
Исторический исходный эксперимент: [REPORT.md](../docs/REPORT.md).

**Повторяемый цикл с подготовкой планировщика:** используйте
[`configs/factory.json`](../configs/factory.json). Глобальный поиск захватов и
ракурсов выполняется заранее; выбранные позы и изменившиеся пути проверяются
по текущему состоянию. Добавлены пробный подъём, интерфейс проверки захвата
по существующей камере, торможение и отдельные настройки суставов.
Профиль ускорения и пределы усилий сохранены.
Планирование номинального цикла сократилось **с 28,16 до 3,36 с**;
в шести успешных сценариях — 3,07–5,02 с. 48 тестов проходят.
[Изменения и порядок запуска](../docs/FACTORY_CONTROL.md) ·
[Серия проверок](../results/factory_acceptance/BENCHMARK.json) ·
[Актуальное видео](../results/factory_video/full_process.mp4).

## Установка на Windows / PowerShell

Проверено на Windows, Python 3.12.10, MuJoCo 3.16.0. Команды из корня проекта:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.lock.txt
uv pip install --python .venv\Scripts\python.exe --no-deps -e .
```

Модель и meshes уже включены в `third_party/robotstudio_so101/`, сеть для
экспериментов не нужна. Для восстановления исходных файлов:

```powershell
.venv\Scripts\python scripts\fetch_model.py
```

Сохраняются исходная лицензия Apache-2.0, commit и SHA-256 каждого файла.
При установке на другой ОС используйте Python 3.12 и соответствующий путь к
интерпретатору. EGL на Linux не проверялся; на Windows рендеринг выполнен через
невидимый GLFW-контекст, без окна viewer.

## Запуски

Выбирайте новый каталог `--out`: существующие эпизоды сохраняются и не перезаписываются.

```powershell
# Тесты
.venv\Scripts\python -m pytest -q

# Обязательный первый геометрический эксперимент: один/два захвата
.venv\Scripts\python scripts\geometry_experiment.py --candidates 3000 --out results\my_geometry

# A: кинематическая проверка полного FSM
.venv\Scripts\active-scan --level A --candidates 1500 --out results\my_A

# Подготовка библиотеки до рабочих циклов: MuJoCo B, выполняется один раз
.venv\Scripts\python scripts\prepare_library.py --config configs\factory.json --out results\my_setup

# B: повторяемый цикл с подготовленной библиотекой, ЯВНАЯ weld-связь
.venv\Scripts\active-scan --config configs\factory.json --level B --out results\my_B

# C: свободный куб и реальные контакты MuJoCo, без weld
.venv\Scripts\active-scan --level C --candidates 1500 --out results\my_C

# Только физический захват, подъём и выдержка
.venv\Scripts\active-scan --level C --pick-only --out results\my_contact_probe

# Шесть контактных sensitivity cases
.venv\Scripts\python scripts\physical_sweep.py --out results\my_physical_sweep

# Сравнение ракурсов с лимитом 10 кадров и сетками 12/24
.venv\Scripts\python scripts\compare_views.py --geometry results\my_geometry --out results\my_comparison.json

# Независимая проверка комплектных результатов
.venv\Scripts\python scripts\validate_artifacts.py
```

`--single-grasp` разрешает закончить после первой серии, если достигнут целевой
coverage. По умолчанию FSM намеренно выполняет перехват для проверки всей цепочки.
`--seed` меняет sampling. `--config` принимает YAML либо JSON; все параметры
сохраняются в каталоге эпизода. Масса, трение, gains, скорости, пороги и геометрия
стенда — предварительные значения, а не измерения реального робота.

Код возврата CLI равен 0, если программа корректно сохранила результат;
**успех манипуляции проверяйте в `summary.json: success`**, включая ожидаемые
неуспешные контактные испытания. Необработанная программная ошибка возвращает
ненулевой код. `pick_and_hold` и `full_episode` — разные scope результатов.

## Где смотреть результат

### Видео полного процесса

[Исправленный полный цикл: MP4](../results/full_process_v3/full_process.mp4) —
общий вид сцены и неподвижная камера, подписи этапов, снимки и накопленное
покрытие. Видео показывает уровень B с явной идеальной weld-связью куба.
Оно не является подтверждением контактного удержания уровня C.

Для новой записи того же эпизода (нужен `ffmpeg` в PATH):

```powershell
.venv\Scripts\python scripts\render_full_process.py --out results\my_full_video
```

По умолчанию: 1600×900, 24 fps, реальное симуляционное время без ускорения.
`--speed 2` записывает ускоренную вдвое версию. Записываются фактические
состояния после `mj_step`; результат повторного эпизода автоматически
сверяется с `efficient_smooth_B`. Численные состояния сохранены в `recorded_states.npz`.
Общий ракурс и дополнительное освещение меняют только отображение.
Планирование выполняется отдельно от часов симуляции; его вычислительные паузы
не входят в видео. Старое видео сохранено в `results/full_process_visualization/`.

- `results/efficient_smooth_B/` — текущий полный цикл с переворотом и проверкой кадров.
- `results/efficient_C_probe/` — текущая неуспешная контактная проверка.
- `results/EFFICIENCY_COMPARISON.json` — проверенные скорости, изображения и сравнение.
- `results/acceptance_A/`, `acceptance_B/`, `acceptance_C/` — исходные исторические эпизоды.
- `results/geometry_final/` — геометрическая достижимость и один/два захвата.
- `results/physical_sweep/summary.json` — шесть вариантов массы/трения.
- `results/view_comparison.json` — сравнение геометрических baselines.
- `results/ARTIFACT_VALIDATION.json` — проверка файлов, поз и пересчитанного покрытия.
- Более ранние `episode_*`, `validated_*`, `final_*`, `release_*` — отладочные
  и предыдущие проверочные прогоны; их не следует смешивать с итоговой приёмкой.

В каждом эпизоде сохраняются конфигурация, seed, версии, начальное состояние,
FSM events, выбранные захваты/ракурсы, траектории, контакты с силами, tracking,
RGB PNG, depth/segmentation NPZ, позы и итоговый coverage. `implementation.json`
фиксирует hashes Python-кода и конфигураций для итоговых прогонов.

## Устройство кода

- `core.py` — SE(3), SceneState, RobotInterface, дискретизация поверхности, coverage.
- `simulation.py` — создание сцены, MuJoCo adapter, named indices, A/B/C, rendering.
- `geometry.py` — FK/Jacobian/IK, явная проверка расстояний, ray casting, захваты.
- `planning.py` — sampling достижимых q, greedy selection, порядок, quintic,
  Cartesian approach, укладка и перехват.
- `control.py` — executor, стабилизация, FSM, logging, ошибки и recovery.
- `scripts/` — воспроизводимые эксперименты и независимая проверка артефактов.

MuJoCo также используется как геометрическая модель планирования. Для перехода
на железо адаптер должен синхронизировать эту модель с измеренными суставами и
внешними позами объектов, отправлять уставки в драйвер и получать изображения.
Драйвер конкретного Hiwonder и аппаратная проверка не реализованы. Контракт
адаптера и параметры нельзя считать подтверждёнными для реального оборудования.

## Неподвижная D455: локализация и реконструкция

Реализация находится в `src/active_scan/perception/`: драйвер и запись RGB-D,
camera-to-base, геометрическая поза куба без simulator truth, temporal health,
маска робота, fixed-scene reference, object-centric TSDF и camera observer.
Установка, запуск, тесты и границы аппаратной проверки описаны в
[D455_IMPLEMENTATION.md](../docs/D455_IMPLEMENTATION.md).

```powershell
uv pip install --python .venv\Scripts\python.exe -e ".[perception,test]"
.venv\Scripts\python.exe -m active_scan.perception.cli probe
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\validate_perception.py --out results\my_perception --frames 240 --mujoco --record
```

Камера в сеансе реализации не обнаружена; численные software fixtures и
MuJoCo не являются подтверждением аппаратной точности или физического цикла.

Для стереошума, коррелированных/систематических ошибок, шумов суставов,
uncertainty gates и проверки старых ракурсов используется новая модель:
[D455_NOISE_MODEL.md](../docs/D455_NOISE_MODEL.md). Итоговые данные —
`results/perception_sensor_model/`, старый цикл — `results/perception_cycle_preflight/`.
Прежние Gaussian-метрики в `perception_final` являются историческим baseline.

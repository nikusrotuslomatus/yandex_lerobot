# Воспроизводимость, проверки и границы результата

## Установка

Подтверждённая локальная среда: Windows, Python 3.12.10. Версии библиотек
зафиксированы в `requirements-perception.lock.txt`; старый минимальный
`requirements.lock.txt` относится к исходной симуляции без perception extras.

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-perception.lock.txt
uv pip install --python .venv\Scripts\python.exe --no-deps --no-build-isolation -e .
.venv\Scripts\python.exe -m pytest -q
```

Модель SO101 и её Apache-2.0 assets включены. Повторный fetch не требуется.
Open3D зафиксирован на 0.19.0: 0.20.0 давал пустой TSDF из корректной
метрической глубины на проверенном хосте. Для этого добавлен отдельный тест.

## Что запускать

| Цель | Команда из корня |
|---|---|
| Все программные тесты | `python -m pytest -q` |
| Найти реальную камеру | `active-perception probe` |
| Текущая sensor campaign | `python scripts/validate_realistic_perception.py --out results/my_sensor --seeds 6 --frames 24 --mujoco --record-first` |
| Прежние позы с новым сенсором | `python scripts/replay_cycle_sensor_noise.py --out results/my_cycle` |
| Диагностика маски без шума | `python scripts/ablate_cycle_visibility.py --out results/my_ablation.json` |
| Старый Gaussian baseline | `python scripts/validate_perception.py --out results/my_gaussian --frames 240 --mujoco --record` |
| Асинхронная нагрузка | `python scripts/stress_perception.py --seconds 30 --out results/my_soak.json` |
| Подготовка B-библиотеки | `python scripts/prepare_library.py --config configs/factory.json --out results/my_setup` |
| Номинальный цикл B | `active-scan --config configs/factory.json --level B --out results/my_B` |

Используйте новый `--out`, не перезаписывайте старый эксперимент.
Здесь `python`/entry points означают интерпретатор и executables активной `.venv`.
На Linux потребуется подходящий GL backend; CI задаёт EGL. Аппаратная работа
и межплатформенное совпадение физики не заявляются по локальным Windows-тестам.

## Что включено в доказательства

- 165 пройденных тестов, один аппаратный skip при отсутствии D455.
- 600 RGB-D кадров: четыре сценария × шесть seeds × 24 кадра + MuJoCo.
- Независимый ray-box генератор, renderer MuJoCo, oracle только в evaluator.
- Связанные ошибки depth, калибровки и суставов, временные faults, split
  train/holdout для noise fit, unsafe grasp/fusion rejection, ограниченные очереди.
- Отдельные model/code/config hashes и точный sensor replay принятых поз.
- Неуспехи сохранены: контактный C, adverse cases и старые рабочие ракурсы.

`software_pass` означает выполнение заявленных assertions, в том числе
ожидаемых отказов. Это не флаг удачного физического цикла. `READINESS.json`
явно содержит `ready_for_hardware_deployment=false`.

## Почему показатели не смешиваются

Геометрическое coverage с известной позой не равно точности RGB-D pose.
Успех B не доказывает контактный C. Чистый синтетический объект без гриппера
не доказывает локализацию удерживаемого объекта. Ошибка fit не равна ошибке
world-frame позы. Коэффициенты инженерного noise profile не являются
измерениями конкретного serial. CPU processing latency не включает USB,
экспозицию, генератор шума, рендер, запись или fusion.

Презентация описывает исторический идеальный B-контроллер; не использовать её
как заявление об аппаратной готовности текущей D455-локализации.

## Перенос результатов

Основные отчёты, 34-кадровый sensor-only эпизод и исходная trajectory для
cycle preflight доступны прямо в Git. Остальные исторические эпизоды,
тяжёлые контакты/видео и старые промежуточные сборки доступны в Release.
См. [PUBLICATION.md](PUBLICATION.md).

Исторические `implementation.json`/`manifest.json` фиксируют байты на момент
прогона. Документация могла измениться позже; это не повод переписывать
исторические hashes. Публикационный манифест фиксирует текущий набор файлов.
Git EOL conversion отключён, чтобы checkout не менял SHA256 источников.

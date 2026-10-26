# SO-101 Active Scan: управление и RGB-D локализация D455

Исследовательский проект осмотра куба 60 мм пятиосевой рукой SO-101 с гриппером:
захват → подъём → съёмка → укладка на другую грань → перехват → досъёмка →
возврат на стол. Камера Intel RealSense D455 неподвижна.

**Статус: программный прототип; к загрузке на физический робот не готов.**

| Что проверено | Результат и границы |
|---|---|
| Полный симуляционный цикл B | выполнен с идеальной weld-связью; это не доказательство контакта |
| Контактный C | пробный подъём не прошёл |
| Программные тесты | 165 passed, 1 hardware skip; D455 не подключена |
| Новая sensor campaign | 600 RGB-D кадров, сценарные коэффициенты, без аппаратной калибровки |
| Старые рабочие ракурсы | 0/6 прошли новую проверку с шумом и запасом маски 4 мм; без шума и с точными суставами — 2/6 |
| Аппаратная интеграция | драйвер D455 есть; драйвер робота и полный measured-state FSM не завершены |

Номинальный synthetic tracking 95.8% и p95 позиции 1.2–3.5 мм относятся к
инженерной модели ошибок и свободным тестовым ракурсам. Они не подтверждают
точность вашей камеры или локализацию объекта за гриппером. Актуальные
[результаты](results/perception_sensor_model/RESULTS.md) и
[READINESS.json](results/perception_sensor_model/READINESS.json) сохраняют эти ограничения.

## Что находится в проекте

- Геометрия FK/IK, столкновения, достижимые ракурсы и выбор по информации/времени.
- Подготовленная библиотека планирования, плавные траектории и ограниченный PD
  с velocity/bias feedforward; FSM с проверками и восстановлением.
- Native RGB-D D455, camera-to-base, локализация куба, tracking с симметриями,
  карта пустой сцены, object-centric TSDF, запись и точный replay.
- Физически мотивированная модель disparity/depth, корреляций, калибровки,
  вибраций, границ, экспозиции, времени и ошибок энкодеров/люфтов.
- Отдельный error budget: малая невязка поверхности не даёт разрешение на
  управление при неизвестной абсолютной точности.
- Полная история экспериментов и видео, исходники/готовый PDF Beamer-презентации.

[Вся документация](docs/INDEX.md) · [Архитектура](docs/ARCHITECTURE.md) ·
[Логика и математика](docs/CONTROL_LOGIC.md) · [Модель шума](docs/D455_NOISE_MODEL.md)

## Быстрый старт

Подтверждённая локальная среда — Windows, Python 3.12.10. Модель робота и
meshes уже включены; после установки Python-зависимостей fetch не требуется.

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-perception.lock.txt
uv pip install --python .venv\Scripts\python.exe --no-deps --no-build-isolation -e .
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\active-perception.exe probe
```

Альтернатива установке из lock: `pip install -e ".[perception,test]"` в Python 3.12.
На Linux пути к executables отличаются; для headless MuJoCo CI задаёт EGL.

```powershell
# Отдельная офлайн-проверка sensor model; новый каталог --out.
.venv\Scripts\python.exe scripts\validate_realistic_perception.py --out results\my_sensor --seeds 6 --frames 24 --mujoco --record-first

# Проверка прежней траектории с гриппером и новой моделью камеры.
.venv\Scripts\python.exe scripts\replay_cycle_sensor_noise.py --out results\my_preflight

# Подготовка и номинальный симуляционный цикл B, без заявления о hardware readiness.
.venv\Scripts\python.exe scripts\prepare_library.py --config configs\factory.json --out results\my_setup
.venv\Scripts\active-scan.exe --config configs\factory.json --level B --out results\my_B
```

Глобальный поиск вынесен в подготовку. После переноса checkout библиотеку
нужно подготовить заново: XML содержит абсолютные пути к assets.
Код CLI может сохранить неуспешный эпизод с нулевым process exit code:
проверяйте `summary.json.success`, `failure` и `READINESS`, а не только exit code.

Собственная неподвижная камера не требует мобильной VIO-одометрии. IMU не
измеряет движение куба; её модель здесь диагностическая. Облако и ROS в
рабочем пути не используются. Команды выше не отправляют уставки реальному роботу.

## Результаты, видео и презентация

В Git включены основной sensor report и 34-кадровый sensor-only эпизод,
cycle preflight с изображениями, ключевые B/C-результаты и исходная траектория.
**Вся** история результатов, видео, cache и промежуточные презентационные
файлы сохранены в [GitHub Release](https://github.com/nikusrotuslomatus/yandex_lerobot/releases/tag/v0.1.0-research-snapshot-2026-10-26).
Инструкции восстановления и checksums: [PUBLICATION.md](docs/PUBLICATION.md).

- [Логика данных и источников состояния](docs/ARCHITECTURE.md).
- [Номинальная серия B](results/factory_acceptance/BENCHMARK.json).
- [Контактный тест C](results/factory_contact/summary.json).
- [Проверка старых ракурсов](results/perception_cycle_preflight/summary.json).
- [Презентация PDF](presentation_beamer/robot_control.pdf),
  [исходник](presentation_beamer/main.tex) и [сборка](presentation_beamer/README.md).
- [Историческое описание](docs/HISTORICAL_README.md), [исходное ТЗ](docs/TECHNICAL_HANDOUT.md).

Исторические видео и презентация показывают идеальную B-связь и не заменяют
новую sensor preflight. Полный аппаратный цикл, контактный захват и порог slip
4 мм не приняты. Нужны реальные записи D455, независимая калибровка и согласование
локализатора частично видимого куба с библиотекой ракурсов.

Сторонние материалы и лицензия модели: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

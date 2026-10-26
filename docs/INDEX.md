# Документация проекта

Этот snapshot объединяет геометрический контроллер SO-101, активный осмотр
куба, визуализацию, Beamer-презентацию и модуль локализации неподвижной D455.

## Читать в таком порядке

1. [README](../README.md): текущий статус, установка и быстрый запуск.
2. [Архитектура](ARCHITECTURE.md): модули, источники состояния, координаты,
   потоки данных и граница между симуляцией и аппаратным кодом.
3. [Логика и математика управления](CONTROL_LOGIC.md): FSM, захват, выбор
   ракурсов, траектории, регулятор, проверка результата и восстановление.
4. [D455: реализация](D455_IMPLEMENTATION.md): RGB-D, PnP, tracking, TSDF,
   recording/replay, API и подключение к контроллеру.
5. [Модель ошибок D455](D455_NOISE_MODEL.md): физическая модель, инженерные
   предположения, error budget, параметры, fit по реальным данным и проверки.
6. [Воспроизводимость и готовность](REPRODUCIBILITY.md): что подтверждено,
   какие тесты запускать и чего не доказывают отчёты.
7. [Артефакты и GitHub Release](PUBLICATION.md): полный архив работ,
   восстановление результатов и контроль целостности.

## Основные доказательства

- [Результаты sensor campaign](../results/perception_sensor_model/RESULTS.md).
- [Машиночитаемая готовность](../results/perception_sensor_model/READINESS.json).
- [600 проверочных RGB-D кадров](../results/perception_sensor_model/summary.json).
- [Проверка прежних рабочих ракурсов](../results/perception_cycle_preflight/summary.json).
- [165 тестов и аппаратный skip](../results/perception_sensor_model/pytest.xml).
- [Номинальная симуляционная серия B](../results/factory_acceptance/BENCHMARK.json).
- [Неуспешный контактный тест C](../results/factory_contact/summary.json).
- [Презентация: исходник](../presentation_beamer/main.tex),
  [PDF](../presentation_beamer/robot_control.pdf),
  [описание сборки](../presentation_beamer/README.md).

## Исторические документы

[Исходное ТЗ](TECHNICAL_HANDOUT.md), [первоначальный отчёт](REPORT.md),
[ускорение и качество кадров](IMPROVEMENTS.md),
[контроллер и библиотека планирования](FACTORY_CONTROL.md),
[исходный план локализации](D455_LOCALIZATION_PLAN.md).
Числа и утверждения в них относятся к указанному этапу работы. Актуальные
ограничения имеют приоритет: полный аппаратный цикл не принят, прежние ракурсы
не квалифицированы для новой локализации, sensor coefficients ещё сценарные.

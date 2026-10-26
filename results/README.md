# Результаты и исходные данные

Актуальная точка входа — [perception_sensor_model/RESULTS.md](perception_sensor_model/RESULTS.md)
и [READINESS.json](perception_sensor_model/READINESS.json).

В Git сохранены основной sensor dataset, итоговые отчёты, cycle preflight,
номинальный B-пример, неуспешный C и исходные recorded states старого видео.
Полный набор исторических экспериментов и само видео находятся в
[GitHub Release](https://github.com/nikusrotuslomatus/yandex_lerobot/releases/tag/v0.1.0-research-snapshot-2026-10-26).
Инструкция восстановления — [PUBLICATION.md](../docs/PUBLICATION.md).

Старый успех B, geometric coverage и Gaussian fixtures нельзя смешивать с
аппаратной точностью D455. Текущий noisy preflight не квалифицировал ни один
из шести старых ракурсов при заданной 4-мм ошибке маски. Physical acceptance false.

# Публикация и полный архив работ

Репозиторий: [nikusrotuslomatus/yandex_lerobot](https://github.com/nikusrotuslomatus/yandex_lerobot).
Первый snapshot публикуется приватно, как research prerelease:
[v0.1.0-research-snapshot-2026-10-26](https://github.com/nikusrotuslomatus/yandex_lerobot/releases/tag/v0.1.0-research-snapshot-2026-10-26).

## Разделение Git и Release

В Git находятся весь авторский код, тесты, конфигурации, документация,
лицензированная модель/meshes, Beamer source и итоговый PDF, основной sensor
dataset с 34 кадрами, итоговые отчёты и входная trajectory для cycle preflight.

Полная локальная история занимает около 1.5 ГБ. Она не удалена и не потеряна:
содержимое сохранено в отдельных release assets с исходными путями.

| Asset | Содержимое |
|---|---|
| `experiment-results.zip` | весь `results/`, включая ранние/неуспешные эпизоды, видео, contact logs, RGB-D, trajectories, профили и отчёты |
| `planning-library.zip` | весь `.planning_library/` на момент публикации |
| `presentation-working-files.zip` | весь `presentation_beamer/`, включая исходники, PDF, картинки и промежуточные сборки |
| `artifact-inventory.json` | размер и SHA256 каждого файла внутри архивов |

Размеры и SHA256 **готовых архивов** находятся в [ARTIFACTS.json](ARTIFACTS.json).
Исходный код того же commit доступен также в автоматически создаваемом
GitHub Source code archive. `.venv`, Python caches, `.git` зависимостей и полный
upstream checkout Menagerie не публикуются: они восстанавливаются из lock,
указанных upstream commits и vendored assets.

## Получить полную историю

Из корня клона (GitHub CLI должен иметь доступ к приватному репозиторию):

```powershell
gh release download v0.1.0-research-snapshot-2026-10-26 --repo nikusrotuslomatus/yandex_lerobot --dir .publish/download
Get-FileHash .publish/download/*.zip -Algorithm SHA256
```

Сверьте hashes с `docs/ARTIFACTS.json`. Архивы содержат корневые директории
`results/`, `.planning_library/` и `presentation_beamer/`. Для чистого клона:

```powershell
Expand-Archive .publish/download/experiment-results.zip -DestinationPath . -Force
Expand-Archive .publish/download/planning-library.zip -DestinationPath . -Force
Expand-Archive .publish/download/presentation-working-files.zip -DestinationPath . -Force
```

`-Force` предназначен для восстановления файлов того же snapshot в чистом клоне.
Если уже есть свои эксперименты, распакуйте в отдельную папку и сравните файлы.
Historical документы могут ссылаться на эпизоды, доступные только после
распаковки; ключевые ссылки актуального README работают прямо на GitHub.

Cache переносится как история проверки. Из-за абсолютных путей в XML его
нельзя считать готовой библиотекой нового checkout: выполните prepare заново.
Для запуска unit/integration tests полные release archives не нужны.

## Как собраны архивы

`scripts/package_publication.py` читает файлы потоково, не меняет исходники,
сохраняет относительные пути и вычисляет per-file SHA256 при записи ZIP.
Проверяется CRC всех entries, затем SHA256 каждого ZIP. Скрипт отказывается
перезаписывать существующие assets. Пример нового snapshot:

```powershell
.venv\Scripts\python.exe scripts/package_publication.py --out .publish/new-snapshot
```

Git не нормализует EOL: `.gitattributes` сохраняет байты, чтобы исходные SHA256
не менялись при checkout. Исторические manifests не переписываются после
редактирования документации. Актуальные публикационные проверки находятся
в `results/github_publication/`.

## CI и валидация

GitHub Actions запускает pytest на Ubuntu/EGL с закреплёнными dependencies.
Это дополнительная software-проверка, не physical acceptance. Подтверждённые
Windows-прогоны и чистый экспорт Git сохраняются отдельно; состояние CI нужно
смотреть у конкретного commit. Отсутствие камеры даёт явно обозначенный skip.

Публикация не меняет `ready_for_hardware_deployment=false` и не делает старые
ракурсы или контактный захват подтверждёнными.

Research snapshot: SO-101 control, D455 localization and sensor uncertainty

This prerelease preserves the complete project work and experiment history.

- Source, tests, native D455 recording/replay, fixed-camera localization, object TSDF and noise models are in the repository.
- Architecture, FSM, controller math, frame conventions, uncertainty logic and reproduction commands are in docs/INDEX.md.
- A clean Git export passes 165 tests with 1 physical-camera skip.
- All experiment results, videos, caches and intermediate Beamer files are attached losslessly; per-file hashes are in artifact-inventory.json.
- Verify archive hashes against ARTIFACTS.json before extraction. ZIP entries preserve their project-relative paths.

Important validation boundary: hardware deployment is not ready. No connected D455 was found, coefficients are engineering scenarios, physical contact grasp is unconfirmed, and all six legacy views fail the conservative noisy localization preflight. Nominal simulator success is not physical acceptance.

Contents:
- experiment-results.zip: complete results/ history, including successful and failed runs, raw datasets and videos.
- planning-library.zip: original .planning_library/ snapshot; rerun preparation after moving the checkout.
- presentation-working-files.zip: complete Beamer source/build history.
- artifact-inventory.json: hashes for every archived input file.
- ARTIFACTS.json: archive sizes, SHA256, entry counts and CRC status.

Documentation: https://github.com/nikusrotuslomatus/yandex_lerobot/blob/main/docs/INDEX.md
Readiness: https://github.com/nikusrotuslomatus/yandex_lerobot/blob/main/results/perception_sensor_model/READINESS.json

Verified GitHub Actions on Ubuntu 22.04/EGL: 165 passed, 1 physical-camera skip.
Run: https://github.com/nikusrotuslomatus/yandex_lerobot/actions/runs/40324743886
Attached github-actions-pytest.xml and CI_VALIDATION.json preserve the remote test proof.

Latest tagged snapshot CI also passed: https://github.com/nikusrotuslomatus/yandex_lerobot/actions/runs/40325973553
Latest proof: github-actions-pytest-latest.xml and CI_VALIDATION_latest.json.

Quick video: factory-cycle-ideal-B.mp4 is attached separately. It demonstrates the historical B ideal-weld cycle, not physical grasp or the new sensor preflight.

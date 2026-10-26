# Third-party materials

The vendored SO101 robot model in `third_party/robotstudio_so101/` comes from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), commit
`cda5236f98383f6fbaa4820c6b18dba892cd087f`. Its original Apache-2.0 license,
README, changelog and per-file hashes are preserved alongside the model.
See [SOURCE.json](third_party/robotstudio_so101/SOURCE.json) and
[LICENSE](third_party/robotstudio_so101/LICENSE).

Model derivation and upstream SO-ARM100 references are described by its
original README. Reference snapshots in `docs/sources/` retain source URLs and
commit metadata in `repositories.json`; they do not transfer ownership.

Python dependencies retain their respective upstream licenses. They are
installed using the supplied version pins rather than vendored binaries.
No separate open-source license is assigned here to the project's original
code; the repository is initially private.

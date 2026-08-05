# `data/`

Datasets (captured episodes, labeled examples for representation probes,
benchmark splits) once there's a real benchmark to store data for. Empty
for now -- the same reason `src/atr/` is empty: see that directory's
README.

`data/runs/` (D-057, `src/atr/evaluation/tracking.py`): tracked experiment
runs land here by default -- one directory per run
(`<UTC timestamp>_<run name>/`), each with a `summary.json` (metadata +
the bootstrap-CI report) and one `.jsonl` episode log per policy (D-056).
Generated, not authored -- same reasoning as everything else in this
directory.

Everything here except this file is gitignored -- datasets don't belong
in git history.

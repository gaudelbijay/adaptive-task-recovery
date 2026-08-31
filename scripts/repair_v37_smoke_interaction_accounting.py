#!/usr/bin/env python3
"""Repair only inherited simulator-transition totals in the V37 smoke completion."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


RUN = Path("results/visual_recovery_ppo/visual_recovery_v19_dense_canonical_v37_smoke/v19_dense_canonical_smoke/seed_1788")


def main():
    path = RUN / "TRAINING_COMPLETE.json"
    completion = json.loads(path.read_text())
    if completion.get("training_protocol") != "dense_paired_domain_repair_v19":
        raise ValueError("unexpected V37 training protocol")
    local = int(completion["dense_canonical_training_transitions"])
    if local != 320000 or int(completion["simulator_transitions"]) != local:
        raise ValueError("unexpected V37 local transition count")
    source_path = Path(completion["source_visual_checkpoint"]).parent / "TRAINING_COMPLETE.json"
    source = json.loads(source_path.read_text())
    initialization = int(source["total_simulator_transitions"])
    if initialization != 100255744:
        raise ValueError("unexpected audited V36 cumulative transition count")
    observed_pair = (
        int(completion["initialization_simulator_transitions"]),
        int(completion["total_simulator_transitions"]),
    )
    if observed_pair not in ((256000, 576000), (initialization, initialization + local)):
        raise ValueError(f"unexpected V37 accounting state: {observed_pair}")
    completion["initialization_simulator_transitions"] = initialization
    completion["total_simulator_transitions"] = initialization + local
    completion["accounting_repair"] = {
        "scope": "completion-report-only; checkpoint tensors and local budget unchanged",
        "reason": "V37 selected V36 local simulator_transitions before cumulative total_simulator_transitions",
        "repair_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_completion_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run V30 development OOD with explicit DAgger accounting."""

from run_v28_unseen_visual_ood import main


if __name__ == "__main__":
    main(
        evaluator_script="evaluate_v30_visual_recovery.py",
        execution_protocol="V30 observed-suite development execution",
        execution_prefix="development_ood_execution",
    )

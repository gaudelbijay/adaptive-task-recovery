#!/usr/bin/env python3
"""Run the observed V35 development suite."""

from run_v28_unseen_visual_ood import main


if __name__ == "__main__":
    main(
        evaluator_script="evaluate_v35_visual_recovery.py",
        execution_protocol="V35 observed-suite development execution",
        execution_prefix="development_ood_execution",
    )

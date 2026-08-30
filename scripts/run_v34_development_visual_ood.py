#!/usr/bin/env python3
"""Run the observed V34 development suite with explicit accounting."""

from run_v28_unseen_visual_ood import main


if __name__ == "__main__":
    main(
        evaluator_script="evaluate_v34_visual_recovery.py",
        execution_protocol="V34 observed-suite development execution",
        execution_prefix="development_ood_execution",
    )

#!/usr/bin/env python3
"""Run one V36 development-domain task."""

from run_v28_unseen_visual_ood import main


if __name__ == "__main__":
    main(
        evaluator_script="evaluate_v36_visual_recovery_development_ood.py",
        execution_protocol="V36 development-domain execution",
        execution_prefix="v36_development_ood_execution",
    )

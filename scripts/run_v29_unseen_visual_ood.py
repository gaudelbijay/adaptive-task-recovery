#!/usr/bin/env python3
"""Run one task from the inherited frozen unseen suite for V29."""

from run_v28_unseen_visual_ood import main


if __name__ == "__main__":
    main(
        evaluator_script="evaluate_v29_visual_recovery_unseen_ood.py",
        execution_protocol="V29 inherited unseen visual-domain execution",
        execution_prefix="unseen_ood_execution",
    )

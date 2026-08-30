#!/usr/bin/env python3
"""Run one task from the untouched D-176 V35 confirmation suite."""

from run_v28_unseen_visual_ood import main


if __name__ == "__main__":
    main(
        evaluator_script="evaluate_v35_visual_recovery_unseen_ood.py",
        execution_protocol="V35 D-176 unseen visual-domain execution",
        execution_prefix="v35_confirmatory_unseen_ood_execution",
    )

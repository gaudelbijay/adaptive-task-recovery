#!/usr/bin/env python3
"""Run one task from the frozen V41 untouched visual-domain suite."""

from run_v28_unseen_visual_ood import main


if __name__ == "__main__":
    main(
        evaluator_script="evaluate_v41_visual_recovery_unseen_ood.py",
        execution_protocol="V41 untouched visual-domain execution",
        execution_prefix="v41_confirmatory_unseen_ood_execution",
    )

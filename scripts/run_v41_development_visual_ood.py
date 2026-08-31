#!/usr/bin/env python3
from run_v28_unseen_visual_ood import main


if __name__ == "__main__":
    main(evaluator_script="evaluate_v41_visual_recovery_development_ood.py",
         execution_protocol="V41 development-domain execution",
         execution_prefix="v41_development_ood_execution")

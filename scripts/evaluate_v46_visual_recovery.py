#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import evaluate_visual_recovery_ppo as base
from evaluate_v44_visual_recovery import install_protocol_adapter
from v44_multiview_feature_agent import CalibratedHybridFeatureV41Agent


def configure_threshold():
    config_path = Path(sys.argv[sys.argv.index("--config") + 1])
    config = json.loads(config_path.read_text())
    experiments = config.get("experiments", [])
    if len(experiments) != 1:
        raise ValueError("V46 evaluation requires exactly one experiment")
    CalibratedHybridFeatureV41Agent.deployment_route_threshold = float(
        experiments[0]["route_threshold"]
    )


if __name__ == "__main__":
    configure_threshold()
    install_protocol_adapter("hybrid_calibrated_feature_adapter_v19")
    base.VisualAgent = CalibratedHybridFeatureV41Agent
    base.main()

#!/usr/bin/env python3
"""Development-only threshold calibration for the frozen V40 checkpoint."""

import os

import evaluate_visual_recovery_ppo as base
import v39_magnitude_gated_agent as v39
from evaluate_v35_visual_recovery_unseen_ood import install_extensions
from evaluate_v39_visual_recovery import install_protocol_adapter


class CalibratedMagnitudeAgent(v39.MagnitudeGatedDenseV19Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.magnitude_threshold = float(os.environ["ATR_V40_MAGNITUDE_THRESHOLD"])


if __name__ == "__main__":
    install_extensions()
    v39.MagnitudeGatedDenseV19Agent = CalibratedMagnitudeAgent
    install_protocol_adapter()
    base.main()

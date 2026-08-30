#!/usr/bin/env python3
"""Diagnose whether V33 pixel-shift failure is routing or synthesis.

This forces every RGB frame through the already-trained canonicalizer.  It is
an ineligible post-gate mechanism diagnostic, not a candidate policy result.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import evaluate_v33_visual_recovery as v33
import evaluate_visual_recovery_ppo as base
import v33_canonical_view_agent as canonical


class AlwaysCanonicalAgent(canonical.CanonicalizedV19Agent):
    """Use the learned V33 canonicalizer for every observation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.route_threshold = 0.0


if __name__ == "__main__":
    v33.install_protocol_adapter()
    base.VisualAgent = AlwaysCanonicalAgent
    annotated_atomic_json = base.atomic_json

    def diagnostic_atomic_json(payload, path):
        if payload.get("visual_perturbation") != "pixel_shift_right_4":
            raise ValueError("forced V33 routing diagnostic requires pixel_shift_right_4")
        payload["always_canonical_diagnostic"] = True
        payload["diagnostic_claim_boundary"] = (
            "Post-gate mechanism diagnostic only. The learned V33 canonicalizer "
            "is forced on every frame, so this is ineligible for candidate, "
            "allocation, held-out-robustness, or paper performance claims."
        )
        payload["evaluation_source_sha256"]["always_canonical_diagnostic"] = (
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        )
        destination = path.with_name(
            f"{path.stem}_always_canonical_diagnostic{path.suffix}"
        )
        annotated_atomic_json(payload, destination)

    base.atomic_json = diagnostic_atomic_json
    base.main()

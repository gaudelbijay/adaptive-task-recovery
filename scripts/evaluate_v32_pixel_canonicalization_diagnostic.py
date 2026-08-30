#!/usr/bin/env python3
"""Diagnostic upper bound for undoing the observed four-pixel sensor shift.

This is explicitly not a learned-policy result or allocation candidate.  It
tests whether coordinate canonicalization, rather than more action imitation,
is the missing mechanism after V32.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

import evaluate_v32_visual_recovery as v32
import evaluate_visual_recovery_ppo as base


def inverse_right_four(rgb: torch.Tensor) -> torch.Tensor:
    image = rgb.permute(0, 3, 1, 2)
    height, width = image.shape[-2:]
    image = F.pad(image, (4, 4, 4, 4), mode="replicate")
    return image[:, :, 4:4 + height, 8:8 + width].permute(0, 2, 3, 1)


if __name__ == "__main__":
    v32.install_protocol_adapter()
    original_perturbation = base.apply_visual_perturbation
    original_atomic_json = base.atomic_json

    def diagnostic_perturbation(rgb, mode):
        shifted = original_perturbation(rgb, mode)
        return inverse_right_four(shifted) if mode == "pixel_shift_right_4" else shifted

    def diagnostic_atomic_json(payload, path):
        if payload.get("visual_perturbation") != "pixel_shift_right_4":
            raise ValueError("canonicalization diagnostic requires the observed pixel shift")
        payload["pixel_canonicalization_diagnostic"] = True
        payload["diagnostic_claim_boundary"] = (
            "Mechanism upper bound only; deterministic inverse of a known observed "
            "shift, not learned routing, held-out robustness, or candidate evidence."
        )
        payload["evaluation_source_sha256"]["canonicalization_diagnostic"] = (
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        )
        destination = path.with_name(
            f"{path.stem}_canonicalization_diagnostic{path.suffix}"
        )
        original_atomic_json(payload, destination)

    base.apply_visual_perturbation = diagnostic_perturbation
    base.atomic_json = diagnostic_atomic_json
    base.main()

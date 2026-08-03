"""Metrics, splits, and counterfactual tests (docs/03's proposed layout).

`harness.py` (D-042) is the first real implementation of
docs/10-evaluation-and-benchmarks.md's "Statistical protocol": paired
episode seeds across methods, bootstrap confidence intervals. Every
policy comparison in this project before this (D-014, D-025, and every
env-variant test) ran a single seed and asserted a point-in-time result
-- real evidence for a toy case, but not what docs/10 itself specifies a
benchmark comparison needs.
"""

from __future__ import annotations

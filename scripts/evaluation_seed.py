#!/usr/bin/env python3
"""Deterministic held-out seeds compatible with ManiSkill's legacy RNG path."""

from __future__ import annotations

import hashlib


MAX_LEGACY_RANDOMSTATE_SEED = 2**31 - 1
SEED_DERIVATION = (
    "linear_seed_base_plus_training_seed_times_100000_plus_completed;"
    "sha256_31bit_fallback_on_overflow_v2"
)


def heldout_batch_seed(seed_base: int, training_seed: int, completed: int) -> int:
    """Return a paired deterministic seed accepted by NumPy ``RandomState``.

    Existing screening runs retain their predeclared linear seed exactly. New
    confirmation seeds are much larger and overflow ManiSkill's internal
    ``RandomState`` constructor, so only those cases use a domain-separated
    SHA-256 fallback in the portable signed 31-bit range.
    """

    parts = tuple(int(value) for value in (seed_base, training_seed, completed))
    if any(value < 0 for value in parts):
        raise ValueError("held-out seed inputs must be nonnegative")
    linear = parts[0] + parts[1] * 100_000 + parts[2]
    if linear <= MAX_LEGACY_RANDOMSTATE_SEED:
        return linear
    message = "atr-heldout-seed-v2:{}:{}:{}".format(*parts).encode("ascii")
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "big") % (
        MAX_LEGACY_RANDOMSTATE_SEED + 1
    )


def validate_record_batch_seeds(record: dict, expected_episodes: int) -> None:
    """Validate seed provenance, requiring it whenever linear seeding overflows."""

    seed_base = int(record["seed_base"])
    training_seed = int(record["training_seed"])
    requires_fallback = (
        seed_base + training_seed * 100_000 + expected_episodes - 1
        > MAX_LEGACY_RANDOMSTATE_SEED
    )
    batch_seeds = record.get("batch_seeds")
    if batch_seeds is None:
        if requires_fallback:
            raise ValueError("overflowing held-out seed lacks 31-bit seed provenance")
        return
    if record.get("seed_derivation") != SEED_DERIVATION:
        raise ValueError("held-out seed derivation mismatch")
    if (
        not isinstance(batch_seeds, list)
        or not batch_seeds
        or expected_episodes % len(batch_seeds)
    ):
        raise ValueError("held-out batch-seed count is invalid")
    batch_size = expected_episodes // len(batch_seeds)
    expected = [
        heldout_batch_seed(seed_base, training_seed, completed)
        for completed in range(0, expected_episodes, batch_size)
    ]
    if batch_seeds != expected:
        raise ValueError("held-out batch seeds do not match deterministic derivation")
    if len(batch_seeds) != len(set(batch_seeds)):
        raise ValueError("held-out batch seeds contain a collision")

"""Frozen V41 deployment threshold over the audited V40 checkpoint."""

from v39_magnitude_gated_agent import MagnitudeGatedDenseV19Agent


class V41MagnitudeGatedDenseV19Agent(MagnitudeGatedDenseV19Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.magnitude_threshold = 0.015

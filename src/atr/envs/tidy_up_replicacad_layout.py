"""Lightweight shared geometry for the Fetch ReplicaCAD task.

This module intentionally depends only on NumPy so policy and navigation tests
can import the task geometry without importing Torch, ManiSkill, or SAPIEN.
"""

from __future__ import annotations

import numpy as np


_TRAY_POSITION = np.array([-1.0, 0.6, 0.7])
_TRAY_HALF_SIZES = (0.3, 0.3, 0.15)

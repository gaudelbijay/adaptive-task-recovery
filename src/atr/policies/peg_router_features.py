"""Shared hole-frame feature contract for external PegInsertion routing."""

from __future__ import annotations

import torch


GEOMETRY_NAMES = tuple(
    f"hole_frame.{relation}.{axis}"
    for relation in ("peg_to_hole", "blocker_to_hole", "tcp_to_peg", "tcp_to_hole")
    for axis in "xyz"
)
LATERAL_Y_INDICES = (1, 4, 7, 10)


def world_to_local(vector: torch.Tensor, quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vectors by the inverse of a wxyz quaternion."""
    scalar = quaternion_wxyz[:, :1]
    inverse_vector = -quaternion_wxyz[:, 1:]
    cross = 2.0 * torch.linalg.cross(inverse_vector, vector, dim=1)
    return vector + scalar * cross + torch.linalg.cross(inverse_vector, cross, dim=1)


def relative_geometry(raw: torch.Tensor) -> torch.Tensor:
    """Return four relative position vectors in the randomized hole frame."""
    pose = raw.reshape(raw.shape[0], 4, 7)
    peg, hole, blocker, tcp = (pose[:, index, :3] for index in range(4))
    hole_quaternion = pose[:, 1, 3:]
    vectors = (peg - hole, blocker - hole, tcp - peg, tcp - hole)
    return torch.cat(
        tuple(world_to_local(vector, hole_quaternion) for vector in vectors), dim=1,
    )

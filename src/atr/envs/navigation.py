"""Grid-based path planning for Fetch navigation in real ReplicaCAD scenes.

A naive point-and-drive controller gets physically stuck against real
apartment walls/doorways (confirmed directly: a raycast at the exact stuck
position hit a `PhysxRigidStaticComponent` 0.29m away in the direction of
travel — a real wall, not a bug). This builds an occupancy grid via
SAPIEN's own `PhysxCpuSystem.raycast` (no new dependency, unlike Habitat's
own `.navmesh` format, which would need `habitat-sim` — a heavy C++ package
we haven't verified installs on Apple Silicon, and we've been burned by that
exact kind of dependency once already with `mplib`, see D-011) and runs
Dijkstra shortest-path on it. Deliberately simple and inspectable over
pulling in an external planner.

D-087 connects planned 2D waypoints to D-086's geometric effect predictor and
D-083's intent guard through `screen_navigation_path()`. D-091 wires that
decision into the Fetch executor: unsafe routes stop before any drive step and
surface the guard reason rather than being executed. D-092 adds one constrained
replan that inflates the threatened objects into temporary grid obstacles; its
result is independently screened before execution. D-093 makes the execution
outcome explicit and observable: step count, blocked reason, whether replanning
occurred, and the predicted affected-object set. D-101 makes mobile-base
screening explicitly planar so floor-level objects sharing the XY corridor are
not missed because their centers differ from a representative travel height.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra

from atr.constraints.effect_predictor import predict_affected_objects_along_path
from atr.constraints.intent_guard import validate_action
from atr.feasibility.oracle import ObjectState, WorldState
from atr.language.goal_graph import GoalGraph

_NEIGHBOR_OFFSETS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


@dataclass(frozen=True)
class NavigationOutcome:
    """Execution-facing result, including safety adaptation metadata."""

    steps_used: int
    blocked_reason: str | None = None
    safety_screened: bool = True
    replanned: bool = False
    predicted_affected_objects: frozenset[str] = frozenset()


def build_occupancy_grid(
    px,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float = 0.15,
    height: float = 0.5,
    robot_radius: float = 0.3,
    n_rays: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A cell is occupied if a short horizontal raycast (at robot-body
    height, so it catches walls/furniture but not the floor) hits anything
    within `robot_radius` in any of `n_rays` directions."""
    xs = np.arange(x_range[0], x_range[1], resolution)
    ys = np.arange(y_range[0], y_range[1], resolution)
    occupied = np.zeros((len(xs), len(ys)), dtype=bool)
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    dirs = np.stack(
        [np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1
    ).astype(np.float32)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            pos = np.array([x, y, height], dtype=np.float32)
            for d in dirs:
                if px.raycast(pos, d, robot_radius) is not None:
                    occupied[i, j] = True
                    break
    return xs, ys, occupied


def _nearest_free_cell(xs, ys, occupied, xy) -> tuple[int, int]:
    i = int(np.clip(np.searchsorted(xs, xy[0]), 0, len(xs) - 1))
    j = int(np.clip(np.searchsorted(ys, xy[1]), 0, len(ys) - 1))
    if not occupied[i, j]:
        return i, j
    # spiral outward for the nearest free cell if the exact point is occupied
    # (e.g. the target object itself occupies its own grid cell)
    for radius in range(1, max(len(xs), len(ys))):
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < len(xs) and 0 <= nj < len(ys) and not occupied[ni, nj]:
                    return ni, nj
    return i, j


def plan_path(
    xs: np.ndarray, ys: np.ndarray, occupied: np.ndarray, start_xy, goal_xy
) -> list[tuple[float, float]] | None:
    """Dijkstra over an 8-connected grid graph. Returns waypoints from start
    to goal, or None if no path exists (fully walled off)."""
    nx, ny = len(xs), len(ys)

    def idx(i, j):
        return i * ny + j

    n = nx * ny
    graph = lil_matrix((n, n))
    cell_size = xs[1] - xs[0] if len(xs) > 1 else 1.0
    for i in range(nx):
        for j in range(ny):
            if occupied[i, j]:
                continue
            for di, dj in _NEIGHBOR_OFFSETS:
                ni, nj = i + di, j + dj
                if 0 <= ni < nx and 0 <= nj < ny and not occupied[ni, nj]:
                    graph[idx(i, j), idx(ni, nj)] = np.hypot(di, dj) * cell_size

    start_i, start_j = _nearest_free_cell(xs, ys, occupied, start_xy)
    goal_i, goal_j = _nearest_free_cell(xs, ys, occupied, goal_xy)
    start_idx, goal_idx = idx(start_i, start_j), idx(goal_i, goal_j)

    _, predecessors = dijkstra(
        graph.tocsr(), indices=start_idx, return_predecessors=True
    )
    if predecessors[goal_idx] < 0 and goal_idx != start_idx:
        return None

    path_idx = [goal_idx]
    cur = goal_idx
    while cur != start_idx:
        cur = predecessors[cur]
        if cur < 0:
            return None
        path_idx.append(cur)
    path_idx.reverse()
    return [(float(xs[p // ny]), float(ys[p % ny])) for p in path_idx]


def plan_path_avoiding_objects(
    xs: np.ndarray,
    ys: np.ndarray,
    occupied: np.ndarray,
    start_xy,
    goal_xy,
    state: WorldState,
    avoid_objects: frozenset[str],
    clearance_radius: float,
    object_radii: dict[str, float] | None = None,
) -> list[tuple[float, float]] | None:
    """Replan with predicted side-effect objects inflated into obstacles.

    This is the constrained alternate-route search D-091 deliberately left
    open. The original occupancy grid is copied, so episode-level cached map
    state is never mutated. Destroyed/missing objects need no avoidance.
    """
    if clearance_radius < 0:
        raise ValueError("clearance_radius must be non-negative")
    radii = object_radii or {}
    constrained = occupied.copy()
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    for object_id in avoid_objects:
        obj = state.get(object_id)
        if obj is None or not obj.exists or obj.position is None:
            continue
        object_radius = radii.get(object_id, 0.0)
        if object_radius < 0:
            raise ValueError("object radii must be non-negative")
        radius = clearance_radius + object_radius
        distance_sq = (grid_x - obj.position[0]) ** 2 + (grid_y - obj.position[1]) ** 2
        constrained |= distance_sq <= radius ** 2
    return plan_path(xs, ys, constrained, start_xy, goal_xy)


def screen_navigation_path(
    waypoints: list[tuple[float, float]],
    target_object: str,
    graph: GoalGraph,
    state: WorldState,
    travel_height: float,
    clearance_radius: float,
    object_radii: dict[str, float] | None = None,
) -> tuple[bool, str, frozenset[str]]:
    """Screen real `plan_path()` output through D-083's intent guard.

    Returns ``(allowed, reason, affected_objects)`` so an executor can log the
    prediction or request a different route. The intended target is excluded
    from incidental effects because the guard checks it implicitly.
    """
    # D-101: Fetch navigation is planar. Project object centers onto the
    # base-sweep plane before using the general 3D effect predictor; otherwise
    # a floor-level obstacle can share the exact XY corridor but be missed only
    # because its center is vertically far from this representative path
    # height. Preserve the original state for semantic feasibility/guard checks.
    planar_state = {
        object_id: ObjectState(
            exists=obj.exists,
            position=(
                np.array([obj.position[0], obj.position[1], travel_height])
                if obj.exists and obj.position is not None
                else None
            ),
            up_vector=obj.up_vector,
        )
        for object_id, obj in state.items()
    }
    path_xyz = tuple((x, y, travel_height) for x, y in waypoints)
    effects = predict_affected_objects_along_path(
        planar_state,
        path_xyz,
        clearance_radius,
        exclude_objects=frozenset({target_object}),
        object_radii=object_radii,
    )
    allowed, reason = validate_action(
        target_object,
        graph,
        state=state,
        affected_objects=effects,
    )
    return allowed, reason, effects

import ctypes
import numpy as np
import pyastar2d.astar
from enum import IntEnum
from typing import Optional, Tuple


# Define array types
ndmat_f_type = np.ctypeslib.ndpointer(
    dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
ndmat_i1_type = np.ctypeslib.ndpointer(
    dtype=np.int32, ndim=2, flags="C_CONTIGUOUS")
ndmat_i2_type = np.ctypeslib.ndpointer(
    dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
ndmat_i3_type = np.ctypeslib.ndpointer(
    dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
# Define input/output types
pyastar2d.astar.restype = ndmat_i1_type  # Nx2 (i, j) coordinates or None
pyastar2d.astar.argtypes = [
    ndmat_i2_type,   # weights
    ndmat_i3_type,   # edges
    ndmat_f_type,   # transitions
    ctypes.c_int,   # height
    ctypes.c_int,   # width
    ctypes.c_int,   # edge_num
    ctypes.c_int,   # start index in flattened grid
    ctypes.c_int,   # goal index in flattened grid
    ctypes.c_bool,  # allow diagonal
    ctypes.c_int,   # heuristic_override
]

class Heuristic(IntEnum):
    """The supported heuristics."""

    DEFAULT = 0
    ORTHOGONAL_X = 1
    ORTHOGONAL_Y = 2

def astar_path(
        weights: np.ndarray,
        edges: np.ndarray,
        trans: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        allow_diagonal: bool = False,
        heuristic_override: Heuristic = Heuristic.DEFAULT) -> Optional[np.ndarray]:
    """
    Run astar algorithm on 2d weights.

    param np.ndarray weights: A grid of weights e.g. np.ones((10, 10), dtype=np.float32)
    param Tuple[int, int] start: (i, j)
    param Tuple[int, int] goal: (i, j)
    param bool allow_diagonal: Whether to allow diagonal moves
    param Heuristic heuristic_override: Override heuristic, see Heuristic(IntEnum)

    """
    
    # Ensure start is within bounds.
    if (start[0] < 0 or start[0] >= weights.shape[0] or
            start[1] < 0 or start[1] >= weights.shape[1]):
        raise ValueError(f"Start of {start} lies outside grid.")
    # Ensure goal is within bounds.
    if (goal[0] < 0 or goal[0] >= weights.shape[0] or
            goal[1] < 0 or goal[1] >= weights.shape[1]):
        raise ValueError(f"Goal of {goal} lies outside grid.")

    height, width = weights.shape
    start_idx = int(np.ravel_multi_index(start, (height, width)))
    goal_idx = int(np.ravel_multi_index(goal, (height, width)))
    m = edges.shape[0]
    path = pyastar2d.astar.astar(
        weights.flatten().astype(np.intc), edges.flatten().astype(np.intc), trans.flatten().astype(np.float32), height, width, m, start_idx, goal_idx, allow_diagonal,
        int(heuristic_override)
    )
    return path

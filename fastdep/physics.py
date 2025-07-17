"""fastdep.physics - transforms between normalised and physical units."""
from __future__ import annotations

import numpy as np


def denormalise_velocities(
    norm_v: np.ndarray, xyz_min: np.ndarray, xyz_max: np.ndarray
) -> np.ndarray:
    """
    Map velocities stored in `[-1, 1]` back to physical units.

        v = (norm_v + 1) * (max - min) / 2 + min
    """
    return (norm_v + 1.0) * (xyz_max - xyz_min) / 2.0 + xyz_min

"""fastdep.plotting - 2D/3D trajectory visualisations."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for 3‑D backend


def plot_xy_tracks(
    arr: np.ndarray,
    particle_idx: Sequence[int],
    xy: tuple[int, int] = (0, 1),
    *,
    ax=None,
    label: str | None = None,
    alpha: float = 0.4,
    color: str | None = None,
) -> None:
    """Plot 2D projections of many trajectories on a single axes."""
    ax = ax or plt.gca()
    for idx in particle_idx:
        ax.plot(
            arr[idx, :, xy[0]],
            arr[idx, :, xy[1]],
            label=label if idx == particle_idx[0] else None,
            alpha=alpha,
            color=color,
        )
    ax.axis("equal")
    ax.set_xlabel(f"coord {xy[0]}")
    ax.set_ylabel(f"coord {xy[1]}")


def plot_3d_tracks(
    arr: np.ndarray,
    particle_idx: Sequence[int],
    *,
    ax=None,
    label: str | None = None,
    alpha: float = 0.4,
    color: str | None = None,
) -> None:
    ax = ax or plt.figure().add_subplot(111, projection="3d")
    for idx in particle_idx:
        ax.plot(
            arr[idx, :, -3],  # Z
            arr[idx, :, -2],  # Y
            arr[idx, :, -1],  # X
            label=label if idx == particle_idx[0] else None,
            alpha=alpha,
            color=color,
        )
    ax.set_xlabel("Z")
    ax.set_ylabel("Y")
    ax.set_zlabel("X")
    ax.legend()

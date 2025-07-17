"""fastdep.io - I/O utilities for trajectory projects."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pandas as pd


log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  HDF5 helpers
# --------------------------------------------------------------------------- #
def h5_summary(path: Path) -> None:
    """Pretty-print datasets, shapes and dtypes contained in an HDF5 file."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        log.info("== HDF5 summary :: %s ==", path)
        for key, obj in f.items():
            if isinstance(obj, h5py.Dataset):
                log.info(" %-15s  %s  %s", key, obj.shape, obj.dtype)
            else:
                log.info(" %-15s  (group)", key)


def save_h5_dataset(
    out_path: Path,
    train: np.ndarray,
    xyz_min: np.ndarray,
    xyz_max: np.ndarray,
    overwrite: bool = False,
) -> None:
    """Create an HDF5 file with `min`, `max`, `train` datasets."""
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists (pass overwrite=True).")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("min", data=xyz_min.astype(np.float32))
        f.create_dataset("max", data=xyz_max.astype(np.float32))
        f.create_dataset("train", data=train.astype(np.float32))
    log.info("Wrote %s [train %s]", out_path, train.shape)


# --------------------------------------------------------------------------- #
#  CSV ↔ Parquet with automatic caching
# --------------------------------------------------------------------------- #
def load_track_file(csv_path: Path) -> pd.DataFrame:
    """
    Load a Star‑CCM+ *track* file.  If a `.parquet` twin exists it is used,
    otherwise the CSV is converted and cached for next time.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    parquet_path = csv_path.with_suffix(".parquet")
    if parquet_path.is_file():
        log.info("Reading cached parquet %s", parquet_path)
        return pd.read_parquet(parquet_path, engine="fastparquet")

    log.info("Reading CSV %s (first time – will cache parquet)", csv_path)
    df = pd.read_csv(csv_path)
    df.to_parquet(parquet_path, compression=None)
    return df


# --------------------------------------------------------------------------- #
#  NumPy helpers
# --------------------------------------------------------------------------- #
def load_npy(path: Path, moveaxis: Tuple[int, int] | None = None) -> np.ndarray:
    """Load ``.npy`` and optionally move an axis (handy for [P,T,C] vs [T,P,C])."""
    arr = np.load(path)
    if moveaxis is not None:
        arr = np.moveaxis(arr, *moveaxis)
    log.debug("Loaded %s  →  %s", path, arr.shape)
    return arr

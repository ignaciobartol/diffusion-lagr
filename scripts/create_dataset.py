#!/usr/bin/env python
"""
create_dataset.py - build a trimmed HDF5 (min / max / train) from

  • Star-CCM+ track CSV  (auto-caches parquet)
  • NumPy array (.npy)   (particle trajectories)

Example
-------
    python scripts/create_dataset.py \
        --csv   data/bb-part-0.42mum.csv \
        --npy   data/bb-part-0.42.npy   \
        --out   datasets/bb-part-0.42.h5 \
        --train-particles 512 \
        --train-timesteps 1024
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from fastdep.io import load_npy, load_track_file, save_h5_dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="Create trimmed HDF5 dataset.")
parser.add_argument("--csv", required=True, type=Path, help="Star-CCM+ CSV file")
parser.add_argument("--npy", required=True, type=Path, help="Raw NumPy trajectories")
parser.add_argument("--out", required=True, type=Path, help="Output .h5 path")
parser.add_argument("--train-particles", type=int, default=512)
parser.add_argument("--train-timesteps", type=int, default=1024)
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()

# --------------------------------------------------------------------------- #
#  1) Load CSV → DataFrame
# --------------------------------------------------------------------------- #
df = load_track_file(args.csv)
df.sort_values(by=["Track: Parcel Index", "Track: Time (s)"], inplace=True)

xyz_min = np.asarray(df[["Track: Position[X] (m)",
                         "Track: Position[Y] (m)",
                         "Track: Position[Z] (m)"]].min().values)
xyz_max = np.asarray(df[["Track: Position[X] (m)",
                         "Track: Position[Y] (m)",
                         "Track: Position[Z] (m)"]].max().values)
log.info("xyz‑min %s", xyz_min)
log.info("xyz‑max %s", xyz_max)

# --------------------------------------------------------------------------- #
#  2) Load NumPy and slice for training subset
# --------------------------------------------------------------------------- #
raw = load_npy(args.npy, moveaxis=(1, 0))  # (particles, timesteps, 3)
train = raw[: args.train_particles, : args.train_timesteps, -3:]
log.info("train subset %s", train.shape)

# --------------------------------------------------------------------------- #
#  3) Save to HDF5
# --------------------------------------------------------------------------- #
save_h5_dataset(
    out_path=args.out,
    train=train,
    xyz_min=xyz_min,
    xyz_max=xyz_max,
    overwrite=args.overwrite,
)

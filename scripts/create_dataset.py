#!/usr/bin/env python
"""
create_dataset.py - build a trimmed HDF5 (min / max / train) from

  • Star-CCM+ track CSV  (auto-caches parquet)
  • NumPy array (.npy)   (particle trajectories)

Example
-------
    python scripts/create_dataset.py \
        --csv   ../Simulations-INL/starccm-mesh/sm-sim-part/sm-part-0.42mum.csv \
        --out   datasets/sm-part-16384.h5 \
        --train-particles 16384 \
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
# parser.add_argument("--npy", required=True, type=Path, help="Raw NumPy trajectories")
parser.add_argument("--out", required=True, type=Path, help="Output .h5 path")
parser.add_argument("--train-particles", type=int, default=2048)
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
log.info("xyz-min %s", xyz_min)
log.info("xyz-max %s", xyz_max)


print("------------Analizing {}---------------".format(args.csv))

try:
    df = df.sort_values(by=['Track: Parcel Index', 'Track: Time (s)'])
    #Get only the last interaction of the particle (Although it says 'first' is the last...)
    z_max = np.amax(df["Track: Position[Z] (m)"])
except KeyError:    
    z_max = np.amax(df["Position[Z] (m)"])

# Find maximum length from all the tracks
group_sizes = df.groupby(['Track: Parcel Index']).size()
longest_length = group_sizes.max()

unique_parcels = df['Track: Parcel Index'].unique()
N = len(unique_parcels)

# Filter the DataFrame to include only the first N parcel indices
filtered_parcels = unique_parcels[:N]
filtered_df = df[df['Track: Parcel Index'].isin(filtered_parcels)]
col_names = list(df.columns.values)

# Find maximum length from all the tracks
group_sizes = df.groupby('Track: Parcel Index').size()
longest_length = group_sizes.max()

# Define other dimensions
M = longest_length
K = len(col_names)  # X, Y, Z, Stuck Mark, etc...

import h5py
import numpy as np

def _col(df, name_variants):
    """Return the first existing column name from a list of candidates."""
    for n in name_variants:
        if n in df.columns:
            return n
    raise KeyError(f"None of these columns exist: {name_variants}")

def save_h5_dataset_streaming(
    out_path: Path,
    df,
    xyz_min: np.ndarray,
    xyz_max: np.ndarray,
    train_particles: int,
    train_timesteps: int,
    overwrite: bool = False,
    compression: str | None = "gzip",
    compression_level: int = 4,
) -> None:
    """
    Stream StarCCM+ tracks from a DataFrame into HDF5 without building a giant numpy array.

    Output datasets:
      - min: (3,)
      - max: (3,)
      - train: (P, T, 3) float32s
    """
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists (pass overwrite=True).")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Column resolution (handles both "Track: ..." and non-prefixed variants)
    parcel_col = _col(df, ["Track: Parcel Index", "Parcel Index"])
    time_col   = _col(df, ["Track: Time (s)", "Time (s)"])
    x_col      = _col(df, ["Track: Position[X] (m)", "Position[X] (m)"])
    y_col      = _col(df, ["Track: Position[Y] (m)", "Position[Y] (m)"])
    z_col      = _col(df, ["Track: Position[Z] (m)", "Position[Z] (m)"])

    # Sort once so each group is time-ordered
    df = df.sort_values(by=[parcel_col, time_col], kind="mergesort")

    P = int(train_particles)
    T = int(train_timesteps)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("min", data=xyz_min.astype(np.float32))
        f.create_dataset("max", data=xyz_max.astype(np.float32))

        # Chunk per-particle to keep writes efficient
        chunks = (1, T, 3)
        ds = f.create_dataset(
            "train",
            shape=(P, T, 3),
            dtype=np.float32,
            chunks=chunks,
            compression=compression,
            compression_opts=(compression_level if compression else None),
        )

        # Optional metadata (nice for reproducibility)
        ds.attrs["train_particles"] = P
        ds.attrs["train_timesteps"] = T
        ds.attrs["columns"] = "x,y,z"
        # ds.attrs["dt_particle"] = 0.00125  # if you want to store this here

        written = 0

        # Iterate particle tracks; stop after P
        for parcel_id, g in df.groupby(parcel_col, sort=False):
            if written >= P:
                break

            # Extract positions as float32 (L,3)
            xyz = g[[x_col, y_col, z_col]].to_numpy(dtype=np.float32)
            L = xyz.shape[0]
            if L == 0:
                continue

            # Pad/truncate to length T using last-value hold (mode='edge')
            if L >= T:
                xyz_T = xyz[:T, :]
            else:
                pad = T - L
                xyz_T = np.pad(xyz, ((0, pad), (0, 0)), mode="edge")

            # Write directly to HDF5
            ds[written, :, :] = xyz_T
            written += 1

        # If we wrote fewer than requested, shrink dataset (optional)
        if written < P:
            ds.resize((written, T, 3))

    log.info("Wrote %s [train (%d, %d, 3)]", out_path, written, T)

save_h5_dataset_streaming(
    out_path=args.out,
    df=df,
    xyz_min=xyz_min,
    xyz_max=xyz_max,
    train_particles=args.train_particles,
    train_timesteps=args.train_timesteps,
    overwrite=args.overwrite,
)

# dataset = np.full((M, N, K), np.nan)

# for parcel_index in filtered_parcels:
#     parcel_data = filtered_df[filtered_df['Track: Parcel Index'] == parcel_index]
#     for i in range(K):
#         end = parcel_data.groupby(['Track: Parcel Index']).size()
#         if len(end) == 1:
#             # To increase the size by:
#             isb = M - end.max()
#             dataset[:,parcel_index-1, i] = np.pad(parcel_data[col_names[i]].to_numpy(),
#                                                    pad_width = (0, isb),
#                                                    mode = 'edge')
#         else:
#             print(len(end))

# # # --------------------------------------------------------------------------- #
# # #  2) Load NumPy and slice for training subset
# # # --------------------------------------------------------------------------- #
# # raw = load_npy(args.npy, moveaxis=(1, 0))  # (particles, timesteps, 3)
# # train_npy = raw[: args.train_particles, : args.train_timesteps, -3:]
# # log.info("train subset %s", train_npy.shape)

# --------------------------------------------------------------------------- #
#  3) Save to HDF5
# --------------------------------------------------------------------------- #
# save_h5_dataset(
#     out_path=args.out,
#     train=train_npy,
#     xyz_min=xyz_min,
#     xyz_max=xyz_max,
#     overwrite=args.overwrite,
# )

#!/usr/bin/env python3
"""
Convert a **-part-*.npy file into the diffusion-lagr HDF5 format.

This script:
1. Loads a .npy file containing particle trajectories.
2. (Optional) Swaps axes to match (time, particles, channels).
3. Computes min/max either from an optional CSV or directly from the .npy.
4. Writes 'train', 'min', and 'max' datasets to an .h5 file.

Example:
    python scripts/convert_bb_part_to_h5.py \
        /path/to/bb-part-0.42.npy \
        datasets/bb-part-0.42.h5 \
        --csv /path/to/bb-part-0.42mum.csv \
        --swap-axes
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np


CSV_MIN_MAX_COLUMNS = [
    ("Track: Position[X] (m)", "Track: Position[Y] (m)", "Track: Position[Z] (m)"),
    ("Position[X] (m)", "Position[Y] (m)", "Position[Z] (m)"),
]


def load_min_max_from_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to compute min/max from CSV.") from exc

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    for x_col, y_col, z_col in CSV_MIN_MAX_COLUMNS:
        if x_col in df.columns and y_col in df.columns and z_col in df.columns:
            min_vals = np.array([df[x_col].min(), df[y_col].min(), df[z_col].min()])
            max_vals = np.array([df[x_col].max(), df[y_col].max(), df[z_col].max()])
            return min_vals, max_vals

    raise KeyError(
        "Could not find position columns in CSV. Expected one of: "
        + ", ".join([str(cols) for cols in CSV_MIN_MAX_COLUMNS])
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert bb-part .npy files into diffusion-lagr .h5 datasets."
    )
    parser.add_argument("input_npy", help="Path to the input .npy file.")
    parser.add_argument("output_h5", help="Path to the output .h5 file.")
    parser.add_argument(
        "--csv",
        dest="csv_path",
        help="Optional CSV with Track/Position columns to compute min/max if .npy is normalized.",
    )
    parser.add_argument(
        "--swap-axes",
        action="store_true",
        help="Swap axis 1 to axis 0 to get (time, particles, channels).",
    )
    parser.add_argument(
        "--n-timesteps",
        type=int,
        default=1024,
        help="Number of timesteps to keep from the .npy array.",
    )
    parser.add_argument(
        "--n-particles",
        type=int,
        default=16384,
        help="Number of particles to keep from the .npy array.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input_npy):
        print(f"Error: input file not found: {args.input_npy}")
        return 1

    print(f"Loading {args.input_npy}...")
    try:
        data_ar = np.load(args.input_npy)
    except Exception as exc:
        print(f"Error loading numpy file: {exc}")
        return 1

    print(f"Original shape: {data_ar.shape}")

    if args.swap_axes:
        print("Swapping axis 1 to axis 0...")
        data_ar = np.moveaxis(data_ar, 1, 0)
        print(f"New shape: {data_ar.shape}")

    if data_ar.ndim != 3:
        print(f"Warning: expected 3D array, got shape {data_ar.shape}")
    if data_ar.shape[-1] < 3:
        print(f"Warning: expected >=3 channels, got {data_ar.shape[-1]}")

    if args.csv_path:
        print(f"Computing min/max from CSV: {args.csv_path}")
        min_vals, max_vals = load_min_max_from_csv(args.csv_path)
    else:
        print("Computing min/max from numpy array...")
        min_vals = np.min(data_ar[..., -3:], axis=(0, 1))
        max_vals = np.max(data_ar[..., -3:], axis=(0, 1))

    print(f"Min coords: {min_vals}")
    print(f"Max coords: {max_vals}")

    out_dir = os.path.dirname(args.output_h5)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    train_data = data_ar[: args.n_timesteps, : args.n_particles, -3:]

    print(f"Writing {args.output_h5}...")
    with h5py.File(args.output_h5, "w") as h5f:
        h5f.create_dataset("min", data=min_vals.astype(np.float32))
        h5f.create_dataset("max", data=max_vals.astype(np.float32))
        h5f.create_dataset("train", data=train_data.astype(np.float32))

    print("Conversion successful.")
    print(f"train shape: {train_data.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

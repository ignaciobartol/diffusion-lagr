#!/usr/bin/env python3
"""
Script to convert a raw NumPy particle array (.npy) into the HDF5 format (.h5)
required for training the diffusion-lagr model.

This script:
1. Loads a .npy file.
2. (Optional) Permutes axes to make shape (N_particles, Time_steps, 3).
3. Calculates global min/max position coordinates for normalization.
4. Saves 'train', 'min', and 'max' datasets to an .h5 file.

Usage:
    python scripts/convert_np_to_hd5.py path/to/data.npy datasets/output_name.h5
"""

import argparse
import os
import sys
import numpy as np
import h5py

def main():
    parser = argparse.ArgumentParser(description="Convert .npy particle tracks to .h5 dataset.")
    parser.add_argument("input_npy", type=str, help="Path to the input .npy file.")
    parser.add_argument("output_h5", type=str, help="Path to the output .h5 file.")
    parser.add_argument("--n_particles", type=int, default=1024,
                        help="Number of particles in the dataset.")
    parser.add_argument("--n_timesteps", type=int, default=512,
                        help="Number of time steps in the dataset.")
    parser.add_argument("--swap_axes", action="store_true", 
                        help="If set, swaps axis 0 and 1. Use if input is (Time, Particles, 3).")
    
    args = parser.parse_args()

    # 1. Load Data
    if not os.path.isfile(args.input_npy):
        print(f"Error: Input file '{args.input_npy}' not found.")
        sys.exit(1)

    print(f"Loading {args.input_npy}...")
    try:
        data_ar = np.load(args.input_npy)
    except Exception as e:
        print(f"Error loading numpy file: {e}")
        sys.exit(1)

    print(f"Original shape: {data_ar.shape}")

    # The training loop expects (Particles, Time, Channels) in the H5 file.
    if args.swap_axes:
        print("Swapping axes 0 and 1...")
        data_ar = np.moveaxis(data_ar, 1, 0)
        print(f"New shape: {data_ar.shape}")

    if data_ar.ndim != 3:
        print(f"Warning: Expected 3 dimensions (Particles, Time, Channels), got {data_ar.ndim}.")
    if data_ar.shape[-1] != 3:
        print(f"Warning: Expected last dimension to be 3 (x, y, z), got {data_ar.shape[-1]}.")

    print("Calculating statistics...")
    min_vals = np.min(data_ar, axis=(0, 1))
    max_vals = np.max(data_ar, axis=(0, 1))

    print(f"  Min coordinates (x,y,z): {min_vals}")
    print(f"  Max coordinates (x,y,z): {max_vals}")

    out_dir = os.path.dirname(args.output_h5)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"Writing to {args.output_h5}...")
    
    data_ar = data_ar.astype(np.float32)
    min_vals = min_vals.astype(np.float32)
    max_vals = max_vals.astype(np.float32)

    with h5py.File(args.output_h5, 'w') as h5f:
        h5f.create_dataset('min', data=min_vals)
        h5f.create_dataset('max', data=max_vals)
        h5f.create_dataset('train', 
                           data=data_ar[0:args.n_timesteps, 0:args.n_particles, -3:])

    print("Verification:")
    with h5py.File(args.output_h5, 'r') as h5f:
        print(f"  'train' shape: {h5f['train'].shape}")
        print(f"  'min' values:  {h5f['min'][:]}")
        print(f"  'max' values:  {h5f['max'][:]}")
    
    print("Conversion successful.")

if __name__ == "__main__":
    main()
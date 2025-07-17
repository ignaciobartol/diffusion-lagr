#!/usr/bin/env python
"""
compare_trajectories.py - overlay CFPD vs diffusion predictions (2D + 3D).

Example
-------
    python scripts/compare_trajectories.py \
        --gt-npy   datasets/bb-part-0.42.npy \
        --pred-npz results/samples_128x1024x3.npz \
        --out      figs/compare.png
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from fastdep.plotting import plot_xy_tracks, plot_3d_tracks

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Compare GT vs diffusion samples.")
parser.add_argument("--gt-npy", required=True, type=Path)
parser.add_argument("--pred-npz", required=True, type=Path)
parser.add_argument("--out", type=Path, help="Save figure instead of showing")
parser.add_argument("--n-train", type=int, default=512)
parser.add_argument("--n-sample", type=int, default=128)
args = parser.parse_args()

gt = np.moveaxis(np.load(args.gt_npy), 1, 0)       # (P,T,3)
pred = np.load(args.pred_npz)["arr_0"]             # (P,T,3)

fig = plt.figure(figsize=(20, 5))
xy_pairs = [(-3, -2), (-3, -1), (-2, -1)]  # (Z,Y), (Z,X), (Y,X)
labels = [("X", "Y"), ("X", "Z"), ("Y", "Z")]

for i, (xy, (xlabel, ylabel)) in enumerate(zip(xy_pairs, labels), start=1):
    ax = fig.add_subplot(1, 4, i)
    plot_xy_tracks(gt[: args.n_train], range(args.n_train), xy=xy, ax=ax,
                   alpha=0.2, color="red", label="Ground Truth")
    plot_xy_tracks(pred[: args.n_sample], range(args.n_sample), xy=xy, ax=ax,
                   alpha=0.4, color="blue", label="Diffusion")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{xlabel}-{ylabel}")

ax3d = fig.add_subplot(1, 4, 4, projection="3d")
plot_3d_tracks(gt[: args.n_train], range(args.n_train), ax=ax3d,
               alpha=0.2, color="red", label="Ground Truth")
plot_3d_tracks(pred[: args.n_sample], range(args.n_sample), ax=ax3d,
               alpha=0.4, color="blue", label="Diffusion")
ax3d.set_title("3‑D")
handles = [
    Line2D([0], [0], color="blue", linestyle="--", label="Diffusion Model"),
    Line2D([0], [0], color="red", linestyle="-", label="Ground Truth"),
]
fig.legend(handles=handles, loc="upper right")
fig.legend(handles=handles, loc="upper right")

fig.tight_layout()
if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    log.info("Figure saved to %s", args.out)
else:
    plt.show()

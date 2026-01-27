#!/usr/bin/env python
"""
Analyze progress snapshots produced by turb_train_monitor.py.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch as th

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    str2bool,
)


SAMPLE_REGEX = re.compile(r"samples_step_(\d+)\.npz")
ENCODER_REGEX = re.compile(r"encoder_step_(\d+)\.npz")


def _load_geometry_paths_from_manifest(manifest_path: str) -> List[str]:
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    return [entry["geo_path"] for entry in manifest]


def _summarize_samples(samples: np.ndarray) -> Dict[str, float]:
    stats = {
        "min": float(samples.min()),
        "max": float(samples.max()),
        "mean": float(samples.mean()),
        "std": float(samples.std()),
    }
    if samples.ndim >= 3:
        channel_means = samples.mean(axis=(0, 1))
        channel_stds = samples.std(axis=(0, 1))
        for idx, value in enumerate(channel_means):
            stats[f"channel_{idx}_mean"] = float(value)
        for idx, value in enumerate(channel_stds):
            stats[f"channel_{idx}_std"] = float(value)
    return stats


def _write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys))
        f.write("\n")
        for row in rows:
            f.write(",".join(str(row.get(key, "")) for key in keys))
            f.write("\n")


def _plot_encoder_projection(proj: np.ndarray, labels: List[str], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        logger.log("matplotlib not available; skipping encoder plot.")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(proj[:, 0], proj[:, 1], color="purple")
    for idx, label in enumerate(labels):
        ax.text(proj[idx, 0], proj[idx, 1], label, fontsize=8)
    ax.set_title("Geometry Encoder Space (PCA)")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_samples(
    samples: np.ndarray,
    out_path: Path,
    max_samples: int,
    plot_dims: Tuple[int, int],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        logger.log("matplotlib not available; skipping sample plot.")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = min(max_samples, samples.shape[0])
    fig, ax = plt.subplots(figsize=(6, 6))
    if samples.ndim >= 3 and samples.shape[-1] >= 2:
        dim_x, dim_y = plot_dims
        if dim_x >= samples.shape[-1] or dim_y >= samples.shape[-1]:
            logger.log("plot_dims exceed sample channel count; skipping sample plot.")
            plt.close(fig)
            return
        for i in range(num_samples):
            ax.plot(
                samples[i, :, dim_x],
                samples[i, :, dim_y],
                color="blue",
                alpha=0.3,
            )
        ax.set_xlabel(f"dim {dim_x}")
        ax.set_ylabel(f"dim {dim_y}")
        ax.set_title("Sample Trajectories")
        ax.axis("equal")
    else:
        time = np.arange(samples.shape[1]) if samples.ndim >= 2 else np.arange(samples.shape[0])
        if samples.ndim >= 2:
            for i in range(num_samples):
                ax.plot(time, samples[i, :, 0], color="blue", alpha=0.3)
        else:
            ax.plot(time, samples, color="blue", alpha=0.7)
        ax.set_xlabel("t")
        ax.set_ylabel("value")
        ax.set_title("Sample Trajectories")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)

def _encode_geometry(model, geometry_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    embeddings = []
    for geo_path in geometry_paths:
        data = np.load(geo_path)
        grid = th.from_numpy(data["binary"]).unsqueeze(0).unsqueeze(0).float().to(device)
        with th.no_grad():
            if getattr(model, "geometry_encoder_type", "deterministic") == "variational":
                emb, _ = model.geo_encoder(grid, sample=False)
            else:
                emb = model.geo_encoder(grid)
        embeddings.append(emb.detach().cpu().numpy()[0])
    embeddings = np.stack(embeddings, axis=0)
    emb = embeddings - embeddings.mean(axis=0, keepdims=True)
    cov = np.cov(emb, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order[:2]]
    proj = emb @ eigvecs
    return embeddings, proj


def _load_model(args) -> th.nn.Module:
    model, _ = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    state_dict = dist_util.load_state_dict(args.model_path, map_location=dist_util.dev())
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress_dir", default="checkpoints/progress")
    parser.add_argument("--output_dir", default="checkpoints/progress_analysis")
    parser.add_argument("--model_path", default="")
    parser.add_argument("--encoder_manifest", default="") # Same as train_manifest, to locate geoms
    parser.add_argument("--plot_samples", default=True, type=str2bool)
    parser.add_argument("--plot_max_samples", default=64, type=int)
    parser.add_argument("--plot_dims", default="0,1")
    defaults = model_and_diffusion_defaults()
    add_dict_to_argparser(parser, defaults)
    args = parser.parse_args()

    progress_dir = Path(args.progress_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rows: List[Dict[str, float]] = []
    for sample_file in sorted(progress_dir.glob("samples_step_*.npz")):
        match = SAMPLE_REGEX.match(sample_file.name)
        if not match:
            continue
        step = int(match.group(1))
        if step % 5000 != 0:
            continue
        samples = np.load(sample_file)["arr_0"]
        stats = _summarize_samples(samples)
        stats["step"] = step
        sample_rows.append(stats)
        if args.plot_samples:
            dim_parts = args.plot_dims.split(",")
            if len(dim_parts) >= 2:
                plot_dims = (int(dim_parts[0]), int(dim_parts[1]))
            else:
                plot_dims = (0, 1)
            _plot_samples(
                samples,
                output_dir / f"samples_step_{step:06d}.png",
                max_samples=args.plot_max_samples,
                plot_dims=plot_dims,
            )

    _write_csv(output_dir / "sample_summary.csv", sample_rows)

    for encoder_file in sorted(progress_dir.glob("encoder_step_*.npz")):
        match = ENCODER_REGEX.match(encoder_file.name)
        if not match:
            continue
        step = int(match.group(1))
        data = np.load(encoder_file, allow_pickle=True)
        embeddings = data["embeddings"]
        proj = data["proj"] if "proj" in data else embeddings[:, :2]
        geometry_paths = data.get("geometry_paths", [])
        labels = [Path(path).stem for path in geometry_paths]
        out_path = output_dir / f"encoder_step_{step:06d}.png"
        _plot_encoder_projection(proj, labels, out_path)

    if args.encoder_manifest and args.model_path:
        geometry_paths = _load_geometry_paths_from_manifest(args.encoder_manifest)
        model = _load_model(args)
        embeddings, proj = _encode_geometry(model, geometry_paths)
        np.savez(
            output_dir / "encoder_recompute.npz",
            embeddings=embeddings,
            proj=proj,
            geometry_paths=geometry_paths,
        )
        labels = [Path(path).stem for path in geometry_paths]
        _plot_encoder_projection(proj, labels, output_dir / "encoder_recompute.png")
    elif args.encoder_manifest and not args.model_path:
        logger.log("encoder_manifest provided without model_path; skipping encoder recompute.")

    # Free memory from GPU to load next model


if __name__ == "__main__":
    main()

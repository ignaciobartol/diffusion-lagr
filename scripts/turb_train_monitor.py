#!/usr/bin/env python
"""
Train a diffusion model with GPU/memory monitoring and periodic progress snapshots.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch as th

from guided_diffusion import dist_util, logger
from guided_diffusion.geometry_util import geometry_guidance_fn
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.turb_datasets import load_data
from guided_diffusion.train_util import TrainLoop


def _init_nvml():
    try:
        import pynvml
    except Exception:
        return None
    try:
        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


def _gpu_metrics(pynvml_mod, device_index: int) -> Dict[str, float]:
    if pynvml_mod is None:
        return {}
    try:
        handle = pynvml_mod.nvmlDeviceGetHandleByIndex(device_index)
        util = pynvml_mod.nvmlDeviceGetUtilizationRates(handle)
        meminfo = pynvml_mod.nvmlDeviceGetMemoryInfo(handle)
        return {
            "gpu_util": float(util.gpu),
            "gpu_mem_util": float(util.memory),
            "gpu_mem_used": float(meminfo.used),
            "gpu_mem_total": float(meminfo.total),
        }
    except Exception:
        return {}


def _torch_mem_metrics(device: th.device) -> Dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "cuda_allocated": float(th.cuda.memory_allocated(device)),
        "cuda_reserved": float(th.cuda.memory_reserved(device)),
        "cuda_max_allocated": float(th.cuda.max_memory_allocated(device)),
        "cuda_max_reserved": float(th.cuda.max_memory_reserved(device)),
    }


def _write_metrics_row(csv_path: Path, row: Dict[str, float]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _load_geometry_for_guidance(geometry_path: str, device: th.device):
    geo_data = np.load(geometry_path)
    sdf = th.from_numpy(geo_data["sdf"]).unsqueeze(0).unsqueeze(0).to(device)
    origin = th.from_numpy(geo_data["origin"]).to(device)
    spacing = th.from_numpy(geo_data["spacing"]).to(device)
    binary = th.from_numpy(geo_data["binary"]).unsqueeze(0).unsqueeze(0).float().to(device)
    return sdf, origin, spacing, binary


def _plot_progress(gt_tracks: np.ndarray, pred_tracks: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    n_gt = min(64, gt_tracks.shape[0])
    n_pred = min(64, pred_tracks.shape[0])
    for i in range(n_gt):
        ax.plot(gt_tracks[i, :, 0], gt_tracks[i, :, 1], color="red", alpha=0.2)
    for i in range(n_pred):
        ax.plot(pred_tracks[i, :, 0], pred_tracks[i, :, 1], color="blue", alpha=0.3)
    ax.set_title("GT vs Pred (XY)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _pca_project(embeddings: np.ndarray) -> np.ndarray:
    emb = embeddings - embeddings.mean(axis=0, keepdims=True)
    cov = np.cov(emb, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order[:2]]
    return emb @ eigvecs


def _save_encoder_progress(model, geometry_paths: List[str], out_path: Path) -> None:
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
    proj = _pca_project(embeddings) if embeddings.shape[0] > 1 else embeddings[:, :2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, embeddings=embeddings, proj=proj, geometry_paths=geometry_paths)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(proj[:, 0], proj[:, 1], color="purple")
    for i, path in enumerate(geometry_paths):
        ax.text(proj[i, 0], proj[i, 1], Path(path).stem, fontsize=8)
    ax.set_title("Geometry Encoder Space (PCA)")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _load_geometry_paths_from_manifest(manifest_path: str) -> List[str]:
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    return [entry["geo_path"] for entry in manifest]


def main() -> None:
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        class_cond=args.class_cond,
        normalize_positions=args.normalize_positions,
        normalization_space=args.normalization_space,
        cache_h5=args.cache_h5,
    )

    loop = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    )

    device = dist_util.dev()
    pynvml_mod = _init_nvml()
    device_index = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    metrics_path = Path(args.log_dir) / "metrics" / "gpu_metrics.csv"

    gt_tracks = None
    if args.gt_npy:
        gt_tracks = np.moveaxis(np.load(args.gt_npy), 1, 0)

    geometry_paths = []
    if args.encoder_manifest:
        geometry_paths = _load_geometry_paths_from_manifest(args.encoder_manifest)

    sdf = origin = spacing = binary_grid = None
    if args.geometry_path:
        sdf, origin, spacing, binary_grid = _load_geometry_for_guidance(
            args.geometry_path, device
        )

    def cond_fn(x, t, y=None, **kwargs):
        return geometry_guidance_fn(
            x_t=x,
            t=t,
            sdf_grid=sdf,
            origin=origin,
            spacing=spacing,
            guidance_scale=args.guidance_scale,
            coord_space=args.coord_space,
        )

    logger.log("training with monitoring...")
    while (
        not args.lr_anneal_steps
        or loop.step + loop.resume_step < args.lr_anneal_steps
    ):
        batch, cond = next(data)
        loop.run_step(batch, cond)

        if loop.step % args.log_interval == 0:
            logger.dumpkvs()

        if loop.step % args.metrics_interval == 0:
            row = {"step": loop.step}
            row.update(_torch_mem_metrics(device))
            row.update(_gpu_metrics(pynvml_mod, device_index))
            _write_metrics_row(metrics_path, row)

        if args.snapshot_interval and loop.step % args.snapshot_interval == 0:
            if args.geometry_path:
                model_kwargs = {"geometry_grid": binary_grid.repeat(args.eval_batch, 1, 1, 1, 1)}
            else:
                model_kwargs = {}

            sample = diffusion.p_sample_loop(
                model,
                (args.eval_batch, args.in_channels, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn if args.geometry_path else None,
                device=device,
            )
            sample = sample.clamp(-1, 1).permute(0, 2, 1).contiguous().cpu().numpy()
            out_dir = Path(args.log_dir) / "progress"
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez(out_dir / f"samples_step_{loop.step:06d}.npz", sample)

            if gt_tracks is not None:
                _plot_progress(
                    gt_tracks[: args.eval_batch],
                    sample,
                    out_dir / f"compare_step_{loop.step:06d}.png",
                )

            if geometry_paths:
                _save_encoder_progress(
                    model,
                    geometry_paths,
                    out_dir / f"encoder_step_{loop.step:06d}.npz",
                )

        if loop.step % args.save_interval == 0:
            loop.save()

        if os.environ.get("DIFFUSION_TRAINING_TEST", "") and loop.step > 0:
            break

        loop.step += 1

    if (loop.step - 1) % args.save_interval != 0:
        loop.save()


def create_argparser():
    defaults = dict(
        dataset_path="",
        dataset_name="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        log_dir="checkpoints",
        metrics_interval=10,
        snapshot_interval=1000,
        eval_batch=16,
        gt_npy="",
        geometry_path="",
        guidance_scale=1.0,
        coord_space="normalized",
        normalize_positions=True,
        normalization_space="normalized",
        cache_h5=True,
        clip_denoised=True,
        encoder_manifest="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

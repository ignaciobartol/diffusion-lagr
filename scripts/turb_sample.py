"""
Generate a large batch of Lagrangian trajectories from a model and save them as a large
numpy array. This can be used to produce samples for statistical evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from guided_diffusion.geometry_util import geometry_guidance_fn

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.results_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    logger.log("model and diffusion created")
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    logger.log("model loaded from", args.model_path)
    logger.log("model parameters:", sum(p.numel() for p in model.parameters()))
    logger.log("model on device:", dist_util.dev())
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    logger.log("model converted to fp16" if args.use_fp16 else "model in fp32")

    logger.log("loading geometries for guidance...")
    # Load the .npz created by process_geometry.py
    geo_data = np.load(args.geometry_path)
    sdf_cpu = th.from_numpy(geo_data["sdf"]).unsqueeze(0).unsqueeze(0)
    origin_cpu = th.from_numpy(geo_data["origin"])
    spacing_cpu = th.from_numpy(geo_data["spacing"])

    sdf_dev = sdf_cpu.to(dist_util.dev())
    origin_dev = origin_cpu.to(dist_util.dev())
    spacing_dev = spacing_cpu.to(dist_util.dev())

    binary_grid = th.from_numpy(geo_data["binary"]).unsqueeze(0).unsqueeze(0).float()
    binary_grid_dev = binary_grid.to(dist_util.dev())
    logger.log("geometries loaded")

    # Code to define cond_fn for geometry guidance closure
    def cond_fn(x, t, y=None, **kwargs):
        """
        Wrapper that freezes the geometry arguments fr the diffusion loop.
        """
        return geometry_guidance_fn(
            x_t=x,
            t=t,
            sdf_grid=sdf_dev,
            origin=origin_dev,
            spacing=spacing_dev,
            guidance_scale=args.guidance_scale,
            )

    logger.log("sampling...")
    all_images = []
    all_labels = []
    #noise = th.zeros(
    # noise = th.ones(
    #     (args.batch_size, args.in_channels, args.image_size),
    #     dtype=th.float32,
    #     device=dist_util.dev()
    # ) * 2
    # noise = th.from_numpy(
    #     np.load('../velocity_module-IS64-NC128-NRB3-DS4000-NScosine-LR1e-4-BS256-sample/fixed_noise_64x1x64x64.npy')
    # ).to(dtype=th.float32, device=dist_util.dev())
    import os
    seed = 0*8 + int(os.environ["CUDA_VISIBLE_DEVICES"])
    th.manual_seed(seed)
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        # Pass geometry to the model (Unet conditioning)
        model_kwargs["geometry_grid"] = binary_grid_dev.repeat(args.batch_size, 1, 1, 1, 1)
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        #sample_fn = diffusion.p_sample_loop_history
        sample = sample_fn(
            model,
            (args.batch_size, args.in_channels, args.image_size),
            #noise=noise,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )
        sample = sample.clamp(-1, 1)
        #sample[:, -1] = sample[:, -1].clamp(-1, 1)
        sample = sample.permute(0, 2, 1)
        #sample = sample.permute(0, 1, 3, 2)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        results_dir="",
        log_dir="sample_logs",
        geometry_path="",
        guidance_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

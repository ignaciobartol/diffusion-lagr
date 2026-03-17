"""
Train a diffusion model on Lagrangian trajectories in 3d turbulence.
"""

import argparse
import os

import blobfile as bf
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.turb_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

def _resolve_init_model_save_path(log_dir : str, user_path : str) -> str:
    if user_path != "":
        return user_path
    return os.path.join(log_dir, "initial_model.pt")

def _maybe_load_init_model(model, load_path : str) -> None:
    if not load_path or not os.path.exists(load_path):
        return
    logger.log(f"Loading initial model from {load_path}")
    state_dict = dist_util.load_state_dict(load_path, map_location=dist_util.dev())
    model.load_state_dict(state_dict)
    dist_util.sync_params(model.parameters())

def _maybe_save_init_model(model, save_path : str) -> None:
    if not save_path:
        return
    if dist.get_rank() == 0:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        logger.log(f"Saving initial model to {save_path}")
        with bf.BlobFile(save_path, "wb") as f:
            th.save(model.state_dict(), f)

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    _maybe_load_init_model(model, args.init_model_load_path)
    if args.save_init_model:
        save_path = _resolve_init_model_save_path(args.log_dir,
                                                  args.init_model_save_path)
        _maybe_save_init_model(model, save_path)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
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
    ).run_loop()


def create_argparser():
    defaults = dict(
        dataset_path="",
        dataset_name="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        log_dir="checkpoints",
        save_init_model=False,
        init_model_save_path="",
        init_model_load_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

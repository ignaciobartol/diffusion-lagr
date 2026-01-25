#!/usr/bin/env python
"""
Inspect geometry encoder outputs for a single geometry file.
"""
import argparse
import numpy as np
import torch

from guided_diffusion.geo_encoder import GeometryEncoder, VariationalGeometryEncoder


def main():
    parser = argparse.ArgumentParser(description="Inspect geometry encoder outputs.")
    parser.add_argument("geometry_path", type=str, help="Path to geometry .npz file.")
    parser.add_argument("--encoder-type", choices=["deterministic", "variational"], default="variational")
    parser.add_argument("--output-dim", type=int, default=512)
    parser.add_argument("--no-sample", action="store_true", help="Use mean instead of sampling for variational encoder.")
    args = parser.parse_args()

    geo = np.load(args.geometry_path)
    grid = geo["binary"].astype(np.float32)
    grid = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0)

    if args.encoder_type == "variational":
        encoder = VariationalGeometryEncoder(output_dim=args.output_dim, input_channels=1)
        embedding, kl = encoder(grid, sample=not args.no_sample)
        print(f"KL divergence (mean): {kl.item():.6f}")
    else:
        encoder = GeometryEncoder(output_dim=args.output_dim, input_channels=1)
        embedding = encoder(grid)
        kl = None

    emb_np = embedding.detach().cpu().numpy()
    print(f"Embedding shape: {emb_np.shape}")
    print(f"Embedding stats: mean={emb_np.mean():.6f}, std={emb_np.std():.6f}, "
          f"min={emb_np.min():.6f}, max={emb_np.max():.6f}")


if __name__ == "__main__":
    main()

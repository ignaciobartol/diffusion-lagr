import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

def geometry_guidance_fn(
        x_t: torch.Tensor,
        t: torch.Tensor,
        sdf_grid: torch.Tensor,
        origin: torch.Tensor,
        spacing: torch.Tensor,
        guidance_scale: float = 10.0,
        coord_space: str = "world",
        debug: bool = False,
        debug_folder: str = "debug_plots"
        ) -> torch.Tensor:
    """
    Computes the gradient of the SDF as a penetration loss,
    to guide particles back into the valid geometry 
    during the diffusion process.

    Args:
        x_t: [B, 3, N] tensor of particle positions at time t
        t: [B] tensor of time steps indices (req by API).
        sdf_grid: [1, 1, D, H, W] tensor representing the SDF grid
            (1 channel, 3D).
        origin: [3] (x,y,z) coordinates of the SDF grid origin.
        spacing: [3] (dx,dy,dz) voxel spacing of the SDF grid.
        guidance_scale: float, strength of the guidance.
        coord_space: str, either "world" or "voxel". One of 
            ["world", "voxel", "normalized"]

    Returns:
        gradient: [B, 3, N] guidance gradient to be added to the 
            model mean output.
    """

    with torch.enable_grad():
        x_in = x_t.detach().requires_grad_(True)  # [B, 3, N]

        # Coordinates conversion: physical -> voxel grid
        dims = torch.tensor(sdf_grid.shape[2:], device=x_in.device).view(1, 3, 1)
        denom = (dims - 1).clamp_min(1.0)

        if coord_space == "normalized":
            norms_coords = x_in
            coords = (norms_coords + 1.0) * 0.5 * denom
        else:
            if coord_space == "voxel":
                coords = x_in
            else:
                origin_dev = origin.to(x_in.device).view(1, 3, 1)
                spacing_dev = spacing.to(x_in.device).view(1, 3, 1)
                coords = (x_in - origin_dev) / spacing_dev

            norms_coords = 2.0 * (coords / denom) - 1.0

        # x_in shape is [B, 3, N]. We permute to [B, 1, 1, N, 3] to trick grid_sample.
        # Format: (Batch, Depth, Height, Width, Channels/XYZ)
        sample_coords = norms_coords.permute(0, 2, 1).unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N, 3]
        # sample_coords = norms_coords.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N, 3]

        if debug:
            os.makedirs(debug_folder, exist_ok=True)
            
            # Convert tensors to numpy for plotting (Take Batch 0)
            # grid_coords is [B, 3, N]. We access [0] -> [3, N]
            gc_np = coords[0].detach().cpu().numpy() 
            sdf_np = sdf_grid[0, 0].detach().cpu().numpy() # [D, H, W]
            # 1. Select Z-Slice based on mean particle position
            # Note: We assume Tensor layout [D, H, W] corresponds to Z, Y, X
            z_idx = int(np.mean(gc_np[2, :]))
            z_idx = np.clip(z_idx, 0, sdf_np.shape[0] - 1)
            
            # 2. Extract Slice
            sdf_slice = sdf_np[z_idx, :, :] # Shape [H, W]
            
            # 3. Filter points close to this slice (Visual Clutter reduction)
            # We only plot particles within +/- 1.0 voxel of the slice
            mask = np.abs(gc_np[2, :] - z_idx) < np.inf
            pts_x = gc_np[0, :][mask] # X coordinates
            pts_y = gc_np[1, :][mask] # Y coordinates
            
            # 4. Plot
            plt.figure(figsize=(8, 8))
            # origin='lower' puts (0,0) at bottom-left. 
            # Check your voxelizer convention. Usually images are 'upper'.
            # We use 'lower' to match standard Cartesian plots.
            plt.imshow(sdf_slice, origin='lower', cmap='viridis', label='SDF')
            plt.colorbar(label='SDF Distance')
            
            # Scatter Particles
            # c='red' for visibility, s=10 for size
            plt.scatter(pts_x, pts_y, c='red', s=10, label='Particles', edgecolors='black')
            
            plt.title(f"SDF Slice (z={z_idx}) vs Particles (t={t[0].item()})")
            plt.xlabel("X Grid Index")
            plt.ylabel("Y Grid Index")
            plt.legend(loc='upper right')
            
            # Save file with timestep index to create a sequence
            fname = os.path.join(debug_folder, f"guidance_t{int(t[0].item()):04d}.png")
            plt.savefig(fname)
            plt.close()

        # Sample SDF values at particle positions
        batch_size = x_in.shape[0]
        sdf_batch = sdf_grid.expand(batch_size, -1, -1, -1, -1)  # [B, 1, D, H, W]
        sampled_sdf = F.grid_sample(
            sdf_batch,
            sample_coords,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        sampled_sdf = sampled_sdf.view(batch_size, -1)  # [B, N]

        if debug == True:
            print(f"\nx_t:\n{x_t.detach().cpu().numpy()}")
            print(f"\nsample_coords:\n{sample_coords.detach().cpu().numpy()}")
            print(f"\nsampled_SDF:\n{sampled_sdf.detach().cpu().numpy()}")

        # Compute loss: penalize negative SDF values (out wall)
        # Conv: SDF < 0 -> loss > 0, SDF >= 0 -> loss = 0
        penetration_loss = F.relu(-sampled_sdf).sum()

        if debug == True:
            print(f"\npenetration_loss: {penetration_loss.item()}")

        # Compute gradients
        if penetration_loss > 0:
            grad  = torch.autograd.grad(
                penetration_loss,
                x_in,
                retain_graph=True
            )[0]  # [B, 3, N]
            grad = F.relu(grad)  # Only push back inside
            return -grad * guidance_scale
        else:
            return torch.zeros_like(x_in)
import torch
import torch.nn.functional as F

def geometry_guidance_fn(
        x_t: torch.Tensor,
        t: torch.Tensor,
        sdf_grid: torch.Tensor,
        origin: torch.Tensor,
        spacing: torch.Tensor,
        guidance_scale: float = 10.0
        ) -> torch.Tensor:
    """
    Computes the gradient of the SDF as a penetration loss,
    to guide particles back into the valid geometry 
    during the diffusion process.

    Args:
        x_t: [B, 3, N] tensor of particle positions at time t, with
        physical units (meters).
        t: [B] tensor of time steps indices (req by API).
        sdf_grid: [1, 1, D, H, W] tensor representing the SDF grid
            (1 channel, 3D).
        origin: [3] (x,y,z) coordinates of the SDF grid origin.
        spacing: [3] (dx,dy,dz) voxel spacing of the SDF grid.
        guidance_scale: float, strength of the guidance.

    Returns:
        gradient: [B, 3, N] guidance gradient to be added to the 
            model mean output.
    """

    with torch.enable_grad():
        x_in = x_t.detach().requires_grad_(True)  # [B, 3, N]
        
        # Coordinates conversion: physical -> voxel grid
        origin_dev = origin.to(x_in.device).view(1, 3, 1)
        spacing_dev = spacing.to(x_in.device).view(1, 3, 1)
        coords = (x_in - origin_dev) / spacing_dev

        # Normalize coordinates to [-1, 1] for grid_sample
        dims = torch.tensor(sdf_grid.shape[2:], device=x_in.device).view(1, 3, 1)
        norms_coords = 2.0 * (coords / dims) - 1.0

        # x_in shape is [B, 3, N]. We permute to [B, 1, 1, N, 3] to trick grid_sample.
        # Format: (Batch, Depth, Height, Width, Channels/XYZ)
        sample_coords = norms_coords.permute(0, 2, 1).unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N, 3]

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

        # Compute loss: penalize negative SDF values (out wall)
        # Conv: SDF < 0 -> loss > 0, SDF >= 0 -> loss = 0
        penetration_loss = F.relu(-sampled_sdf).sum()

        # Compute gradients
        if penetration_loss > 0:
            grad  = torch.autograd.grad(
                penetration_loss,
                x_in,
                retain_graph=True
            )[0]  # [B, 3, N]
            return -grad * guidance_scale
        else:
            return torch.zeros_like(x_in)
        
        




import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometryEncoder(nn.Module):
    """
    Encodes a 3D Voxel grid (from STL) into a latent representation
    for conditioning a diffusion model.

    Inputs: (B, 1, D, H, W) voxel grids
    Outputs: (B, time_embed_dim) latent context vectors
    """
    def __init__(self, output_dim, input_channels=1):
        super().__init__()

        # Lightweight 3D CNN architecture
        self.encoder = nn.Sequential(
            # Downsample
            nn.Conv3d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.SiLU(),
            nn.MaxPool3d(2),

            # Feature extraction
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.SiLU(),
            nn.MaxPool3d(2),

            # Deeper features
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        # Proj head to match the U-net time embedding dimension
        self.projector = nn.Sequential(
            nn.Linear(64, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        """
        x: Tensor of shape (B, 1, D, H, W), voxel grids.
        Returns: Tensor of shape (B, output_dim), latent context vectors.
        """
        features = self.encoder(x)  # (B, 64, 1, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 64)
        embedding = self.projector(features)  # (B, output_dim)
        return embedding
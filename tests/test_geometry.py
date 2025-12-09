import torch
import numpy as np
import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from guided_diffusion.geometry_util import geometry_guidance_fn

class TestGeometryGuidance(unittest.TestCase):
    
    def setUp(self) -> None:
        """
        Setup a synthetic spherical SDF grid for testing.
        Mathh: SDF(r) = ||r - center|| - radius
        Sphere centered at (10,10,10) with radius 4 in a 20x20x20 grid.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dim = 64
        self.spacing_val = 1.0  # 1 unit per voxel
        self.origin_val = 0.0   # Origin at (10,10,10)

        self.origin = torch.tensor([self.origin_val, self.origin_val, self.origin_val],
                                   device=self.device)
        self.spacing = torch.tensor([self.spacing_val, self.spacing_val, self.spacing_val],
                                    device=self.device)
        x = torch.arange(0, self.dim, device=self.device).float()
        y = torch.arange(0, self.dim, device=self.device).float()
        z = torch.arange(0, self.dim, device=self.device).float()
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')

        center = 10.0
        radius = 4.0
        dist_from_center = torch.sqrt((grid_x - center) ** 2 +
                                      (grid_y - center) ** 2 +
                                      (grid_z - center) ** 2)
        self.sdf_data = radius - dist_from_center  # SDF values
        self.sdf_grid = self.sdf_data.unsqueeze(0).unsqueeze(0) # [1,1,D,H,W]
        return super().setUp() 
    
    def test_safe_particle(self):
        """
        Case 1: Particle inside the sphere. Should have zero gradient.
        """
        x_t = torch.tensor([[[10.0], [10.0], [10.0]]], device=self.device)  # [1,3,1]
        t = torch.tensor([0], device=self.device)

        grad = geometry_guidance_fn(x_t, t, self.sdf_grid, self.origin,
                                    self.spacing, guidance_scale=1.0)
        grad_np = grad.cpu().numpy()
        print(f"\n[Test Safe] Grad {grad_np.flatten()}")
        self.assertTrue(np.allclose(grad_np, 0.0, atol=1e-5),
                        "Gradient should be zero for safe particle.")
        return None
    
    def test_wall_particle_x_axis(self):
        """
        Case 2: Particle outside the sphere along x-axis.
        Expect gradient pointing towards (left) negative x direction.
        """
        x_t = torch.tensor([[[16.0], [10.0], [10.0]]], device=self.device)  # [1,3,1]
        t = torch.tensor([0], device=self.device)

        grad = geometry_guidance_fn(x_t, t, self.sdf_grid, self.origin,
                                    self.spacing, guidance_scale=1.0)
        grad_np = grad.cpu().numpy().flatten()

        print(f"\n[Test Wall X] Particle at 16.0, pushed by: {grad_np}")
        self.assertTrue(grad_np[0] < 0, "Gradient x-component should be negative.")
        self.assertTrue(np.isclose(grad_np[1], 0.0, atol=1e-5),
                        "Gradient y-component should be zero.")
        self.assertTrue(np.isclose(grad_np[2], 0.0, atol=1e-5),
                        "Gradient z-component should be zero.")
        return None
    
    def test_wall_particle_diagonal(self):
        """
        Case 3: Particle outside the sphere along diagonal (1,1,1).
        Expect gradient pointing towards center direction (-1,-1,-1).
        """
        val = 15.0
        x_t = torch.tensor([[[val], [val], [val]]], device=self.device)  # [1,3,1]
        t = torch.tensor([0], device=self.device)

        grad = geometry_guidance_fn(x_t, t, self.sdf_grid, self.origin,
                                    self.spacing, guidance_scale=1.0)
        grad_np = grad.cpu().numpy().flatten()

        print(f"\n[Test Wall Diagonal] Particle at diag pos {val}, pushed by: {grad_np}")
        self.assertTrue(np.all(grad_np < 0), "All Gradient components should be negative.")
        return None
    
    def test_batch_processing(self):
        """
        Case 4: Particle at wall, outside, and inside in a batch.
        """
        x_t = torch.tensor([[[10.0, 16.0, 10.0, 13.0],
                             [14.0, 17.0, 10.0, 14.0],
                             [10.0, 18.0, 10.0, 10.0],
                             ]], device=self.device)  # [1,3,3]
        t = torch.tensor([0], device=self.device)

        # print (f"\nSDF values are\n{self.sdf_grid.cpu().numpy()}")
        grad = geometry_guidance_fn(x_t, t, self.sdf_grid, self.origin,
                                    self.spacing, guidance_scale=1.0)
        grad_np = grad.cpu().numpy().flatten().reshape(3,x_t.shape[2])

        print(f"\n[Test Batch] Gradients:\n{grad_np}")
        # First particle should have zero gradient
        self.assertTrue(np.allclose(grad_np[:, 0], 0.0, atol=1e-4), "First particle gradient should be around zero.")
        # Second particle should have negative gradients
        self.assertTrue(np.all(grad_np[:, 1] <= 0),
                        "Second particle gradients should be negative.")
        # Third particle should have zero gradient
        self.assertTrue(np.allclose(grad_np[:, 2], 0.0, atol=1e-5),
                        "Third particle gradients should be zero.")
        return None
    
if __name__ == "__main__":
    unittest.main()
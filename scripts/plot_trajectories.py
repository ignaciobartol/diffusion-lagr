#%%
import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py
# Debug version: Replace argparse with hardcoded variables for easier debugging.
# Commented out the original code below for reference.

# def main():
#     parser = argparse.ArgumentParser(description="Plot sampled and ground truth particle trajectories.")
#     parser.add_argument('--sampled', type=str, required=True, help='Path to sampled .npz file')
#     parser.add_argument('--ground_truth', type=str, required=True, help='Path to ground truth .npz file')
#     parser.add_argument('--particles', type=int, nargs='+', default=[0, 1], help='Indices of particles to plot')
#     parser.add_argument('--xy', type=int, nargs=2, default=[0, 1], help='Which coordinates to plot (default: x=0, y=1)')
#     args = parser.parse_args()
#
#     sampled = np.load(args.sampled)
#     sampled_key = list(sampled.keys())[0]
#     sampled_data = sampled[sampled_key]  # shape: (num_particles, num_timesteps, 3)
#
#     gt = np.load(args.ground_truth)
#     gt_key = list(gt.keys())[0]
#     gt_data = gt[gt_key]  # shape: (num_particles, num_timesteps, 3)
#
#     n_particles = len(args.particles)
#     fig, axs = plt.subplots(1, n_particles, figsize=(6 * n_particles, 5))
#     if n_particles == 1:
#         axs = [axs]
#     for i, idx in enumerate(args.particles):
#         ax = axs[i]
#         ax.plot(sampled_data[idx, :, args.xy[0]], sampled_data[idx, :, args.xy[1]], label='Sampled', color='blue') # type: ignore
#         ax.plot(gt_data[idx, :, args.xy[0]], gt_data[idx, :, args.xy[1]], label='Ground Truth', color='red', linestyle='--') # type: ignore
#         ax.set_title(f'Particle {idx} Trajectory') # type: ignore
#         ax.set_xlabel(f'Coord {args.xy[0]}') # type: ignore
#         ax.set_ylabel(f'Coord {args.xy[1]}') # type: ignore
#         ax.legend() # type: ignore
#         ax.axis('equal') # type: ignore
#
#     plt.tight_layout()
#     plt.savefig('trajectories_plot.png')
#     plt.show()
#
# if __name__ == "__main__":
#     main()

# Debug version below (no argparse, hardcoded variables):

# Hardcoded file paths for debugging
sampled_path = '../results/samples_128x2000x3.npz'
ground_truth_path = '../datasets/Lagr_u3c_diffusion-demo.h5'
particles = [0, 1]  # Indices of particles to plot
xy = [0, 1]         # Coordinates to plot (x=0, y=1)

# Load sampled data
sampled = np.load(sampled_path)
sampled_key = list(sampled.keys())[0]
sampled_data = sampled[sampled_key]  # shape: (num_particles, num_timesteps, 3)


# Load ground truth data
with h5py.File(ground_truth_path, 'r') as h5f:
    rx0 = np.array(h5f.get('min'))
    rx1 = np.array(h5f.get('max'))
    u3c = np.array(h5f.get('train'))

gt_data = (u3c+1)*(rx1-rx0)/2 + rx0
sampled_data = (sampled_data+1)*(rx1-rx0)/2 + rx0
#%% 
# gt_key = list(gt.keys())[0]
gt_data = gt_data[particles]
# gt_data = gt[gt_key]  # shape: (num_particles, num_timesteps, 3)

n_particles = len(particles)
fig, axs = plt.subplots(1, n_particles, figsize=(6 * n_particles, 5))
if n_particles == 1:
    axs = [axs]
for i, idx in enumerate(particles):
    ax = axs[i]
    ax.plot(sampled_data[idx, :, xy[0]], sampled_data[idx, :, xy[1]], label='Sampled', color='blue')
    ax.plot(gt_data[idx, :, xy[0]], gt_data[idx, :, xy[1]], label='Ground Truth', color='red', linestyle='--')
    ax.set_title(f'Particle {idx} Trajectory')
    ax.set_xlabel(f'Coord {xy[0]}')
    ax.set_ylabel(f'Coord {xy[1]}')
    ax.legend()
    ax.axis('equal')

plt.tight_layout()
plt.savefig('trajectories_plot.png')
plt.show()

# %%

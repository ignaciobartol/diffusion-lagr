#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D

#%%
with h5py.File('datasets/Lagr_u3c_diffusion-demo.h5', 'r') as h5f:
    rx0 = np.array(h5f.get('min'))
    rx1 = np.array(h5f.get('max'))
    u3c = np.array(h5f.get('train'))

velocities = (u3c+1)*(rx1-rx0)/2 + rx0

# %%
# Print header of the h5 file:
print("Header of the HDF5 file:")
h5f = h5py.File('datasets/Lagr_u3c_diffusion-demo.h5', 'r')
for key in h5f.keys():
    obj = h5f[key]
    if isinstance(obj, h5py.Dataset):
        print(f"{key}: {obj.shape} - {obj.dtype}")
    else:
        print(f"{key}: (not a dataset, type: {type(obj)})")

#%% 
# Print max min values from h5 file:
print("\nMax and Min values in the dataset:")
print(f"Min: {rx0}, Max: {rx1}")
# %%
particles = [0, 1]  # Indices of particles to plot
xy = [0, 1]         # Coordinates to plot (x=0, y=1)


gt_data = velocities[particles]
# gt_data = gt[gt_key]  # shape: (num_particles, num_timesteps, 3)

n_particles = len(particles)
fig, axs = plt.subplots(1, n_particles, figsize=(6 * n_particles, 5))
if n_particles == 1:
    axs = [axs]
for i, idx in enumerate(particles):
    ax = axs[i]
    ax.plot(gt_data[idx, :, xy[0]], gt_data[idx, :, xy[1]], label='Ground Truth', color='red', linestyle='--')
    ax.set_title(f'Particle {idx} Trajectory')
    ax.set_xlabel(f'Coord {xy[0]}')
    ax.set_ylabel(f'Coord {xy[1]}')
    ax.legend()
    ax.axis('equal')

plt.tight_layout()
# plt.savefig('trajectories_plot.png')
plt.show()

# %%
data_ar = np.load('/storage/project/r-sdewji3-0/ibartol3/FastDep-CFPD/dataset/bb-part-0.42.npy')
data_ar = np.moveaxis(data_ar, 1, 0)  # Move the second axis to the first position
print("Shape of the data array:", data_ar.shape)

# %%

def loading_trackfile(csv_pathfile: str) -> pd.DataFrame:
    """
    Loads a track file from CSV or Parquet format. If a Parquet version does not exist,
    it will be created from the CSV for faster future loading.

    Parameters
    ----------
    csv_pathfile : str
        Full path to the CSV file.

    Returns
    -------
    f : pd.DataFrame
        Loaded DataFrame from the track file.
    """
    if not isinstance(csv_pathfile, str):
        raise TypeError("csv_pathfile must be a string path to the CSV file.")

    folder_path, csv_filename = os.path.split(csv_pathfile)
    base_filename, _ = os.path.splitext(csv_filename)
    parquet_pathfile = os.path.join(folder_path, base_filename + ".parquet")

    if not os.path.isfile(csv_pathfile) and not os.path.isfile(parquet_pathfile):
        raise FileNotFoundError(f"Not a parquet or .csv file were found in: {folder_path}")
    
    if not os.path.isfile(parquet_pathfile):
        print("Reading csv and converting to parquet, this may take a while...")
        f = pd.read_csv(csv_pathfile)
        f.to_parquet(parquet_pathfile, compression=None)
    else:
        print("Reading parquet format")
        f = pd.read_parquet(parquet_pathfile, engine="fastparquet")

    return f
# %%
csv_path = "/storage/project/r-sdewji3-0/ibartol3/StarCCM+/bb-sim-part/bb-part-0.42mum.csv" 
print(f"------------Analyzing {csv_path}---------------")
df = loading_trackfile(csv_path)
try:
    df = df.sort_values(by=["Track: Parcel Index", "Track: Time (s)"])
    z_max = np.amax(df["Track: Position[Z] (m)"])
except KeyError:
    z_max = np.amax(df["Position[Z] (m)"])

group_sizes = df.groupby(["Track: Parcel Index"]).size()
longest_length = group_sizes.max()
unique_parcels = df["Track: Parcel Index"].unique()
n_parcels = len(unique_parcels)
filtered_df = df[df["Track: Parcel Index"].isin(unique_parcels)]
col_names = list(df.columns.values)

#%%
print(col_names[-3:])  # Print the last three column names
print(df['X (m)'][0], df["Track: Position[X] (m)"][0])
print(df['Y (m)'][0], df["Track: Position[Y] (m)"][0])
print(df['Z (m)'][0], df["Track: Position[Z] (m)"][0])

# %%

# Create a h5 file with the data
h5_file_path = "/storage/project/r-sdewji3-0/ibartol3/diffusion-lagr/datasets/bb-part-0.42.h5"
if os.path.isfile(h5_file_path):
    assert os.path.isfile(h5_file_path), "HDF5 file already exists, please remove it before running this script."

with h5py.File(h5_file_path, 'w') as h5f:
    h5f.create_dataset('min', data=np.array([df["Track: Position[X] (m)"].min(),
                                             df["Track: Position[Y] (m)"].min(),
                                             df["Track: Position[Z] (m)"].min()]))
    h5f.create_dataset('max', data=np.array([df["Track: Position[X] (m)"].max(),
                                             df["Track: Position[Y] (m)"].max(),
                                             df["Track: Position[Z] (m)"].max()]))
    h5f.create_dataset('train', data=data_ar[0:512, 0:1024, -3:])
# %%
# Check if the file was created successfully
if os.path.isfile(h5_file_path):
    print(f"HDF5 file created successfully at {h5_file_path}")
else:
    print(f"Failed to create HDF5 file at {h5_file_path}")      

#Print the shape of the newly created array and check that it matches the original data
with h5py.File(h5_file_path, 'r') as h5f:
    train_data = np.array(h5f.get('train'))
    print("Shape of the train dataset in the HDF5 file:", train_data.shape)
    print("Shape of the original data array:", data_ar[0:512, 0:1500, -3:].shape)
    print(np.array(h5f.get('min')), np.array(h5f.get('max')))
    print(data_ar[0:5, 0, -3:])
    print('------')
    print(train_data[0:5, 0, :])
    assert train_data.shape == data_ar[0:512, 0:1024, -3:].shape, "Shapes do not match!"
# %%
particles = [0, 51]  # Indices of particles to plot
xy = [0, 1]         # Coordinates to plot (x=0, y=1)
n_particles = len(particles)

fig, axs = plt.subplots(1, n_particles, figsize=(6 * n_particles, 5))
if n_particles == 1:
    axs = [axs]
for i, idx in enumerate(particles):
    ax = axs[i]
    for j in range(50):
        ax.plot(data_ar[idx+j, :, -1], data_ar[idx+j, :, -2],
                label='Ground Truth', linestyle='--')
    ax.set_title(f'Particle {idx} Trajectory')
    ax.set_xlabel(f'Coord {xy[0]}')
    ax.set_ylabel(f'Coord {xy[1]}')
    # ax.legend()
    ax.axis('equal')

plt.tight_layout()
# plt.savefig('trajectories_plot.png')
plt.show()
# %%

data_df = np.load('/storage/home/hcoda1/7/ibartol3/r-sdewji3-0/diffusion-lagr/results-1/samples_128x1024x3.npz')['arr_0']
# data_ar = np.moveaxis(data_ar, 1, 0)  # Move the second axis to the first position
print("Shape of the data array:", data_df.shape)

#%%
fig, axs = plt.subplots(1, n_particles, figsize=(6 * n_particles, 5))
particles = [0]  # Indices of particles to plot
xy = [-2, -3]  # Coordinates to plot (x=0, y=1)
# n_particles = len(particles)
if n_particles == 1:
    axs = [axs]
for i, idx in enumerate(particles):
    ax = axs[i]
    for j in range(512):
        ax.plot(data_ar[idx+j, :, xy[0]], data_ar[idx+j, :, xy[1]],
                label='Ground Truth', linestyle='-', color='red',
                alpha=0.2)
    for j in range(128):
        ax.plot(data_df[idx+j, :, xy[0]], data_df[idx+j, :, xy[1]],
                label='Diffusion Model', linestyle='--', color='blue',
                alpha=0.4)

    handles = [
        plt.Line2D([0], [0], color='blue', linestyle='--', label='Diffusion Model'),
        plt.Line2D([0], [0], color='red', linestyle='-', label='Ground Truth')
    ]
    ax.legend(handles=handles)
    ax.set_title(f'Particle {idx} Trajectory')
    ax.set_xlabel(f'Coord {xy[0]}')
    ax.set_ylabel(f'Coord {xy[1]}')
    # ax.legend()
    ax.axis('equal')

plt.tight_layout()
# plt.savefig('trajectories_plot.png')
plt.show()
# %%
particles = [0]
xy_pairs = [(-3, -2), (-3, -1), (-2, -1)]  # (Z,Y), (Z,X), (Y,X)
labels = [('X', 'Y'), ('X', 'Z'), ('Y', 'Z')]
n_proj = len(xy_pairs)
n_sample = 128
n_train = 512

fig = plt.figure(figsize=(20, 5))
for i, (xy, (xlabel, ylabel)) in enumerate(zip(xy_pairs, labels)):
    ax = fig.add_subplot(1, n_proj + 1, i + 1)
    for j in range(n_train):
        ax.plot(data_ar[j, :, xy[0]], data_ar[j, :, xy[1]], color='red', alpha=0.2)
    for j in range(n_sample):
        ax.plot(data_df[j, :, xy[0]], data_df[j, :, xy[1]], color='blue', alpha=0.4, linestyle='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    handles = [
        plt.Line2D([0], [0], color='blue', linestyle='--', label='Diffusion Model'),
        plt.Line2D([0], [0], color='red', linestyle='-', label='Ground Truth')
    ]
    ax.legend(handles=handles)
    ax.set_title(f'{xlabel}-{ylabel} Projection')
    ax.axis('equal')

# 3D plot
ax3d = fig.add_subplot(1, n_proj + 1, n_proj + 1, projection='3d')
for j in range(n_train):
    ax3d.plot(data_ar[j, :, -3], data_ar[j, :, -2], data_ar[j, :, -1], color='red', alpha=0.2)
for j in range(n_sample):
    ax3d.plot(data_df[j, :, -3], data_df[j, :, -2], data_df[j, :, -1], color='blue', alpha=0.4, linestyle='--')
ax3d.set_xlabel('Z')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('X')
handles = [
        plt.Line2D([0], [0], color='blue', linestyle='--', label='Diffusion Model'),
        plt.Line2D([0], [0], color='red', linestyle='-', label='Ground Truth')
    ]
ax3d.legend(handles=handles)
ax3d.set_title('3D Trajectory')
plt.tight_layout()
plt.show()
# %%

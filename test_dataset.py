#%%
import h5py
import numpy as np

with h5py.File('datasets/Lagr_u3c_diffusion-demo.h5', 'r') as h5f:
    rx0 = np.array(h5f.get('min'))
    rx1 = np.array(h5f.get('max'))
    u3c = np.array(h5f.get('train'))

velocities = (u3c+1)*(rx1-rx0)/2 + rx0
# %%
dir(h5f)
# %%
h5f = h5py.File('datasets/Lagr_u3c_diffusion-demo.h5', 'r')

# %%
h5f.keys()
# %%
h5f.get('train').shape

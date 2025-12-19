from mpi4py import MPI
import torch
import torch.nn.functional as F
import h5py
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import bisect

def load_data(
    *,
    dataset_path,
    dataset_name = "",
    batch_size,
    class_cond=False,
    deterministic=False,
    **kwargs #Catch all for legacy args
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param dataset_path: a dataset path.
    :param dataset_name: a dataset name.
    :param batch_size: the batch size of each returned pair.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. Not implemented.
    :param deterministic: if True, yield results in a deterministic order.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    with open(dataset_path, 'r') as f:
        manifest = json.load(f)

    file_lengths = []
    cumulative_lengths = [0]
    total_len = 0

    for entry in manifest:
        h5_path = entry['h5_path']
        d_name = entry.get('dataset_name', dataset_name)
        with h5py.File(h5_path, 'r', driver='mpio', comm=MPI.COMM_SELF) as f:
        # with h5py.File(dataset_path, 'r') as f:  # replace the above line with this line for serial h5py
            length = f[d_name].len()
        file_lengths.append(length)
        total_len += length
        cumulative_lengths.append(total_len)

    # Determine the chunk of data this worker will process    
    chunk_size = total_len // size
    start_idx  = rank * chunk_size

    dataset = MultiGeometryDataset(
        manifest = manifest,
        cumulative_lengths = cumulative_lengths,
        start_idx = start_idx,
        chunk_size = chunk_size,
        class_cond = class_cond
    )
    # dataset = TurbDataset(
    #     dataset_path, dataset_name, class_cond, start_idx, chunk_size,
    # )

    # When deterministic=True we want to disable shuffling so that
    # each worker always processes the same subset in the same order.
    shuffle = not deterministic
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=True
    )

    while True:
        yield from loader

class MultiGeometryDataset(Dataset):
    def __init__(
        self,
        manifest,
        cumulative_lengths,
        start_idx,
        chunk_size,
        class_cond=False
    ):
        super().__init__()
        self.manifest = manifest
        self.cumulative_lengths = cumulative_lengths
        self.class_cond = class_cond
        self.start_idx  = start_idx
        self.chunk_size = chunk_size

        self.geo_cache = {}
        print(f"Loading {len(manifest)} geometries into memory...")

        raw_grids = {}
        max_d, max_h, max_w = 0, 0, 0

        for entry in manifest:
            geo_path = entry["geo_path"]
            if geo_path not in self.geo_cache:
                try:
                    data = np.load(geo_path)
                    # Grid [D, H, W]
                    grid = data["binary"].astype(np.float32)
                    grid = np.expand_dims(grid, axis=0)  # [1, D, H, W]
                    raw_grids[geo_path] = grid

                    _, d, h, w = grid.shape
                    max_d = max(max_d, d)
                    max_h = max(max_h, h)
                    max_w = max(max_w, w)

                except Exception as e:
                    print(f"FAILED to load geometry from {geo_path}: ({e})")
                    raise e
        print(f"Max geometry grid size: D={max_d}, H={max_h}, W={max_w}, padding all grids to this size.")

        for path, grid in raw_grids.items():
            _, d, h, w = grid.shape
            pad_d = max_d - d
            pad_h = max_h - h
            pad_w = max_w - w

            pad_width = (0, pad_w, 0, pad_h, 0, pad_d) # Pading for last 3 dims
            tensor_grid = torch.from_numpy(grid)
            padded_grid = F.pad(tensor_grid, pad_width, mode='constant', value=0)
            self.geo_cache[path] = padded_grid
                
    def __len__(self):
        return self.chunk_size
    
    def __getitem__(self, idx):
        global_idx = idx + self.start_idx

        # Find which file this index belongs to
        file_idx = bisect.bisect_right(self.cumulative_lengths, global_idx) - 1

        # Local index within the file
        file_offset = self.cumulative_lengths[file_idx]
        local_sample_idx = global_idx - file_offset

        # Retrieve metadata for this file
        entry = self.manifest[file_idx]
        h5_path = entry['h5_path']
        d_name = entry.get('dataset_name', 'train')
        geo_path = entry['geo_path']

        with h5py.File(h5_path, 'r', driver='mpio', comm=MPI.COMM_SELF) as f:
            data = f[d_name][local_sample_idx].astype(np.float32)
            data = np.moveaxis(data, -1, 0)  # [C, D, H, W]
            out_dict = {}

            if self.class_cond:
                raise NotImplementedError()
                # out_dict["y"] = f[d_name + '_y'][local_sample_idx]
        out_dict["geometry_grid"] = self.geo_cache[geo_path]

        return data, out_dict


# class TurbDataset(Dataset):
#     def __init__(
#         self,
#         dataset_path,
#         dataset_name,
#         class_cond,
#         start_idx,
#         chunk_size,
#     ):
#         super().__init__()
#         self.dataset_path = dataset_path
#         self.dataset_name = dataset_name
#         self.class_cond = class_cond
#         self.start_idx  = start_idx
#         self.chunk_size = chunk_size

#     def __len__(self):
#         return self.chunk_size

#     def __getitem__(self, idx):
#         idx += self.start_idx

#         with h5py.File(self.dataset_path, 'r', driver='mpio', comm=MPI.COMM_SELF) as f:
#         # with h5py.File(self.dataset_path, 'r') as f:  # replace the above line with this line for serial h5py
#             data = f[self.dataset_name][idx].astype(np.float32)
#             data = np.moveaxis(data, -1, 0)

#             out_dict = {}
#             if self.class_cond:
#                 raise NotImplementedError()
#                 out_dict["y"] = f[self.dataset_name + '_y'][idx]

#         return data, out_dict

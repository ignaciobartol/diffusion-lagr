# conditioned-diff-lagr

Geometry-conditioned diffusion models for generating 3D Lagrangian particle trajectories in subject-specific airway geometries.

This branch extends the original [diffusion-lagr](https://github.com/SmartTURB/diffusion-lagr) codebase with:

- conditioning on an arbitrary human airway geometry using variational autoencoders. 
- **multi-geometry training** through a JSON manifest that pairs each trajectory dataset with a processed airway geometry,
- a **3D geometry encoder** that embeds voxelized airway lumen geometry and injects it into the 1D U-Net denoiser,
- **SDF-guided inference** that adjusts sampled trajectories back inside the valid airway domain by affecting the mean predicted values,
- utilities to build training datasets directly from **Star-CCM+ track CSV/parquet exports**,
- utilities to convert **STL airway surfaces** into voxel grids and signed distance fields (SDFs), and
- plotting / comparison scripts for visualizing generated trajectories.

The current geometry-conditioned setup is intended for **position trajectories** with shape **(P, T, C)**, where:

- `P` = number of particle tracks,
- `T` = number of timesteps per track,
- `C` = number of channels (i.e. 3 for 3 dimensions).

In the current airway workflow, the standard choice is:

- `C = 3`
- `channel 0 = x`
- `channel 1 = y`
- `channel 2 = z`

representing **world coordinates** when stored in the HDF5 file.

---

## 1. Repository layout

The files you will use most often are:

| Path | Role |
|---|---|
| `scripts/create_dataset.py` | Build a trimmed training `.h5` from a Star-CCM+ particle track CSV. Automatically caches a parquet copy for faster reruns. |
| `scripts/stl_to_numpy.py` | Convert a watertight STL into a binary voxel volume (`.npy`). Useful if you want voxelization only. |
| `scripts/process_geometry.py` | Convert an STL into a compressed geometry package (`.npz`) containing the binary occupancy, SDF, origin, and spacing. This is the standard preprocessing step for geometry-conditioned training/inference. |
| `scripts/turb_train.py` | Base training script. |
| `scripts/turb_train_monitor.py` | Training script with GPU logging, snapshot sampling, optional GT comparisons, and optional geometry-encoder embedding snapshots. This is the most useful entry point for experiments. |
| `scripts/turb_sample.py` | Inference / sampling script. Generates `.npz` trajectory samples from a trained checkpoint on a single geometry. |
| `scripts/compare_trajectories.py` | Visualization / qualitative comparison of GT and predicted trajectories, optionally with STL overlays. |
| `guided_diffusion/turb_datasets.py` | Multi-geometry dataset loader. Reads a JSON manifest, loads HDF5 tracks, loads geometry grids, and normalizes positions into voxel / normalized coordinates. |
| `guided_diffusion/geo_encoder.py` | 3D CNN geometry encoder (deterministic or variational). |
| `guided_diffusion/unet.py` | 1D U-Net denoiser with geometry embedding injection. |
| `guided_diffusion/geometry_util.py` | SDF-based guidance function used at inference time. |
| `guided_diffusion/script_util.py` | Model and diffusion defaults, plus hyperparameter parser definitions. |
| `guided_diffusion/train_util.py` | Main training loop (optimizer, EMA, checkpointing, distributed training). |
| `fastdep/io.py` | CSV/parquet/HDF5 helper utilities. |
| `datasets/train_manifest.json` | Example multi-geometry training manifest. |

---

## 2. Installation

### 2.1 Clone the repository

```bash
git clone https://github.com/ignaciobartol/cond-diffusion-lagr.git
cd cond-diffusion-lagr
```

### 2.2 Create the environment

The repository ships with `dev_environment.yml` containing the core deep-learning dependencies:

```yaml
python=3.7.16
pytorch=1.13.1
torchvision=0.14.1
torchaudio=0.13.1
pytorch-cuda=11.6
numpy
scipy
matplotlib
scikit-learn
h5py
pyyaml
tqdm
numpy-stl
```

Create and activate the environment:

```bash
conda env create -f dev_environment.yml
conda activate diffusion-lagr
```

Then install the package in editable mode:

```bash
pip install -e .
```

### 2.3 Extra packages required by the geometry / data-prep scripts

Some helper scripts rely on packages that are not always included in the base environment file. In practice, install these as well:

```bash
conda install -c conda-forge mpi4py pandas fastparquet scipy scikit-image vtk
```

Justification for those packages:

- `mpi4py` and MPI-enabled `h5py` are used by the dataset loader.
- `pandas` + `fastparquet` are used to cache CSV track files as parquet.
- `vtk` is used by `stl_to_numpy.py` / `process_geometry.py`.
- `scikit-image` and `numpy-stl` are used by visualization scripts.

### 2.4 MPI / HDF5 note

The current dataset loader opens HDF5 files with MPI-aware arguments (`driver='mpio'`). That means the cleanest setup is:

- MPI installed,
- HDF5 built with MPI,
- `h5py` compiled against that MPI-enabled HDF5.

If you only have **serial h5py**, you will need to replace calls of the form:

```python
h5py.File(path, 'r', driver='mpio', comm=MPI.COMM_SELF)
```

with plain:

```python
h5py.File(path, 'r')
```

in `guided_diffusion/turb_datasets.py`.

---

## 3. Data model and file formats

The geometry-conditioned pipeline uses **three main data types**:

1. **trajectory HDF5** (`.h5`) for training data,
2. **processed geometry** (`.npz`) for conditioning and guidance,
3. **sample output** (`.npz`) for generated trajectories.

### 3.1 Training HDF5 format

Each training `.h5` file is expected to contain at least:

- `train`: trajectory tensor with shape **(P, T, C)**,
- `min`: component-wise minimum of the stored channels,
- `max`: component-wise maximum of the stored channels.

For the current airway setup:

- `P` = number of tracks in that file,
- `T` = fixed number of timesteps (for example `1024`),
- `C = 3` with:
  - channel 0 = `x`,
  - channel 1 = `y`,
  - channel 2 = `z`.

Example:

```python
with h5py.File("datasets/case01.h5", "r") as f:
    train = f["train"][:]   # shape (P, T, 3)
    xyz_min = f["min"][:]   # shape (3,)
    xyz_max = f["max"][:]   # shape (3,)
```

#### Important note about `min` / `max`

In the **current geometry-conditioned branch**, the training loader normalizes positions using the geometry metadata (`origin`, `spacing`, `dims`) from the processed geometry `.npz` when `normalize_positions=True`. Therefore:

- `min` and `max` are kept mainly for compatibility / sanity checks,
- the loader does **not** use them as the primary normalization mechanism for geometry-conditioned position training.

#### If you add more channels

You may extend `C` beyond 3 (for example positions + drag force), but then:

- `--in_channels` must match the new `C`,
- the **first 3 channels must remain position channels** if you keep `normalize_positions=True` and/or SDF guidance enabled,

### 3.2 Processed geometry `.npz`

`process_geometry.py` writes a compressed `.npz` with:

- `binary`: binary occupancy grid of the lumen,
- `sdf`: signed distance field sampled on the same grid,
- `origin`: world-space grid origin,
- `spacing`: voxel spacing.

Example:

```python
geo = np.load("geometry/processed/case01_geometry.npz")
binary = geo["binary"]   # (D, H, W)
sdf    = geo["sdf"]      # (D, H, W)
origin = geo["origin"]   # (3,)
spacing= geo["spacing"]  # (3,)
```

This file is used in two different places:

- **conditioning**: the `binary` grid is passed to the geometry encoder,
- **guidance**: the `sdf` is queried during reverse diffusion to penalize samples that leave the airway.

### 3.3 Training manifest (`train_manifest.json`)

The geometry-conditioned data loader expects **a JSON manifest**, not a single HDF5 path.
Each entry tells the loader which trajectory file belongs to which geometry.

Minimal example:

```json
[
  {
    "h5_path": "datasets/case01.h5",
    "dataset_name": "train",
    "geo_path": "geometry/processed/case01_geometry.npz"
  },
  {
    "h5_path": "datasets/case02.h5",
    "dataset_name": "train",
    "geo_path": "geometry/processed/case02_geometry.npz"
  },
  {"..."}
]
```


#### How to adapt this file to your own dataset

For each geometry / subject / case:

1. create an HDF5 file containing the particle tracks,
2. create a processed geometry `.npz` from the STL,
3. add one JSON entry that points to both files.

Use paths that are valid from the directory where you launch training (for example repo-root relative paths or absolute paths).

### 3.4 Sample output `.npz`

`turb_sample.py` writes generated samples as:

```text
samples_{P}x{T}x{C}.npz
```

For example:

```text
samples_256x1024x3.npz
```

Inside that file, the default array is stored under `arr_0` and has shape:

- `(num_samples, image_size, in_channels)` = `(P, T, C)`.

#### Why `.npz` instead of `.h5` for sampled outputs?

Sampling produces a **single dense array** and does not require the metadata / chunked multi-file structure used for training.
Using `.npz` keeps the output:

- easy to save with NumPy,
- easy to move around,
- easy to load later with `np.load(...)`,
- compact enough for one-off sampling results.

If you later want chunked I/O or richer metadata, you can always convert the `.npz` into `.h5`. See `scripts/convert_np2hd5.py`.

---

## 4. End-to-end workflow

A typical workflow is:

1. Prepare STL geometry (watertight and manifold).
2. Convert STL to processed geometry (`binary` + `sdf` + metadata).
3. Convert Star-CCM+ track CSV to HDF5.
4. Build `train_manifest.json`.
5. Train the model.
6. Sample trajectories for one geometry.
7. Convert / visualize / compare results.

### 4.1 Prepare geometry from STL

#### Option A: standard geometry-conditioned path (recommended)

Use `process_geometry.py`:

```bash
python scripts/process_geometry.py \
    geometry/raw/case01.stl \
    geometry/processed/case01 \
    --dim 64
```

This writes:

```text
geometry/processed/case01_geometry.npz
```

with keys:

- `binary`
- `sdf`
- `origin`
- `spacing`

#### What `--dim` does

`--dim` controls the **maximum grid dimension** used for voxelization.
In practice:

- larger `--dim` -> finer geometric detail,
- but also larger memory footprint and slower geometry encoding.

This matters because `guided_diffusion/turb_datasets.py` loads all geometry grids and pads them to the maximum shape found in the manifest.
A larger voxel resolution therefore increases:

- CPU RAM during dataset initialization,
- GPU compute in the geometry encoder,
- disk size of the processed geometry files.

`64` is a reasonable starting point.

#### Option B: voxelization only

If you only want the binary lumen volume and not the SDF, use `stl_to_numpy.py`:

```bash
python scripts/stl_to_numpy.py \
    geometry/raw/case01.stl \
    geometry/processed/case01_binary.npy \
    --max-dim 64
```

You can also specify `--voxel-size` explicitly instead of `--max-dim`.

#### Watertight mesh requirement

`stl_to_numpy.py` expects a watertight and manifold mesh by default.
If your STL has open boundaries or leaks, fix the mesh first whenever possible.
There is a `--no-watertight-check` flag, but skipping the check is not recommended unless you know what you are doing.

#### SDF sign convention: verify once

The guidance function in `guided_diffusion/geometry_util.py` assumes a sign convention such that the penalty is zero inside the airway and positive only when a point leaves the domain. In practice, **verify the sign convention on your generated SDF once before large-scale sampling**.

If guidance appears to push particles the wrong way, you likely need to flip the sign of `sdf` or adjust the penalty in `geometry_guidance_fn`.

### 4.2 Create the training HDF5 from Star-CCM+ tracks

Use `create_dataset.py`:

```bash
python scripts/create_dataset.py \
    --csv data/case01_tracks.csv \
    --out datasets/case01.h5 \
    --train-particles 50000 \
    --train-timesteps 1024 \
    --overwrite
```

#### What this script does

- loads the Star-CCM+ track CSV,
- automatically caches a parquet faster reruns,
- extracts the `x,y,z` columns,
- sorts tracks by parcel index and time,
- truncates / pads each track to `--train-timesteps`,
- writes the result as `train` in an HDF5 file.

#### Padding behavior

If a track is shorter than `--train-timesteps`, it is padded using last-value hold (the last available position is repeated). If it is longer, it is truncated.

#### What `--train-particles` does

`--train-particles` is the number of tracks written into the HDF5 file.
Use this to build smaller subsets for ablations or to cap memory / disk usage.

### 4.3 Build the manifest

After you have created one `.h5` and one processed geometry `.npz` per case, write a `train_manifest.json` like this:

```json
[
  {
    "h5_path": "datasets/case01.h5",
    "dataset_name": "train",
    "geo_path": "geometry/processed/case01_geometry.npz"
  },
  {
    "h5_path": "datasets/case02.h5",
    "dataset_name": "train",
    "geo_path": "geometry/processed/case02_geometry.npz"
  },
  ...,
  {
    "h5_path": "datasets/caseN.h5",
    "dataset_name": "train",
    "geo_path": "geometry/processed/caseN_geometry.npz"
  }  
]
```

### 4.4 Train the model

A typical single-GPU is:

```bash
DATA_FLAGS="--dataset_path datasets/train_manifest.json --dataset_name train"
MODEL_FLAGS="--dims 1 --image_size 1024 --in_channels 3 --num_channels 128 \
 --num_res_blocks 3 --attention_resolutions 250,125 --channel_mult 1,1,2,3,4"
DIFFUSION_FLAGS="--diffusion_steps 800 --noise_schedule tanh6,1"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64"

python scripts/turb_train_monitor.py \
  $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

The multi-thread GPU input will depend on what scheduler is being used on you HPC system.

#### What training writes

Inside `log_dir` (default: `checkpoints`) you will typically get:

- `modelXXXXXX.pt` : raw model checkpoints,
- `ema_0.9999_XXXXXX.pt` : EMA checkpoints,
- `optXXXXXX.pt` : optimizer state,
- `metrics/gpu_metrics.csv` : GPU / memory monitoring (with `turb_train_monitor.py`),
- `progress/samples_step_XXXXXX.npz` : periodic sample snapshots,
- `progress/compare_step_XXXXXX.png` : optional GT-vs-snapshot comparisons,
- `progress/encoder_step_XXXXXX.npz` : optional geometry-encoder embedding dumps.

#### Important distinction: `dataset_path` vs `geometry_path`

During training:

- `--dataset_path` should point to the **manifest JSON** that maps trajectory files to geometries.
- `--geometry_path` in `turb_train_monitor.py` is **not** the training dataset geometry source. It is only used for **snapshot sampling / monitoring** with a single geometry during training.

The actual per-sample geometry conditioning during training comes from the `geo_path` entries in the manifest.

### 4.5 Run inference / sample trajectories

A typical sampling run is:

```bash
MODEL_FLAGS="--dims 1 --image_size 1024 --in_channels 3 --num_channels 128 \
 --num_res_blocks 3 --attention_resolutions 250,125 --channel_mult 1,1,2,3,4"
DIFFUSION_FLAGS="--diffusion_steps 800 --noise_schedule tanh6,1"
SAMPLE_FLAGS="--num_samples 16384 --batch_size 128 \
 --model_path checkpoints/9/ema_0.9999_XXXXXXXX.pt \
 --results_dir results/XX \
 --guidance_scale 1.0 \
 --geometry_path geometry/processed/geometry_XX.npz"

python scripts/turb_sample.py \
  $SAMPLE_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS
```

This will write something like:

```text
results/XX/samples_16384x1024x3.npz
```

#### What `geometry_path` is used for during sampling

The same geometry `.npz` provides:

- `binary` -> passed into the geometry encoder for conditioning,
- `sdf` -> queried in `geometry_guidance_fn` to apply SDF-based guidance,
- `origin`, `spacing` -> used to interpret the coordinate system.

### 4.6 Convert sampled trajectories back to world coordinates

If you train with:

- `normalize_positions=True`
- `normalization_space="normalized"`

then the model operates in **normalized geometry coordinates** and the output `.npz` is also in that normalized space.

### 4.7 Compare / visualize trajectories

A typical comparison call is:

```bash
python scripts/compare_trajectories.py \
  --gt datasets/case01.h5 --gt-key train \
  --pred results/XX/samples_Px1024x3.npz --pred-key arr_0 \
  --geometry-npz geometry/processed/case01_geometry.npz \
  --normalization-space normalized \
  --stl geometry/raw/case01.stl \
  --out results/XX/
```

This script can load:

- GT from `.h5`, `.npy`, or `.npz`,
- predictions from `.npz`, `.npy`, or `.h5`,
- optional STL overlays for context.

---

## 5. How the model uses geometry

At a high level, geometry enters the model in two places:

1. **conditioning during denoising**
2. **guidance during sampling**

### 5.1 Conditioning during training and inference

The geometry encoder receives the voxelized airway occupancy grid and maps it to a latent embedding. That embedding is added to the diffusion timestep embedding and injected into the 1D U-Net denoiser.

Effectively:

- the **trajectory signal** is 1D in time for each position,
- the **conditioning signal** is 3D in space,
- the geometry encoder turns the 3D geometry into a compact vector that the denoiser can use at every diffusion step.

### 5.2 Guidance at inference time

During sampling, `geometry_guidance_fn` queries the SDF at each predicted point and computes a penalty for points that leave the geometry. The gradient of that penalty is used as a guidance term in the reverse diffusion process.

Practical meaning:

- if a point is already inside the lumen, guidance does nothing,
- if a point drifts outside, the SDF gradient nudges it back inward by affecting the denoising predicted mean,
- `guidance_scale` controls how strongly that correction is applied.

---

## 6. Hyperparameter guide

Below is a guide to the most important hyperparameters in this branch.

### 6.1 Data / geometry preprocessing

| Argument | Where | Meaning | Effect |
|---|---|---|---|
| `--train-particles` | `create_dataset.py` | Number of particle tracks to write into the HDF5 file. | Larger values give more training data; also larger training time. |
| `--train-timesteps` | `create_dataset.py` | Fixed number of timesteps per track. | Must match `--image_size` during training. |
| `--dim` | `process_geometry.py` | Target max grid dimension for the geometry voxelization. | Larger grids preserve more detail but increase CPU/GPU memory and geometry-encoder cost. |
| `--max-dim` | `stl_to_numpy.py` | Same idea as above for raw voxelization. | Use when you want binary volume only. |
| `--voxel-size` | `stl_to_numpy.py` | Explicit voxel spacing in STL units. | More control than `--max-dim`, but requires choosing a spacing manually. |

### 6.2 Dataset loader / normalization

| Argument | Where | Meaning | Effect |
|---|---|---|---|
| `--dataset_path` | train scripts | Path to the JSON manifest. | This is the main training input. It should point to a manifest, not a single HDF5 file. |
| `--dataset_name` | train scripts | Name of the dataset inside each HDF5 file. | Usually `train`. |
| `--normalize_positions` | train scripts | Whether to convert the first 3 channels from world coordinates into geometry-relative coordinates. | Keep this `True` for the current geometry-conditioned workflow. |
| `--normalization_space` | train scripts | Position space after normalization. Typically `normalized` or `voxel`. | `normalized` maps positions to `[-1,1]` for each channel, which is the default and the safest choice. |
| `--cache_h5` | train scripts | Keep HDF5 handles open during training. | Faster repeated access, but more file handles / memory. |

### 6.3 Model architecture

| Argument | Meaning | Practical effect |
|---|---|---|
| `--dims` | Dimensionality of the signal processed by the U-Net. | Use `1` for trajectory sequences. |
| `--image_size` | Sequence length seen by the model. | For position tracks, set this equal to the number of timesteps in the HDF5 `train` tensor. |
| `--in_channels` | Number of channels per timestep. | `3` for `(x,y,z)` positions. Add more as needed |
| `--num_channels` | Base U-Net width. | Larger values increase capacity and GPU memory. |
| `--num_res_blocks` | Number of residual blocks per level. | More blocks increase capacity and runtime. |
| `--attention_resolutions` | Resolutions where self-attention is inserted. | Helps model long-range time dependencies; more attention usually increases memory use. |
| `--channel_mult` | Channel multiplier per U-Net level. | Controls the width growth through the encoder / decoder. |
| `--dropout` | Dropout in residual blocks. | Usually keep at `0.0` unless regularization is needed. |


### 6.4 Geometry encoder hyperparameters

| Argument | Meaning | Practical effect |
|---|---|---|
| `--geometry_encoder_type` | Geometry encoder variant (`variational` or `deterministic`). | `variational` adds a KL regularizer and a sampled latent; `deterministic` uses a direct embedding. Using deterministic is much more stringent and will not adapt to out-of-sample geometries.|
| `--geometry_sample` | If using a variational encoder, sample from the latent distribution or use the mean. | Sampling can improve latent coverage but adds stochasticity. |
| `--geometry_kl_weight` | Weight of the KL term. | Higher values regularize the latent geometry space more strongly but may weaken conditioning if too large. |

The default branch behavior is configured around a variational geometry encoder with:

```bash
--geometry_encoder_type variational
--geometry_sample True
--geometry_kl_weight 1e-4
```

### 6.5 Diffusion process

| Argument | Meaning | Practical effect |
|---|---|---|
| `--diffusion_steps` | Number of diffusion steps. | More steps improve fidelity but increase training and sampling cost. |
| `--noise_schedule` | Beta schedule. | Affects how noise is added / removed across time. |
| `--timestep_respacing` | Optional accelerated sampling schedule. | Leave empty for full-step DDPM; use only if you know you want respaced sampling. |
| `--learn_sigma` | Whether the model predicts variance as well as mean/noise. | By default `False` in this work. |
| `--predict_xstart` | Whether the model predicts `x_0` instead of epsilon. | Usually `False`; this work uses epsilon-prediction. |

Your current configuration:

```bash
--diffusion_steps 800
--noise_schedule tanh6,1
```

### 6.6 Training

| Argument | Meaning | Practical effect |
|---|---|---|
| `--lr` | Learning rate. | Higher can speed convergence but destabilize training. |
| `--batch_size` | Batch size per rank. | In distributed training, global batch size = `world_size * batch_size`. |
| `--microbatch` | Gradient accumulation chunk size. | Useful when GPU memory is limited. |
| `--ema_rate` | Exponential moving average decay. | EMA checkpoints usually sample better than raw checkpoints. |
| `--weight_decay` | AdamW weight decay. | Regularization; current default is `0.0`. |
| `--resume_checkpoint` | Resume training from a saved checkpoint. | Continue interrupted runs. |
| `--use_fp16` | Mixed precision. | Can reduce memory and speed up training, but keep off if debugging instability. |
| `--save_interval` | Checkpoint interval. | Shorter intervals give more recovery points but more disk usage. |
| `--log_interval` | Logging interval. | More frequent logs are useful for debugging but noisier. |

Current examples use:

```bash
--lr 1e-4
--batch_size 128
```

`guided_diffusion/train_util.py` uses **AdamW** and maintains EMA checkpoints.

### 6.7 Training monitor only

| Argument | Meaning | Observations |
|---|---|---|
| `--metrics_interval` | How often GPU/memory stats are written. | Useful for profiling. |
| `--snapshot_interval` | How often a sample snapshot is generated during training. | Very useful for checking progress, but slows training if too frequent. |
| `--eval_batch` | Number of trajectories generated in each monitoring snapshot. | Controls snapshot cost. |
| `--gt_npy` | Optional GT `.npy` for comparison plots during training. | Monitoring only; not used for optimization. |
| `--geometry_path` | Single processed geometry used for monitoring snapshots. | Monitoring only; does not replace the manifest. |
| `--encoder_manifest` | Optional manifest whose geometries are passed through the encoder for latent snapshots. | Useful for checking latent collapse / separation. |
| `--save_init_model` | Save the randomly initialized model before training. | Helps reproducibility or controlled restarts. |
| `--init_model_save_path` | Where to save that initial model. | Use with `--save_init_model`. |
| `--init_model_load_path` | Load a fixed initial model before training. | Useful if you want multiple runs to start from exactly the same weights. |

### 6.8 Sampling / inference

| Argument | Meaning | Practical effect |
|---|---|---|
| `--num_samples` | Number of trajectories to generate. | More samples cost more time, but is linear. |
| `--model_path` | Checkpoint to sample from. | Use EMA checkpoints unless you specifically want the raw model. |
| `--results_dir` | Output directory. | Generated `.npz` is saved here. |
| `--geometry_path` | Processed geometry `.npz` used for conditioning and guidance. | Required for geometry-conditioned sampling. |
| `--guidance_scale` | Strength of SDF guidance. | Higher values enforce geometry more strongly, but too high can distort the learned statistics. |
| `--coord_space` | Coordinate space expected by the guidance function. | Must match the model output space; usually `normalized`. |
| `--use_ddim` | Use DDIM instead of DDPM sampling. | Faster / different sampling behavior if enabled. |
| `--clip_denoised` | Clamp denoised predictions. | Usually improves stability. |

A good default starting point is:

```bash
--guidance_scale 1.0
--coord_space normalized
```

---

## 7. Expected tensor layouts

This repo uses a few different tensor layouts depending on where you are in the pipeline.

### In HDF5 / on disk

Standard training layout:

```text
(P, T, C)
```

Example:

```text
(16384, 1024, 3)
```

### Inside the training loader

The loader moves the channel axis to the front:

```text
(C, T)
```

for each sample, then batches them as:

```text
(B, C, T)
```

This is what the 1D U-Net expects.

### Geometry grid layout

The binary geometry is loaded as:

```text
(B, 1, D, H, W)
```

where the leading `1` is the single occupancy channel.

---

## 8. Common pitfalls

### 8.1 `--image_size` must match the HDF5 timestep length

If your HDF5 `train` dataset is `(P, 1024, 3)`, then training must use:

```bash
--image_size 1024
```

### 8.2 `--in_channels` must match the channel count in HDF5

If `train.shape[-1] == 3`, use:

```bash
--in_channels 3
```

### 8.3 Keep the first 3 channels as positions

If you extend the input with extra features, the first three channels should still be `(x,y,z)` unless you also rewrite the normalization / guidance logic.

### 8.4 Output `.npz` files are typically still in model space

If you trained with normalized positions, the generated `samples_*.npz` will also be in normalized coordinates until you map them back using the geometry metadata.

### 8.5 Verify the SDF sign once

If guidance appears to push particles into walls or away from the lumen, check the SDF sign convention first.

### 8.6 Geometry grids are loaded into memory at dataset initialization

If you create many large geometries (for example very large `--dim`), startup memory can grow quickly because all geometry grids are loaded and padded to a common shape.

### 8.7 Parquet caching is your friend

On the first `create_dataset.py` run, the CSV is cached as a parquet file.
Subsequent reruns are much faster.

---

## 9. Recommended minimal checklist before launching a new experiment

- [ ] STL is watertight or intentionally cleaned.
- [ ] `process_geometry.py` ran successfully and produced `binary`, `sdf`, `origin`, `spacing`.
- [ ] HDF5 `train` tensor has shape `(P, T, 3)`.
- [ ] `--image_size == T`.
- [ ] `--in_channels == 3`.
- [ ] Manifest entries correctly pair each `.h5` with the matching geometry `.npz`.
- [ ] `--geometry_path` for inference points to the same geometry representation you want to condition on.
- [ ] If sampling uses guidance, `--coord_space` matches the model output space.
- [ ] You verified the SDF sign convention on one test case.

---

## 10. Summary

If you are new to this branch, the shortest useful path is:

1. **STL -> geometry**
   ```bash
   python scripts/process_geometry.py geometry/raw/case01.stl geometry/processed/case01 --dim 64
   ```
2. **StarCCM+ CSV -> HDF5**
   ```bash
   python scripts/create_dataset.py --csv data/case01_tracks.csv --out datasets/case01.h5 --train-particles 50000 --train-timesteps 1024
   ```
3. **Create `train_manifest.json`** linking `case01.h5` and `case01_geometry.npz`.
4. **Train** with `scripts/turb_train_monitor.py`.
5. **Sample** with `scripts/turb_sample.py`.
6. **Visualize** with `scripts/compare_trajectories.py`.

That is the complete geometry-conditioned workflow.

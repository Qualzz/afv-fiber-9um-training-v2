# AFV V2 trainer

PyTorch training for [AFV fibre model V2](https://huggingface.co/Qualzz20/afv_fiber_9um_v2): H/V segmentation, fibre centres, local affinities and axial direction. Input: grayscale CT at 8.64–9.6 µm per voxel. Training patches are 128³ voxels.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use a PyTorch build compatible with the selected CUDA device. CPU execution is supported for testing.

## Data

Prepare local CT windows and spline annotations. A manifest is a JSON list; paths are relative to the manifest. See [examples/manifest.json](examples/manifest.json).

Each entry specifies:

- `id`: unique window name.
- `scroll`: scroll identifier, used for equal sampling across scrolls.
- `split`: `train` or `validation`. Use spatially separate windows.
- `ct`: a uint8 `.npy` array in **Z, Y, X** order, on an isotropic grid. Use 640³ windows; the minimum is 512³.
- `annotations`: a `.npz` file with the arrays below.
- `voxel_size_um`: physical voxel size, between 8.64 and 9.6.
- Optional `mean` and `std`: CT normalization statistics. Otherwise, statistics are computed from the window.

Annotation arrays:

| Array | Shape | Meaning |
|---|---|---|
| `points_zyx` | N × 3 | Floating-point centre coordinates in the local CT window, in voxels |
| `instances` | N | Positive integer ID for each uninterrupted observed fibre run |
| `families` | N | `1` for vertical, `2` for horizontal |

Store each instance in one contiguous, ordered run, sampled at 0.5-voxel arc-length intervals. Assign separate IDs across missing or unobserved sections. Coordinates must already match the CT window's origin and voxel spacing.

To convert polylines, use an NPZ containing `points_zyx`, `offsets` of length K+1, and `family` of length K. Optional `valid_edges` has length N−1 and marks observed edges between consecutive points; without it, every within-polyline edge is treated as observed.

```bash
python prepare_annotations.py polylines.npz annotations.npz
```

The converter splits invalid edges, trims two voxels from each run end and samples at 0.5-voxel intervals. CT windows and annotations must be supplied separately.

## Train

Download `checkpoint_final.pth` from the [model files](https://huggingface.co/Qualzz20/afv_fiber_9um_v2/tree/main), then run:

```bash
python train.py \
  --manifest data/manifest.json \
  --init-checkpoint weights/checkpoint_final.pth \
  --output runs/finetune \
  --steps 1000
```

Settings are in [configs/train.json](configs/train.json). Use `--config` for another configuration and `--device cuda:0` to choose a device. The default effective batch is 8, with two microbatches of 4. If memory is insufficient, reduce `microbatch_size` to a divisor of `batch_size`. Loss reductions are computed per microbatch, so changing its size can change the optimization.

The loader interleaves four windows, draws 32 examples per window, and gives each scroll one window per shuffled cycle. It keeps up to `cache_windows` prepared windows per loader process; increasing `workers` also increases memory use. Provide the desired CT source variants as manifest entries.

Compression deforms CT and annotations together. Local CT occultations preserve the annotations. Semantic supervision uses observed spline cells; unannotated voxels are ignored. The objective combines segmentation, centre, affinity, axial-direction and probability-field geometry terms. Their settings are defined in the configuration and [losses.py](afv_training/losses.py).

## Checkpoints and resume

- `last.pt`: model, EMA, optimizer, random state and sample cursor.
- `raw.pth` / `ema.pth`: inference weights, updated at checkpoint intervals.
- `metrics.jsonl`: training losses and periodic validation losses for raw and EMA weights.

```bash
python train.py \
  --manifest data/manifest.json \
  --resume runs/finetune/last.pt \
  --output runs/finetune \
  --steps 2000
```

`--steps` is the total number of optimizer updates in the run, including completed updates. `--stop-after N` saves and exits after N additional updates. The checkpoint's global step is reported separately. Initializing from exported inference weights starts a fresh optimizer; `--resume` restores its saved state. Preserve input files and training code when resuming.

Validation reports the specified losses on fixed, unaugmented examples. It does not select a checkpoint automatically or measure anatomical accuracy.

## Tests

```bash
pip install pytest
python -m pytest -q
```

## License

Apache-2.0 for this code. The architecture is provided by [dynamic-network-architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures); PyTorch, NumPy, SciPy and Numba retain their respective licenses. Model weights and CT/annotation datasets retain their own terms.

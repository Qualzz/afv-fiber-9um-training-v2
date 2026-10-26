"""Bounded local windows and deterministic, scroll-balanced sampling."""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from .augmentations import occlude, sample


def load_manifest(path, split):
    path = Path(path).resolve()
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("The manifest must be a nonempty list of windows")
    names = [r["id"] for r in records]
    if len(set(names)) != len(names):
        raise ValueError("Window IDs must be unique")
    # Catch direct file reuse across splits. Physical overlap still needs checking
    # when preparing windows from a larger scan.
    split_by_ct = {}
    result = []
    for record in records:
        if record["split"] not in ("train", "validation"):
            raise ValueError("split must be train or validation")
        row = dict(record)
        for key in ("ct", "annotations"):
            row[key] = (path.parent / row[key]).resolve()
            if not row[key].is_file():
                raise FileNotFoundError(row[key])
        previous = split_by_ct.setdefault(row["ct"], row["split"])
        if previous != row["split"]:
            raise ValueError("A CT file appears in both training and validation")
        if not 8.64 <= float(row["voxel_size_um"]) <= 9.6:
            raise ValueError("Prepare isotropic CT at 8.64–9.6 µm per voxel")
        if row["split"] == split:
            result.append(row)
    if not result:
        raise ValueError(f"No {split} windows in the manifest")
    return result


def load_window(record):
    ct = np.load(record["ct"], mmap_mode="r", allow_pickle=False)
    if ct.dtype != np.uint8 or ct.ndim != 3 or min(ct.shape) < 512:
        raise ValueError("CT must be a uint8 ZYX .npy window, at least 512³")
    with np.load(record["annotations"], allow_pickle=False) as data:
        points = np.array(data["points_zyx"], dtype=np.float32)
        instances = np.array(data["instances"], dtype=np.int32)
        families = np.array(data["families"], dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("points_zyx must be a nonempty N×3 array")
    if instances.shape != (len(points),) or families.shape != (len(points),):
        raise ValueError("One instance ID and family are required per point")
    if (
        not np.isfinite(points).all()
        or (points < 0).any()
        or (points >= ct.shape).any()
    ):
        raise ValueError("Points must be finite, in local CT voxel coordinates")
    if (instances <= 0).any() or not np.isin(families, (1, 2)).all():
        raise ValueError("Use positive instance IDs; family 1=V, 2=H")
    starts = np.r_[True, instances[1:] != instances[:-1]]
    if len(np.unique(instances)) != int(starts.sum()):
        raise ValueError("Each instance must occupy one contiguous point run")
    same = instances[1:] == instances[:-1]
    if np.any(families[1:][same] != families[:-1][same]):
        raise ValueError("Each instance must have one family")
    # Statistics are computed in bounded z slices if no region statistics exist.
    if "mean" in record and "std" in record:
        mean, std = float(record["mean"]), float(record["std"])
    else:
        hist = np.zeros(256, dtype=np.int64)
        for z in range(0, len(ct), 16):
            hist += np.bincount(ct[z : z + 16].ravel(), minlength=256)
        values = np.arange(256, dtype=np.float64)
        mean = float(hist @ values / hist.sum())
        std = float(np.sqrt(hist @ ((values - mean) ** 2) / hist.sum()))
    if not np.isfinite([mean, std]).all() or std < 0:
        raise ValueError("Invalid normalization statistics")
    return dict(
        name=record["id"],
        ct=ct,
        points=points,
        instances=instances,
        families=families,
        tree=cKDTree(points),
        mean=mean,
        std=max(std, 10.0),
    )


class FiberDataset(Dataset):
    """Four interleaved windows, 32 examples per window, equal scroll quotas.

    Every shuffled cycle visits each scroll once, independent of its number of
    windows. Within each scroll the windows rotate. Sample seeds depend only on
    the index, so DataLoader prefetch cannot advance the saved training cursor.
    """

    def __init__(
        self,
        manifest,
        split,
        count,
        start=0,
        seed=301018,
        cache_windows=4,
        occlusion_probability=0.25,
    ):
        self.records = load_manifest(manifest, split)
        self.split, self.count, self.start = split, count, start
        self.seed, self.cache_windows = seed, cache_windows
        self.occlusion_probability = occlusion_probability
        self.cache = OrderedDict()
        self.by_scroll = {}
        for i, record in enumerate(self.records):
            self.by_scroll.setdefault(record["scroll"], []).append(i)
        self.scrolls = sorted(self.by_scroll)

    def __len__(self):
        return self.count

    def descriptor(self, index):
        wave, position = divmod(index, 128)
        group = wave * 4 + position % 4
        cycle, slot = divmod(group, len(self.scrolls))
        order = np.random.default_rng([self.seed, cycle]).permutation(self.scrolls)
        choices = self.by_scroll[str(order[slot])]
        record_index = choices[cycle % len(choices)]
        sample_index = group * 32 + (position // 4 + group % 4) % 32
        return record_index, sample_index

    def __getitem__(self, index):
        index += self.start
        if self.split == "train":
            record_index, sample_index = self.descriptor(index)
        else:
            record_index = index % len(self.records)
            sample_index = index
        if record_index not in self.cache:
            while len(self.cache) >= self.cache_windows:
                self.cache.popitem(last=False)
            self.cache[record_index] = load_window(self.records[record_index])
        self.cache.move_to_end(record_index)
        row = sample(
            self.cache[record_index], sample_index, augment=self.split == "train"
        )
        if self.split == "train":
            row = occlude(row, probability=self.occlusion_probability)
        meta = row.pop("meta")
        row["hard"] = meta["kind"] == "hard" and not meta["compression_rejected"]
        row["occluded"] = meta.get("occlusion", {}).get("applied", False)
        return {
            key: torch.as_tensor(np.ascontiguousarray(value))
            if isinstance(value, np.ndarray)
            else torch.tensor(value)
            for key, value in row.items()
        }


def input_fingerprint(manifest):
    """Bind resumes to the manifest and annotation contents, plus CT metadata."""
    path = Path(manifest).resolve()
    digest = hashlib.sha256(path.read_bytes())
    for row in json.loads(path.read_text()):
        annotation = (path.parent / row["annotations"]).resolve()
        digest.update(annotation.read_bytes())
        ct = (path.parent / row["ct"]).resolve()
        stat = ct.stat()
        digest.update(f"{ct}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()

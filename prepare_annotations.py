"""Convert local ZYX polylines to the dense annotation arrays used by training."""

import argparse
from pathlib import Path

import numpy as np


def prepare(source, destination):
    with np.load(source, allow_pickle=False) as data:
        points = np.asarray(data["points_zyx"], dtype=np.float64)
        offsets = np.asarray(data["offsets"], dtype=np.int64)
        families = np.asarray(data["family"], dtype=np.uint8)
        # valid_edges[i] describes points[i] -> points[i+1]. If omitted, every
        # within-polyline edge is explicitly treated as observed.
        edges = (
            np.asarray(data["valid_edges"], dtype=bool)
            if "valid_edges" in data
            else np.ones(len(points) - 1, bool)
        )
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite N×3 points_zyx")
    if (
        offsets.ndim != 1
        or len(offsets) != len(families) + 1
        or offsets[0] != 0
        or offsets[-1] != len(points)
        or np.any(np.diff(offsets) < 2)
    ):
        raise ValueError(
            "Offsets must partition points into polylines of at least two points"
        )
    if not np.isin(families, (1, 2)).all() or edges.shape != (len(points) - 1,):
        raise ValueError("Invalid families or valid_edges")
    dense, ids, labels = [], [], []
    instance = 0
    for a, b, family in zip(offsets[:-1], offsets[1:], families):
        boundaries = np.r_[a, a + np.flatnonzero(~edges[a : b - 1]) + 1, b]
        for first, last in zip(boundaries[:-1], boundaries[1:]):
            line = points[first:last]
            if len(line) < 2:
                continue
            keep = np.r_[True, np.linalg.norm(np.diff(line, axis=0), axis=1) > 1e-8]
            line = line[keep]
            if len(line) < 2:
                continue
            arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
            # Same endpoint trim and source sampling interval as the final run.
            distance = np.arange(2.0, arc[-1] - 2.0, 0.5)
            if not len(distance):
                continue
            sampled = np.stack(
                [np.interp(distance, arc, line[:, k]) for k in range(3)], axis=1
            )
            instance += 1
            dense.append(sampled.astype(np.float32))
            ids.append(np.full(len(sampled), instance, np.int32))
            labels.append(np.full(len(sampled), family, np.uint8))
    if not dense:
        raise ValueError(
            "No observed run is long enough after trimming two voxels at each end"
        )
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    np.savez_compressed(
        destination,
        points_zyx=np.concatenate(dense),
        instances=np.concatenate(ids),
        families=np.concatenate(labels),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.destination)

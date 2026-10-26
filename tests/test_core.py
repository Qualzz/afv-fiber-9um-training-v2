import numpy as np
import pytest
import torch

from afv_training.augmentations import occlude
from afv_training.compression import Compaction
from afv_training.geometry_loss import energy_loss, targets, tensors
from afv_training.interpolation import map_coordinates
from afv_training.losses import affinity_targets, loss_for
from afv_training.targets import output_core, spline_tangents
from prepare_annotations import prepare


def test_interpolation_matches_scipy():
    from scipy.ndimage import map_coordinates as scipy_map

    rng = np.random.default_rng(42)
    volume = rng.integers(0, 255, (10, 12, 14), dtype=np.uint8)
    coordinates = rng.uniform(-1, 15, (3, 1000)).astype(np.float32)
    for mode in ("constant", "nearest"):
        expected = scipy_map(
            volume, coordinates, order=1, mode=mode, output="f4", prefilter=False
        )
        actual = map_coordinates(
            volume, coordinates, order=1, mode=mode, output="f4", prefilter=False
        )
        np.testing.assert_array_equal(actual, expected)


def test_no_spline_bridge_across_gap_or_instance():
    points = np.array(
        [[10.0, 10, 10], [10.5, 10, 10], [11, 10, 10], [18, 10, 10], [18.5, 10, 10]]
    )
    ids = np.array([1, 1, 1, 1, 2])
    direction, valid = spline_tangents(points, ids, np.arange(5), points)
    assert valid.tolist() == [False, True, False, False, False]
    np.testing.assert_allclose(direction[1], [0, 0, 1])
    core = output_core(points, np.arange(5), ids, np.ones(5, np.uint8), points, size=32)
    assert not core["core"][13:17, 10, 10].any()


def test_unknown_and_inter_instance_affinities_ignored():
    semantic = torch.full((1, 8, 8, 8), 255, dtype=torch.long)
    instances = torch.zeros_like(semantic)
    semantic[:, 2, 2, 2:5] = 1
    instances[:, 2, 2, 2:4] = 1
    instances[:, 2, 2, 4] = 2
    target, valid = affinity_targets(instances, semantic)
    assert target[0, 2, 2, 2, 2] == 1
    assert valid[0, 2, 2, 2, 2] == 1
    assert valid[0, 2, 2, 2, 3] == 0
    assert valid[0, 2, 2, 2, 4] == 0


def test_occlusion_preserves_targets():
    row = dict(
        ct=np.ones((1, 128, 128, 128), np.float32),
        centre=np.zeros((128, 128, 128), np.uint8),
        instances=np.zeros((128, 128, 128), np.int32),
        direction_valid=np.zeros((128, 128, 128), np.uint8),
        direction=np.zeros((3, 128, 128, 128), np.float32),
        meta={"index": 7},
    )
    for key in ("centre", "instances", "direction_valid"):
        row[key][20:108, 64, 64] = 1
    row["direction"][2, 20:108, 64, 64] = 1
    result = occlude(row, force=True)
    assert result["meta"]["occlusion"]["applied"]
    assert np.any(result["ct"] != row["ct"])
    for key in ("centre", "instances", "direction_valid", "direction"):
        np.testing.assert_array_equal(result[key], row[key])
    assert np.all(row["ct"] == 1)


def test_compaction_roundtrip():
    ct = np.full((512, 512, 512), 100, np.uint8)
    warp = Compaction(ct, [255.5] * 3, [1.0, 0.0, 0.0], 0.7, 0.05)
    points = np.random.default_rng(3).uniform(-40, 40, (3, 32)).astype(np.float32)
    source = warp.source_coordinates(points)
    restored, retained = warp.output_points(source.T, return_indices=True)
    np.testing.assert_array_equal(retained, np.arange(32))
    np.testing.assert_allclose(restored, points.T, atol=0.032)


def test_geometry_prefers_correct_axis_and_has_gradient():
    torch.set_num_threads(2)
    q = torch.arange(64, dtype=torch.float32) - 31.5
    z, y, x = torch.meshgrid(q, q, q, indexing="ij")
    tube = (
        torch.exp(-(x * x + y * y) / 18)[None, None]
        .repeat(1, 2, 1, 1, 1)
        .requires_grad_()
    )
    tensor = tensors(tube)
    good = torch.zeros_like(tensor)
    bad = torch.zeros_like(tensor)
    good[:, :, 0] = 1
    bad[:, :, 2] = 1
    valid = torch.zeros((1, 2, 32, 32, 32), dtype=torch.bool)
    valid[:, :, 14:18, 15:17, 15:17] = True
    aligned = energy_loss(tensor, good, valid, [1.0])
    wrong = energy_loss(tensor, bad, valid, [1.0])
    assert aligned < 0.01 and wrong > 0.9
    wrong.backward()
    assert torch.isfinite(tube.grad).all() and tube.grad.abs().sum() > 0


def test_geometry_targets_are_sign_invariant_and_mask_unknown():
    batch = dict(
        semantic=torch.full((1, 64, 64, 64), 255),
        centre=torch.zeros((1, 64, 64, 64)),
        direction_valid=torch.zeros((1, 64, 64, 64)),
        direction=torch.zeros((1, 3, 64, 64, 64)),
    )
    batch["semantic"][:, 16:48, 32, 32] = 1
    batch["centre"][:, 16:48, 32, 32] = 1
    batch["direction_valid"][:, 16:48, 32, 32] = 1
    batch["direction"][:, 2] = 1
    moments, valid = targets(batch)
    batch["direction"] *= -1
    inverse, inverse_valid = targets(batch)
    assert (
        torch.equal(moments, inverse)
        and torch.equal(valid, inverse_valid)
        and valid.any()
    )
    batch["semantic"].fill_(255)
    assert not targets(batch)[1].any()


def test_empty_labels_have_no_gradient():
    torch.manual_seed(4)
    shape = (2, 16, 16, 16)
    output = {
        name: torch.randn(2, channels, 16, 16, 16, requires_grad=True)
        for name, channels in (
            ("seg", 4),
            ("centre", 1),
            ("affinity", 6),
            ("direction", 3),
        )
    }
    output["deep"] = []
    batch = {
        key: torch.zeros(shape)
        for key in (
            "centre",
            "presence",
            "distance",
            "valid",
            "target",
            "direction_valid",
        )
    }
    batch["semantic"] = torch.full(shape, 255, dtype=torch.long)
    batch["instances"] = torch.zeros(shape, dtype=torch.long)
    batch["direction"] = torch.zeros((2, 3, 16, 16, 16))
    base, direction, _ = loss_for(output, batch, [1.0, 2.0])
    assert base == 0 and direction == 0
    (base + direction).backward()
    assert all(not value.grad.any() for key, value in output.items() if key != "deep")


def test_annotation_preparation_splits_missing_edges(tmp_path):
    points = np.stack([np.arange(30), np.zeros(30), np.zeros(30)], axis=1)
    edges = np.ones(29, bool)
    edges[14] = False
    source, destination = tmp_path / "source.npz", tmp_path / "dense.npz"
    np.savez(source, points_zyx=points, offsets=[0, 30], family=[1], valid_edges=edges)
    prepare(source, destination)
    with np.load(destination) as data:
        assert set(data["instances"]) == {1, 2}
        assert not (
            (data["points_zyx"][:, 0] > 12) & (data["points_zyx"][:, 0] < 17)
        ).any()
    with pytest.raises(FileExistsError):
        prepare(source, destination)

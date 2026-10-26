import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from afv_training import train


class FixedDataset(Dataset):
    """Deterministic tensor fixture for checkpoint/update tests."""

    def __init__(self, manifest, split, count, start=0, **kwargs):
        self.count, self.start = count, start

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        size = 32
        generator = torch.Generator().manual_seed(index + self.start)
        semantic = torch.full((size, size, size), 255, dtype=torch.long)
        semantic[8:24, 16, 16] = 1
        centre = (semantic == 1).float()
        direction = torch.zeros((3, size, size, size))
        direction[2] = centre
        return dict(
            ct=torch.randn((1, size, size, size), generator=generator),
            semantic=semantic,
            centre=centre,
            presence=centre,
            instances=centre.long(),
            distance=1 - centre,
            valid=centre,
            target=centre,
            direction=direction,
            direction_valid=centre,
            hard=torch.tensor(False),
            occluded=torch.tensor(False),
        )


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_equal(left, right)
    else:
        assert a == b


def test_resume_matches_uninterrupted_updates(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "FiberDataset", FixedDataset)
    monkeypatch.setattr(train, "input_fingerprint", lambda _: "fixed-fixture")
    root = Path(__file__).resolve().parents[1]
    plans = json.loads((root / "configs/plans.json").read_text())
    architecture = plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]
    architecture.update(
        n_stages=3,
        features_per_stage=[2, 4, 8],
        kernel_sizes=[[3, 3, 3]] * 3,
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2]],
        n_blocks_per_stage=[1, 1, 1],
        n_conv_per_stage_decoder=[1, 1],
    )
    plan_path = tmp_path / "plans.json"
    plan_path.write_text(json.dumps(plans))
    config = json.loads((root / "configs/train.json").read_text())
    config.update(
        steps=2,
        batch_size=2,
        microbatch_size=1,
        cpu_threads=2,
        precision="fp32",
        validate_every=1,
        validation_samples=1,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    common = [
        "--manifest",
        str(tmp_path / "fixture.json"),
        "--config",
        str(config_path),
        "--plans",
        str(plan_path),
        "--device",
        "cpu",
    ]
    train.main(common + ["--output", str(tmp_path / "full"), "--from-scratch"])
    train.main(
        common
        + ["--output", str(tmp_path / "resumed"), "--from-scratch", "--stop-after", "1"]
    )
    train.main(
        common
        + [
            "--output",
            str(tmp_path / "resumed"),
            "--resume",
            str(tmp_path / "resumed/last.pt"),
        ]
    )
    full = torch.load(tmp_path / "full/last.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/last.pt", weights_only=True)
    assert_equal(full, resumed)
    assert (tmp_path / "full/metrics.jsonl").read_bytes() == (
        tmp_path / "resumed/metrics.jsonl"
    ).read_bytes()
    config["hard_weight"] = 1.0
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="hard_weight"):
        train.main(
            common
            + [
                "--output",
                str(tmp_path / "resumed"),
                "--resume",
                str(tmp_path / "resumed/last.pt"),
                "--steps",
                "3",
            ]
        )

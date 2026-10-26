"""Train the final AFV V2 recipe on a single CPU or CUDA device."""

import argparse
import copy
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import FiberDataset, input_fingerprint
from .geometry_loss import loss as geometry_loss
from .losses import loss_for
from .model import FiberDetector

ROOT = Path(__file__).resolve().parents[1]


def save_checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def initialize(model, checkpoint):
    """Accept the public V2 export or a backbone-only V1 nnU-Net export."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.backbone.load_state_dict(state["network_weights"], strict=True)
    auxiliary = state.get("auxiliary_weights")
    if auxiliary is not None:
        weights = {"backbone." + k: v for k, v in state["network_weights"].items()}
        weights.update(auxiliary)
        model.load_state_dict(weights, strict=True)
    else:
        print(
            "Backbone-only initialization: centre, affinity and direction heads start fresh.",
            flush=True,
        )
    return int(state.get("step", 0))


def optimizer_for(model, config):
    return torch.optim.AdamW(
        [
            {
                "params": [
                    p
                    for name, p in model.named_parameters()
                    if not name.startswith("direction.")
                ],
                "lr": config["backbone_lr"],
            },
            {
                "params": list(model.direction.parameters()),
                "lr": config["direction_lr"],
            },
        ],
        weight_decay=config["weight_decay"],
    )


def total_loss(output, batch, config):
    weights = torch.where(batch["hard"], config["hard_weight"], 1.0)
    base, direction, parts = loss_for(output, batch, weights)
    geometry = geometry_loss(output["seg"], batch, weights)
    total = (
        base
        + config["direction_weight"] * direction
        + config["geometry_weight"] * geometry
    )
    return total, {
        **parts,
        "direction": direction.detach(),
        "geometry": geometry.detach(),
    }


def precision_context(device, config):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if (device.type == "cuda" and config["precision"] == "bf16")
        else nullcontext()
    )


@torch.inference_mode()
def validate(model, loader, device, config):
    model.eval()
    totals = {}
    count = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with precision_context(device, config):
            output = model(batch["ct"])
        loss, parts = total_loss(output, batch, config)
        for key, value in {"loss": loss, **parts}.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        count += 1
    if not count:
        raise ValueError("Empty validation loader")
    return {k: v / count for k, v in totals.items()}


def check_config(config):
    for key in (
        "steps",
        "batch_size",
        "microbatch_size",
        "cache_windows",
        "cpu_threads",
        "checkpoint_every",
        "validate_every",
        "validation_samples",
    ):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["batch_size"] % config["microbatch_size"]:
        raise ValueError("batch_size must be divisible by microbatch_size")
    if config["workers"] < 0 or config["precision"] not in ("bf16", "fp32"):
        raise ValueError("Invalid workers or precision")
    if (
        not 0 <= config["ema_decay"] < 1
        or not 0 <= config["occlusion_probability"] <= 1
    ):
        raise ValueError("Invalid EMA decay or occlusion probability")
    for key in ("backbone_lr", "direction_lr", "gradient_clip", "hard_weight"):
        if not 0 < config[key] < float("inf"):
            raise ValueError(f"{key} must be finite and positive")
    for key in ("weight_decay", "direction_weight", "geometry_weight"):
        if not 0 <= config[key] < float("inf"):
            raise ValueError(f"{key} must be finite and nonnegative")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train.json")
    parser.add_argument("--plans", type=Path, default=ROOT / "configs/plans.json")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--init-checkpoint", type=Path)
    mode.add_argument("--resume", type=Path)
    mode.add_argument(
        "--from-scratch",
        action="store_true",
        help="For implementation tests; no pretrained weights",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--steps",
        type=int,
        help="Total updates in this run, including updates already completed",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        help="Save and exit after this many additional updates",
    )
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if args.steps is not None:
        config["steps"] = args.steps
    check_config(config)
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices: cpu or cuda")
    if (
        device.type == "cuda"
        and config["precision"] == "bf16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("BF16 is unavailable; select fp32 in the config")
    if args.stop_after is not None and args.stop_after <= 0:
        raise ValueError("stop-after must be positive")
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    plans = json.loads(args.plans.read_text())
    fingerprint = input_fingerprint(args.manifest)
    code_digest = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        code_digest.update(source.name.encode())
        code_digest.update(source.read_bytes())
    code_sha256 = code_digest.hexdigest()
    model = FiberDetector(plans).to(device)
    base_step = initialize(model, args.init_checkpoint) if args.init_checkpoint else 0
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = optimizer_for(model, config)
    completed, next_sample = 0, 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        # Extending the requested duration and changing loader concurrency do not
        # alter the objective, update order or deterministic sample identities.
        mutable = {
            "steps",
            "workers",
            "cache_windows",
            "checkpoint_every",
            "validate_every",
            "validation_samples",
        }
        for key in config.keys() | state["config"].keys():
            if key not in mutable and config.get(key) != state["config"].get(key):
                raise ValueError(f"Resume changes training parameter: {key}")
        if state["input_fingerprint"] != fingerprint or state["plans"] != plans:
            raise ValueError("Resume input manifest or architecture changed")
        if state["code_sha256"] != code_sha256:
            raise ValueError("Training source changed since the checkpoint")
        model.load_state_dict(state["model"], strict=True)
        ema.load_state_dict(state["ema"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        completed, next_sample, base_step = (
            state["completed"],
            state["next_sample"],
            state["base_step"],
        )
        if next_sample != completed * config["batch_size"]:
            raise ValueError("Invalid saved sample cursor")
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        del state
    elif args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use an empty output directory for a new run")
    if config["steps"] <= completed:
        raise ValueError("Requested steps must exceed completed updates")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    dataset = FiberDataset(
        args.manifest,
        "train",
        (config["steps"] - completed) * config["batch_size"],
        start=next_sample,
        seed=config["seed"],
        cache_windows=config["cache_windows"],
        occlusion_probability=config["occlusion_probability"],
    )
    validation = FiberDataset(
        args.manifest,
        "validation",
        config["validation_samples"],
        seed=config["seed"],
        cache_windows=1,
    )
    # Separate generator prevents DataLoader startup from consuming model RNG.
    loader = DataLoader(
        dataset,
        batch_size=config["microbatch_size"],
        num_workers=config["workers"],
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(config["seed"]),
        **({"prefetch_factor": 1} if config["workers"] else {}),
    )
    val_loader = DataLoader(
        validation,
        batch_size=1,
        num_workers=0,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    iterator = iter(loader)
    accumulation = config["batch_size"] // config["microbatch_size"]
    stop_at = (
        min(config["steps"], completed + args.stop_after)
        if args.stop_after
        else config["steps"]
    )

    def checkpoint():
        save_checkpoint(
            args.output / "last.pt",
            dict(
                model=model.state_dict(),
                ema=ema.state_dict(),
                optimizer=optimizer.state_dict(),
                completed=completed,
                next_sample=next_sample,
                base_step=base_step,
                config=config,
                plans=plans,
                input_fingerprint=fingerprint,
                code_sha256=code_sha256,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(device)
                if device.type == "cuda"
                else None,
            ),
        )
        for name, network in (("raw", model), ("ema", ema)):
            save_checkpoint(
                args.output / f"{name}.pth",
                dict(
                    network_weights=network.backbone.state_dict(),
                    auxiliary_weights={
                        k: v
                        for k, v in network.state_dict().items()
                        if not k.startswith("backbone.")
                    },
                    step=base_step + completed,
                ),
            )

    history = args.output / "metrics.jsonl"
    # Remove records beyond a restored checkpoint, for crash recovery in place.
    if args.resume and history.exists():
        records = [
            line
            for line in history.read_text().splitlines()
            if json.loads(line)["update"] <= completed
        ]
        history.write_text("".join(line + "\n" for line in records))
    model.train()
    while completed < stop_at:
        optimizer.zero_grad(set_to_none=True)
        totals = {}
        hard_count = occluded_count = 0
        for _ in range(accumulation):
            batch = {
                k: v.to(device, non_blocking=True) for k, v in next(iterator).items()
            }
            with precision_context(device, config):
                output = model(batch["ct"])
            loss, parts = total_loss(output, batch, config)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Nonfinite loss; last completed checkpoint is unchanged"
                )
            (loss / accumulation).backward()
            for key, value in {"loss": loss.detach(), **parts}.items():
                totals[key] = totals.get(key, 0.0) + float(value) / accumulation
            hard_count += int(batch["hard"].sum())
            occluded_count += int(batch["occluded"].sum())
            del batch, output, loss, parts
        gradient = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["gradient_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        with torch.no_grad():
            for target, source in zip(ema.parameters(), model.parameters()):
                target.lerp_(source, 1 - config["ema_decay"])
        completed += 1
        next_sample += config["batch_size"]
        record = dict(
            update=completed,
            step=base_step + completed,
            **totals,
            gradient_norm=float(gradient),
            hard_examples=hard_count,
            occluded_examples=occluded_count,
        )
        if completed % config["validate_every"] == 0:
            record["validation_raw"] = validate(model, val_loader, device, config)
            record["validation_ema"] = validate(ema, val_loader, device, config)
            model.train()
        with history.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
        if completed % config["checkpoint_every"] == 0 or completed == stop_at:
            checkpoint()


if __name__ == "__main__":
    main()

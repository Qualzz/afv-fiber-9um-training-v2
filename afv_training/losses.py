"""Masked semantic, centre, affinity and axial losses from the final recipe."""

import torch
import torch.nn.functional as F


def affinity_targets(instance, semantic):
    targets = []
    valids = []
    # Channel order: dz1,dy1,dx1,dz4,dy4,dx4. No wraparound edges.
    for step in (1, 4):
        for axis in (1, 2, 3):
            a = [slice(None)] * 4
            b = a.copy()
            a[axis] = slice(None, -step)
            b[axis] = slice(step, None)
            a = tuple(a)
            b = tuple(b)
            left, right = instance[a], instance[b]
            valid = (
                (semantic[a] != 255)
                & (semantic[b] != 255)
                & (
                    ((left == right) & (left > 0))
                    | ((left > 0) & (semantic[b] == 0))
                    | ((right > 0) & (semantic[a] == 0))
                )
            )
            target = (left == right) & (left > 0)
            output = torch.zeros_like(instance, dtype=torch.float32)
            mask = output.clone()
            output[a] = target
            mask[a] = valid
            targets.append(output)
            valids.append(mask)
    return torch.stack(targets, 1), torch.stack(valids, 1)


def cube_weights(weights, reference):
    w = torch.as_tensor(weights, device=reference.device, dtype=torch.float32)
    if w.ndim != 1 or len(w) != len(reference):
        raise ValueError("One weight required per cube")
    if not bool(torch.isfinite(w).all() and (w > 0).all()):
        raise ValueError("Invalid cube weights")
    return (w / w.mean()).reshape(-1, 1, 1, 1)


def mean(error, mask, w):
    weighted = mask.float() * w
    return (error * weighted).sum() / weighted.sum().clamp_min(1)


def semantic(logits, target, union, w):
    logits = logits.float()
    valid = target != 255
    ce = F.cross_entropy(logits, target.long(), ignore_index=255, reduction="none")
    fg = valid & (target > 0)
    bg = valid & (target == 0)
    union = union & ~valid
    presence = torch.logsumexp(logits, 1) - torch.logsumexp(logits[:, 1:], 1)
    positive = ((ce * fg + presence * union) * w).sum()
    ce = 0.5 * (positive / ((fg | union) * w).sum().clamp_min(1) + mean(ce, bg, w))
    p = logits.softmax(1)
    dice = []
    for cls in (1, 2):
        truth = (target == cls).float()
        pred = p[:, cls] * valid
        dice.append(
            1
            - (2 * (pred * truth * w).sum() + 1)
            / ((pred * w).sum() + (truth * w).sum() + 1)
        )
    return ce + 0.5 * sum(dice)


def loss_for(output, batch, weights):
    sem = batch["semantic"]
    core = batch["centre"] > 0
    w = cube_weights(weights, sem)
    union = (
        batch["presence"] > 0
        if "presence" in batch
        else core | ((sem > 0) & (sem != 255))
    )
    seg = semantic(output["seg"], sem, union, w)
    deep = seg * 0
    for weight, logits in zip((0.25, 0.125), output["deep"]):
        factor = sem.shape[-1] // logits.shape[-1]
        target = F.interpolate(
            sem[:, None].float(), size=logits.shape[2:], mode="nearest"
        )[:, 0].long()
        unknown = (
            F.max_pool3d((sem == 255)[:, None].float(), factor, stride=factor)[:, 0] > 0
        )
        target[unknown] = 255
        deep = deep + weight * semantic(
            logits, target, torch.zeros_like(target, dtype=torch.bool), w
        )
    p = output["seg"].float().softmax(1)
    fg = p[:, 1] + p[:, 2] + p[:, 3]
    recall = mean(1 - fg, core, w)
    targets, valid = affinity_targets(batch["instances"], sem)
    error = F.binary_cross_entropy_with_logits(
        output["affinity"].float(), targets.float(), reduction="none"
    )
    affinity = 0.5 * (
        mean(error, valid * targets, w[:, None])
        + mean(error, valid * (1 - targets), w[:, None])
    )
    distance = batch["distance"]
    known = batch["valid"] > 0
    error = F.binary_cross_entropy_with_logits(
        output["centre"][:, 0].float(), batch["target"].float(), reduction="none"
    )
    zones = (
        (distance <= 1) & known,
        (distance > 1) & (distance <= 6) & known,
        (distance > 6) & known,
    )
    centre = sum(a * mean(error, m, w) for a, m in zip((0.25, 0.65, 0.10), zones))
    base = seg + deep + 0.25 * recall + 0.3 * affinity + 2 * centre
    pred = F.normalize(output["direction"].float(), dim=1, eps=1e-6)
    truth = F.normalize(batch["direction"].float(), dim=1, eps=1e-6)
    direction = mean(
        1 - (pred * truth).sum(1).square().clamp(0, 1), batch["direction_valid"], w
    )
    return (
        base,
        direction,
        dict(
            semantic=seg.detach(),
            deep=deep.detach(),
            centre=centre.detach(),
            skeleton_recall_loss=recall.detach(),
            affinity=affinity.detach(),
        ),
    )

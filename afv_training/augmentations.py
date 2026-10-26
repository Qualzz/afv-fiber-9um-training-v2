"""128-cubed CT sampling, aligned compression and short local occultations."""

import hashlib
import numpy as np
from scipy.ndimage import gaussian_filter
from .compression import Compaction, sheet_normal
from .interpolation import map_coordinates
from .targets import source_fields, lookup, output_core, attach_directions

OFFSETS_A = (-1 / 3, 0.0, 1 / 3)


def eligible_anchors(region, wanted):
    # A 512-voxel context must fit inside the 640-voxel resident window.
    # Pick an anchor that can actually lie in its central 128-voxel prediction.
    p = region["points"]
    shape = np.asarray(region["ct"].shape)
    reachable = ((p >= 232) & (p <= shape - 232)).all(1)
    eligible = np.flatnonzero(reachable & (region["families"] == wanted))
    if not len(eligible):
        eligible = np.flatnonzero(reachable)
    if not len(eligible):
        raise ValueError("No reachable spline anchor in " + region["name"])
    return eligible


def _select(region, index, augment=True, force_factors=None):
    """Verbatim replay of production sample() up to the warp; same RNG draws."""
    rng = np.random.default_rng(62016000 + index)
    ct = region["ct"]
    shape = np.array(ct.shape)
    wanted = 1 + index % 2
    eligible = eligible_anchors(region, wanted)
    anchor_index = int(eligible[rng.integers(len(eligible))])
    center = np.clip(
        region["points"][anchor_index] + rng.integers(-24, 25, 3), 256, shape - 256
    )
    start = np.rint(center - 255.5).astype(int)
    start = np.clip(start, 0, shape - 512)
    block = ct[tuple(slice(int(a), int(a) + 512) for a in start)]
    material = void = 1.0
    kind = "native"
    if force_factors is not None:
        material, void = force_factors
        kind = "forced"
    elif augment and rng.random() < 0.5:
        r = rng.random()
        if r < 0.2:
            material = 1.0
            void = rng.uniform(0.3, 0.6)
            kind = "mild"
        elif r < 0.6:
            material = rng.uniform(0.85, 1.0)
            void = rng.uniform(0.08, 0.25)
            kind = "moderate"
        else:
            material = rng.uniform(0.55, 0.8)
            void = rng.uniform(0.02, 0.1)
            kind = "hard"
    normal, confidence = sheet_normal(block, [255.5] * 3)
    warp = Compaction(block, [255.5] * 3, normal, material, void)
    return rng, dict(
        wanted=int(wanted),
        anchor_index=anchor_index,
        center=center,
        start=start,
        kind=kind,
        material=material,
        void=void,
        normal=normal,
        confidence=confidence,
        warp=warp,
    )


def _retained(region, s):
    near = np.asarray(
        sorted(region["tree"].query_ball_point(s["center"], 445)), dtype=np.int64
    )
    transformed, kept = s["warp"].output_points(
        region["points"][near] - s["start"], return_indices=True
    )
    return transformed + 63.5, near[kept]


def sample(region, index, augment=True, force_factors=None):
    rng, s = _select(region, index, augment, force_factors)
    ct = region["ct"]
    shape = np.asarray(ct.shape)
    start = s["start"]
    warp = s["warp"]
    normal = s["normal"]
    offsets = (0.0,) if warp.native else OFFSETS_A
    fields = source_fields(region)
    grid = np.indices((128,) * 3, dtype="f4").reshape(3, -1) - 63.5
    x = np.zeros((128,) * 3, "f4")
    inside = np.ones(x.shape, bool)
    profile = np.ones(x.shape, bool)
    owner = None
    for off in offsets:
        c = (
            warp.source_coordinates(grid + normal[:, None] * off).reshape(
                3, 128, 128, 128
            )
            + start[:, None, None, None]
        )
        relative = c - start[:, None, None, None]
        bounded = ((relative >= 0) & (relative <= 511)).all(0) & (
            (c >= 0) & (c <= (shape - 1)[:, None, None, None])
        ).all(0)
        inside &= bounded
        values = map_coordinates(
            ct, c, order=1, mode="constant", cval=0, prefilter=False, output="f4"
        )
        values[~bounded] = 0
        x += values / len(offsets)
        d2, code, identity = lookup(fields, c)
        if off == 0.0:
            distance = np.sqrt(d2.astype("f4"))
        if owner is None:
            owner = identity
        profile &= bounded & (d2 <= 4) & ((code & 12) == 0) & (identity == owner)
    transformed, selected = _retained(region, s)
    geo = output_core(
        transformed, selected, region["instances"], region["families"], region["points"]
    )
    core = geo["core"] & inside
    if core.sum() < 16 or inside.mean() < 0.8:
        if not warp.native:
            row = sample(region, index, augment=augment, force_factors=(1.0, 1.0))
            row["meta"]["compression_rejected"] = s["kind"]
            return row
        raise ValueError("Insufficient observed centres")
    semantic = np.full(x.shape, 255, "u1")
    semantic[core] = geo["family"][core]
    instances = np.zeros(x.shape, "i4")
    instances[core] = geo["instance"][core]
    instances[semantic == 255] = 0
    distance[core] = 0
    valid = (profile & inside) | core
    target = np.exp(-0.5 * (distance / 1.5) ** 2).astype("f4")
    target[~valid] = 0
    target[core] = 1
    row = dict(
        semantic=semantic,
        instances=instances,
        centre=core.astype("u1"),
        distance=distance,
        valid=valid.astype("u1"),
        target=target,
        presence=core.astype("u1"),
        source_inside=inside.astype("u1"),
        instance_ambiguous=(core & (instances == 0)).astype("u1"),
    )
    directions = attach_directions(row, transformed, selected, region)
    x = (x - region["mean"]) / max(region["std"], 10.0)
    if augment:
        x = x * rng.uniform(0.85, 1.15) + rng.uniform(-0.15, 0.15)
        x += rng.normal(0, rng.uniform(0, 0.05), x.shape).astype("f4")
        if rng.random() < 0.5:
            x = gaussian_filter(x, rng.uniform(0.1, 0.6))
        for axis in range(3):
            if rng.random() < 0.5:
                x = np.flip(x, axis)
                for key, value in row.items():
                    row[key] = np.flip(value, axis + (key == "direction")).copy()
                row["direction"][2 - axis] *= -1
    row["ct"] = np.ascontiguousarray(x[None], dtype="f4")
    row["meta"] = dict(
        region=region["name"],
        index=int(index),
        kind=s["kind"],
        material=float(s["material"]),
        void=float(s["void"]),
        normal_confidence=float(s["confidence"]),
        compression_rejected=None,
        center_voxels=int(core.sum()),
        direction_voxels=directions,
        target_version="splines-only-local-centre-2vx-v1",
        teacher_used=False,
        semantic_background_voxels=0,
        centre_profile_source_radius_vx=2.0,
        valid_source_fraction=float(inside.mean()),
        retained_indices_sha256=hashlib.sha256(
            np.ascontiguousarray(selected, np.int64).tobytes()
        ).hexdigest(),
    )
    assert not np.any(row["semantic"] == 0)
    assert np.all(row["semantic"][row["presence"] == 0] == 255)
    assert not np.any(row["valid"][row["distance"] > 2])
    return row


def occlude(row, probability=0.25, force=False):
    """Hide a short interior CT segment, never remove or bridge target curves."""
    rng = np.random.default_rng([int(row["meta"]["index"]), 211025])
    meta = dict(row["meta"])
    result = dict(row)
    result["meta"] = meta
    info = dict(applied=False, probability=probability, targets_unchanged=True)
    meta["occlusion"] = info
    if not force and rng.random() >= probability:
        return result
    candidates = np.argwhere(
        (row["centre"] > 0) & (row["direction_valid"] > 0) & (row["instances"] > 0)
    )
    candidates = candidates[((candidates >= 16) & (candidates < 112)).all(1)]
    if not len(candidates):
        return result
    for candidate in rng.permutation(len(candidates))[:64]:
        centre = candidates[candidate]
        index = tuple(centre)
        tangent = row["direction"][:, *index][::-1].astype("f8")
        tangent /= max(np.linalg.norm(tangent), 1e-8)
        owner = int(row["instances"][index])
        half = float(rng.uniform(2.0, 5.0))
        # Both visible ends must belong to THIS annotated instance.
        valid = True
        for sign in (-1, 1):
            end = np.rint(centre + sign * (half + 4) * tangent).astype(int)
            box = tuple(slice(v - 2, v + 3) for v in end)
            if not np.any((row["instances"][box] == owner) & (row["centre"][box] > 0)):
                valid = False
        if not valid:
            continue
        box = tuple(slice(int(v - 12), int(v + 13)) for v in centre)
        delta = np.indices((25,) * 3, dtype="f4").transpose(1, 2, 3, 0) - 12
        longitudinal = delta @ tangent
        perpendicular = np.sum(delta * delta, axis=-1) - longitudinal**2
        radius = np.sqrt((longitudinal / half) ** 2 + np.maximum(perpendicular, 0) / 9)
        alpha = np.clip((1.25 - radius) / 0.25, 0, 1).astype("f4")
        identities = row["instances"][box][alpha > 0.1]
        if np.any((identities > 0) & (identities != owner)):
            continue
        original = row["ct"][0][box]
        # Photometric corruption only: no semantic label is inferred from its fill.
        fill = rng.normal(
            float(np.median(original)),
            max(float(original.std()) * 0.25, 0.02),
            original.shape,
        ).astype("f4")
        image = row["ct"].copy()
        image[0][box] = original * (1 - alpha) + fill * alpha
        result["ct"] = image
        info.update(
            applied=True,
            centre_zyx=centre.tolist(),
            instance=owner,
            half_length_vx=half,
            radius_vx=3.0,
            changed_voxels=int((alpha > 0).sum()),
            hidden_observed_centres=int(
                ((row["centre"][box] > 0) & (alpha > 0.5)).sum()
            ),
        )
        return result
    info["reason"] = "No unambiguous interior segment with two visible annotated ends"
    return result

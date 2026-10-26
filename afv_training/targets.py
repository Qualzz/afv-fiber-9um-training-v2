"""Observed spline cells, ownership masks and axial direction targets."""

import time
import numpy as np
from numba import njit
from scipy.ndimage import distance_transform_edt, binary_dilation
from scipy.spatial import cKDTree

HALO = 8
SLAB = 48


def _boundary(labels):
    out = np.zeros(labels.shape, bool)
    for ax in range(labels.ndim):
        a = [slice(None)] * labels.ndim
        b = list(a)
        a[ax] = slice(None, -1)
        b[ax] = slice(1, None)
        a = tuple(a)
        b = tuple(b)
        diff = labels[a] != labels[b]
        out[a] |= diff
        out[b] |= diff
    return out


def seed_records(shape, points, instances, families):
    """One record per seed voxel floor(p+.5); the closest point wins, collisions are flagged."""
    p = np.asarray(points, "f8").reshape(-1, 3)
    v = np.floor(p + 0.5).astype(np.int64)
    ok = ((v >= 0) & (v < np.asarray(shape))).all(1)
    p = p[ok]
    v = v[ok]
    ids = np.asarray(instances)[ok].astype(np.int64)
    fam = np.asarray(families)[ok].astype("u1")
    flat = np.ravel_multi_index(tuple(v.T), tuple(shape))
    order = np.lexsort((((p - v) ** 2).sum(1), flat))
    flat = flat[order]
    ids = ids[order]
    fam = fam[order]
    first = np.r_[True, flat[1:] != flat[:-1]] if len(flat) else np.zeros(0, bool)
    group = np.cumsum(first) - 1
    count = int(first.sum())
    chosen_id = ids[first]
    chosen_family = fam[first]
    return dict(
        flat=flat[first],
        instance=chosen_id.astype(np.int32),
        family=chosen_family,
        id_collision=np.bincount(
            group, weights=(ids != chosen_id[group]).astype("f8"), minlength=count
        )
        > 0,
        family_collision=np.bincount(
            group, weights=(fam != chosen_family[group]).astype("f8"), minlength=count
        )
        > 0,
    )


def build_source_fields(
    shape, points, instances, families, slab=SLAB, halo=HALO, workers=1
):
    """Source-lattice label fields built in z slabs with a z halo.

    d2: squared EDT distance to the nearest seed voxel, exact where d2<=halo**2,
    otherwise >halo**2 (clipped to 255). code bits 0-1 nearest family, bit 2
    family ambiguity, bit 3 instance ambiguity (A's collision-cell and
    ownership-boundary rule, dilated by one voxel). instance: nearest seed ID.
    """
    assert halo >= 8, "d<=6 thresholds plus boundary dilation need halo>=8"
    start = time.monotonic()
    shape = tuple(int(v) for v in shape)
    Z, Y, X = shape
    plane = Y * X
    seeds = seed_records(shape, points, instances, families)
    assert int(seeds["family"].max(initial=0)) <= 3
    itype = np.uint16 if int(seeds["instance"].max(initial=0)) < 65535 else np.int32
    d2 = np.full(shape, 255, "u1")
    code = np.zeros(shape, "u1")
    inst = np.zeros(shape, itype)
    seed_z = seeds["flat"] // plane
    yy = np.arange(Y, dtype=np.int32)[None, :, None]
    xx = np.arange(X, dtype=np.int32)[None, None, :]

    def build_slab(z0):
        z1 = min(Z, z0 + slab)
        a = max(0, z0 - halo)
        b = min(Z, z1 + halo)
        lo, hi = np.searchsorted(seed_z, (a, b))
        if lo == hi:
            return
        record = np.full((b - a) * plane, -1, np.int32)
        record[seeds["flat"][lo:hi] - a * plane] = np.arange(lo, hi, dtype=np.int32)
        record = record.reshape(b - a, Y, X)
        nearest = np.empty((3, b - a, Y, X), np.int32)
        distance_transform_edt(
            record < 0, return_distances=False, return_indices=True, indices=nearest
        )
        owner = record[nearest[0], nearest[1], nearest[2]]
        core = slice(z0 - a, z1 - a)
        dz = nearest[0, core] - np.arange(z0 - a, z1 - a, dtype=np.int32)[:, None, None]
        dy = nearest[1, core] - yy
        dx = nearest[2, core] - xx
        d2[z0:z1] = np.minimum(dz * dz + dy * dy + dx * dx, 255)
        del record, nearest, dz, dy, dx
        fam = seeds["family"][owner]
        ids = seeds["instance"][owner]
        family_amb = binary_dilation(seeds["family_collision"][owner] | _boundary(fam))
        id_amb = (
            binary_dilation(seeds["id_collision"][owner] | _boundary(ids)) | family_amb
        )
        code[z0:z1] = (
            fam[core]
            | (family_amb[core].astype("u1") << 2)
            | (id_amb[core].astype("u1") << 3)
        )
        inst[z0:z1] = ids[core]
        del owner, fam, ids, family_amb, id_amb

    for z0 in range(0, Z, slab):
        build_slab(z0)
    return dict(
        shape=shape,
        d2=d2,
        code=code,
        instance=inst,
        seeds=int(len(seeds["flat"])),
        halo=halo,
        slab=slab,
        seconds=time.monotonic() - start,
        bytes=int(d2.nbytes + code.nbytes + inst.nbytes),
    )


def lookup(fields, c):
    flat = np.ravel_multi_index(
        tuple(np.floor(c + 0.5).astype(np.int64)), fields["shape"], mode="clip"
    )
    return (
        fields["d2"].reshape(-1)[flat],
        fields["code"].reshape(-1)[flat],
        fields["instance"].reshape(-1)[flat].astype("i4"),
    )


def _per_voxel(flat, value, base, conflict):
    key = np.unique(flat * base + value)
    voxel = key // base
    val = key % base
    first = np.r_[True, voxel[1:] != voxel[:-1]]
    count = np.diff(np.r_[np.flatnonzero(first), len(key)])
    return voxel[first], np.where(count == 1, val[first], conflict)


@njit(cache=True)
def segment_cell_midpoints(a, b):
    capacity = 0
    for i in range(len(a)):
        for k in range(3):
            capacity += int(np.ceil(abs(b[i, k] - a[i, k])))
        capacity += 4
    points = np.empty((capacity, 3), np.float64)
    which = np.empty(capacity, np.int64)
    used = 0
    for i in range(len(a)):
        delta = b[i] - a[i]
        count = 2
        maxbreak = 2
        for k in range(3):
            maxbreak += int(np.ceil(abs(delta[k]))) + 1
        breaks = np.empty(maxbreak, np.float64)
        breaks[0] = 0.0
        breaks[1] = 1.0
        for k in range(3):
            if abs(delta[k]) < 1e-12:
                continue
            low = min(a[i, k], b[i, k])
            high = max(a[i, k], b[i, k])
            for edge in range(int(np.floor(low - 0.5)) + 1, int(np.ceil(high - 0.5))):
                t = (edge + 0.5 - a[i, k]) / delta[k]
                if 0.0 < t < 1.0:
                    breaks[count] = t
                    count += 1
        ordered = np.sort(breaks[:count])
        for j in range(count - 1):
            if ordered[j + 1] - ordered[j] < 1e-14:
                continue
            t = (ordered[j] + ordered[j + 1]) * 0.5
            points[used] = a[i] + t * delta
            which[used] = i
            used += 1
    return points[:used], which[:used]


def output_core(
    transformed,
    selected,
    instances,
    families,
    source_points,
    size=128,
    halo=HALO,
    substeps=16,
    distance=True,
):
    """Union presence core in output space.

    Every retained point's voxel floor(p+.5) is core. Valid spline segments
    (same ID, consecutive source index, source step <=0.8 as spline_tangents)
    add every voxel traversed over a positive-length segment interval (exact grid-plane crossings). Family conflicts give
    255, instance conflicts 0; presence stays positive in both cases.
    """
    p = np.asarray(transformed, "f8").reshape(-1, 3)
    sel = np.asarray(selected, np.int64)
    instances = np.asarray(instances)
    families = np.asarray(families)
    G = size + 2 * halo
    near = ((p >= -halo - 1) & (p <= size + halo)).all(1)
    pieces = [p[near]]
    owners = [sel[near]]
    if len(p) > 1:
        seg = (
            near[:-1]
            & (np.diff(sel) == 1)
            & (instances[sel[:-1]] == instances[sel[1:]])
        )
        k = np.flatnonzero(seg)
        k = k[
            np.linalg.norm(source_points[sel[k + 1]] - source_points[sel[k]], axis=1)
            <= 0.8
        ]
        a = p[k]
        b = p[k + 1]
        crossed, segment_index = segment_cell_midpoints(a, b)
        pieces.append(crossed)
        owners.append(sel[k][segment_index])
    q = np.concatenate(pieces)
    owner = np.concatenate(owners)
    v = np.floor(q + 0.5).astype(np.int64) + halo
    ok = ((v >= 0) & (v < G)).all(1)
    v = v[ok]
    owner = owner[ok]
    core = np.zeros(G**3, bool)
    family = np.zeros(G**3, "u1")
    instance = np.zeros(G**3, "i4")
    if len(v):
        flat = np.ravel_multi_index(tuple(v.T), (G,) * 3)
        ids = instances[owner].astype(np.int64)
        voxels, vf = _per_voxel(flat, families[owner].astype(np.int64), 4, 255)
        _, vi = _per_voxel(flat, ids, int(ids.max()) + 1, 0)
        core[voxels] = True
        family[voxels] = vf
        instance[voxels] = vi
    core = core.reshape((G,) * 3)
    crop = (slice(halo, halo + size),) * 3
    out = dict(
        core=core[crop].copy(),
        family=family.reshape((G,) * 3)[crop].copy(),
        instance=instance.reshape((G,) * 3)[crop].copy(),
    )
    if distance:
        full = distance_transform_edt(~core) if core.any() else np.full((G,) * 3, 1e3)
        out["distance"] = full[crop].astype("f4")
    return out


def spline_tangents(points_zyx, instance, source_indices, source_points=None):
    """Axial XYZ tangents from the transformed spline, never prediction voxels.

    Central differences require both adjacent samples on the same original
    spline. Cropping, re-entry gaps and separate instances cannot be bridged.
    """
    p = np.asarray(points_zyx, dtype="f4")
    ids = np.asarray(instance)
    indices = np.asarray(source_indices)
    v = np.zeros_like(p)
    valid = np.zeros(len(p), bool)
    if len(p) > 2:
        good = (
            (ids[:-2] == ids[1:-1])
            & (ids[1:-1] == ids[2:])
            & (np.diff(indices)[:-1] == 1)
            & (np.diff(indices)[1:] == 1)
        )

        if source_points is not None:
            adjacent = np.linalg.norm(np.diff(source_points, axis=0), axis=1) <= 0.8
            good &= adjacent[:-1] & adjacent[1:]
        delta = p[2:] - p[:-2]
        norm = np.linalg.norm(delta, axis=1)
        good &= norm > 1e-6
        v[1:-1] = delta / np.maximum(norm[:, None], 1e-6)
        valid[1:-1] = good
    v[~valid] = 0
    return v[:, ::-1].copy(), valid


def attach_directions(target, transformed_zyx, selected, region):
    selected = np.asarray(selected, np.int64)
    ids = region["instances"][selected]
    tangents, valid = spline_tangents(
        transformed_zyx, ids, selected, region["points"][selected]
    )
    shape = target["semantic"].shape
    direction = np.zeros((3, *shape), np.float32)
    mask = np.zeros(shape, np.uint8)
    # A single known owner is required. Crossings and unobserved joins are ignored.
    eligible = (
        (target["semantic"] > 0)
        & (target["semantic"] < 255)
        & (target["instances"] > 0)
    )
    voxels = np.argwhere(eligible)
    if len(voxels) and len(transformed_zyx):
        distance, nearest = cKDTree(transformed_zyx).query(voxels, workers=1)
        ix = tuple(voxels.T)
        good = (
            valid[nearest]
            & (ids[nearest] == target["instances"][ix])
            & (distance <= 3.0)
        )
        q = tuple(voxels[good].T)
        direction[(slice(None),) + q] = tangents[nearest[good]].T
        mask[q] = 1
    target["direction"] = direction
    target["direction_valid"] = mask
    return int(mask.sum())


def source_fields(region):
    if "source_fields" not in region:
        region["source_fields"] = build_source_fields(
            region["ct"].shape,
            region["points"],
            region["instances"],
            region["families"],
        )
    return region["source_fields"]

"""Monotone, void-first compression shared by CT and spline coordinates."""

import numpy as np
from scipy.ndimage import gaussian_filter, grey_closing, map_coordinates
from .interpolation import map_coordinates as fast_map_coordinates


def sheet_normal(ct, center):
    ci = np.clip(np.rint(center).astype(int), 12, 499)
    patch = np.asarray(
        ct[tuple(slice(int(c - 12), int(c + 13)) for c in ci)], dtype="f4"
    )
    g = np.array(np.gradient(gaussian_filter(patch, 1))).reshape(3, -1)
    values, vectors = np.linalg.eigh(g @ g.T / max(1, g.shape[1]))
    normal = vectors[:, -1]
    return normal.astype("f4"), float(
        (values[-1] - values[-2]) / max(float(values[-1]), 1e-9)
    )


def frame_for(normal):
    n = np.asarray(normal, dtype="f4")
    n /= np.linalg.norm(n)
    t = np.eye(3, dtype="f4")[int(np.argmin(np.abs(n)))]
    t -= n * np.dot(n, t)
    t /= np.linalg.norm(t)
    return np.stack((n, t, np.cross(n, t)), axis=1).astype("f4")


class Compaction:
    def __init__(self, ct, center, normal, material_factor, void_factor):
        assert 0.55 <= material_factor <= 1 and 0.02 <= void_factor <= 1
        self.center = np.asarray(center, dtype="f4")
        self.frame = frame_for(normal)
        self.native = material_factor == 1 and void_factor == 1
        self.material_factor = material_factor
        self.void_factor = void_factor
        if self.native:
            return
        u = np.arange(-256, 257, dtype="f4")
        v = np.arange(-128, 129, 4, dtype="f4")
        q = np.stack(np.meshgrid(u, v, v, indexing="ij")).reshape(3, -1)
        xyz = (self.center[:, None] + self.frame @ q).reshape(3, len(u), len(v), len(v))
        raw = map_coordinates(
            ct, xyz, order=1, mode="constant", cval=0, prefilter=False, output="f4"
        )
        # Close tiny dark texture holes before estimating material support.
        # Density guides displacement only; CT intensities are sampled afterwards.
        smooth = gaussian_filter(raw, sigma=(0.8, 0.6, 0.6))
        support = grey_closing(smooth, size=(3, 1, 1))
        density = gaussian_filter(
            np.clip((support - 30) / 40, 0, 1), sigma=(0.8, 1.0, 1.0)
        )
        weights = void_factor + (material_factor - void_factor) * density
        cumulative = np.cumsum(weights, axis=0) - 0.5 * weights
        cumulative -= cumulative[len(u) // 2][None]
        output_u = np.arange(-128, 129, dtype="f4")
        inverse = np.empty((len(output_u), len(v), len(v)), "f4")
        for iy in range(len(v)):
            for ix in range(len(v)):
                positions = cumulative[:, iy, ix]
                values = np.interp(output_u, positions, u)
                low = output_u < positions[0]
                high = output_u > positions[-1]
                values[low] = u[0] + (output_u[low] - positions[0]) / weights[0, iy, ix]
                values[high] = (
                    u[-1] + (output_u[high] - positions[-1]) / weights[-1, iy, ix]
                )
                inverse[:, iy, ix] = values
        assert np.all(np.diff(inverse, axis=0) > 0), "Non-invertible compaction"
        self.inverse = inverse
        self.mean_material_density = float(density.mean())

    def input_u(self, q):
        if self.native:
            return q[0]
        index = np.array([q[0] + 128, (q[1] + 128) / 4, (q[2] + 128) / 4], dtype="f4")
        return fast_map_coordinates(
            self.inverse, index, order=1, mode="nearest", prefilter=False
        )

    def source_coordinates(self, output_points):
        q = self.frame.T @ np.asarray(output_points, dtype="f4")
        q[0] = self.input_u(q)
        return self.center[:, None] + self.frame @ q

    def output_points(self, source_points, return_indices=False):
        q = self.frame.T @ (
            np.asarray(source_points, dtype="f4").T - self.center[:, None]
        )
        selected = np.arange(q.shape[1])
        if self.native:
            result = (self.frame @ q).T
            return (result, selected) if return_indices else result
        lateral = (np.abs(q[1:]) <= 128).all(0)
        q = q[:, lateral]
        selected = selected[lateral]
        low = np.full(q.shape[1], -128.0, "f4")
        high = -low
        inside = (q[0] >= self.input_u(np.stack((low, q[1], q[2])))) & (
            q[0] <= self.input_u(np.stack((high, q[1], q[2])))
        )
        q = q[:, inside]
        low = low[inside]
        high = high[inside]
        selected = selected[inside]
        original_u = q[0].copy()
        # <0.032 output voxel error; invert precisely the same field used for CT.
        for _ in range(13):
            middle = (low + high) * 0.5
            values = self.input_u(np.stack((middle, q[1], q[2])))
            lower = values < original_u
            low = np.where(lower, middle, low)
            high = np.where(lower, high, middle)
        q[0] = (low + high) * 0.5
        result = (self.frame @ q).T
        return (result, selected) if return_indices else result

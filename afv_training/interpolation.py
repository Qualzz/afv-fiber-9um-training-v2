"""Trilinear interpolation with the training kernel's accumulation order."""

import numpy as np
from numba import njit
from scipy.ndimage import map_coordinates as scipy_map_coordinates


@njit(cache=True, nogil=True)
def _linear(volumes, c, constant, cval):
    channels = len(volumes)
    zsize, ysize, xsize = volumes[0].shape
    out = np.empty((channels, c.shape[1]), np.float32)
    for i in range(c.shape[1]):
        z = np.float64(c[0, i])
        y = np.float64(c[1, i])
        x = np.float64(c[2, i])
        if constant and (
            z < 0 or y < 0 or x < 0 or z > zsize - 1 or y > ysize - 1 or x > xsize - 1
        ):
            for ch in range(channels):
                out[ch, i] = cval
            continue
        iz = int(np.floor(z))
        iy = int(np.floor(y))
        ix = int(np.floor(x))
        wz0 = 1.0 - (z - iz)
        wy0 = 1.0 - (y - iy)
        wx0 = 1.0 - (x - ix)
        wz = (wz0, 1.0 - wz0)
        wy = (wy0, 1.0 - wy0)
        wx = (wx0, 1.0 - wx0)
        for ch in range(channels):
            value = 0.0
            for dz in range(2):
                for dy in range(2):
                    for dx in range(2):
                        coef = np.float64(
                            volumes[ch][
                                min(max(iz + dz, 0), zsize - 1),
                                min(max(iy + dy, 0), ysize - 1),
                                min(max(ix + dx, 0), xsize - 1),
                            ]
                        )
                        coef *= wz[dz]
                        coef *= wy[dy]
                        coef *= wx[dx]
                        value += coef
            out[ch, i] = value
    return out


def map_coordinates(
    input, coordinates, output=None, order=3, mode="constant", cval=0.0, prefilter=True
):
    a = np.asarray(input)
    c = np.asarray(coordinates)
    dtype = (
        np.dtype(output)
        if output is not None and not isinstance(output, np.ndarray)
        else a.dtype
    )
    if (
        a.ndim == 3
        and order == 1
        and mode in ("nearest", "constant")
        and dtype == np.dtype("f4")
        and c.dtype in (np.dtype("f4"), np.dtype("f8"))
        and not isinstance(output, np.ndarray)
    ):
        return _linear((a,), c.reshape(3, -1), mode == "constant", float(cval))[
            0
        ].reshape(c.shape[1:])
    return scipy_map_coordinates(
        input,
        coordinates,
        output=output,
        order=order,
        mode=mode,
        cval=cval,
        prefilter=prefilter,
    )

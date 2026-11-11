"""Resampling of arrays between the grids of OME-Zarr pyramid levels."""

from typing import Literal, Optional, Sequence, Tuple

import dask.array as da
import numpy as np
from scipy import ndimage

Method = Literal["nearest", "mean", "linear"]


def _fit(array: da.Array, target: Sequence[Optional[int]]) -> da.Array:
    """Crop or edge-pad `array` to `target` sizes (``None`` leaves an axis alone).

    Level sizes are rounded when a pyramid is written, so integer-factor
    resampling can be a pixel or two off the stored level shape.
    """
    out = array
    out = out[
        tuple(slice(0, min(n, t)) if t is not None else slice(None) for n, t in zip(out.shape, target))
    ]
    pad = [(0, t - n) if t is not None and t > n else (0, 0) for n, t in zip(out.shape, target)]
    if any(p[1] for p in pad):
        out = da.pad(out, pad, mode="edge")
    return out


def _downscale(array: da.Array, factors: Sequence[int], method: str) -> da.Array:
    if all(f == 1 for f in factors):
        return array
    if method == "nearest":
        index = tuple(slice(0, (n // f) * f, f) for n, f in zip(array.shape, factors))
        return array[index]
    averaged = da.coarsen(
        np.mean,
        array,
        {axis: f for axis, f in enumerate(factors) if f != 1},
        trim_excess=True,
    )
    if np.issubdtype(array.dtype, np.integer):
        averaged = da.rint(averaged)
    return averaged.astype(array.dtype)


def _upscale(array: da.Array, factors: Sequence[int], method: str) -> da.Array:
    axes = [axis for axis, f in enumerate(factors) if f != 1]
    if not axes:
        return array
    if method == "nearest":
        out = array
        for axis in axes:
            out = da.repeat(out, factors[axis], axis=axis)
        return out

    # Linear: interpolate each block with a one-pixel halo (so block edges agree
    # with the whole-array result), then trim the halo.
    depth = {axis: 1 for axis in axes}
    zoom = tuple(float(f) for f in factors)
    halo = da.overlap.overlap(array, depth=depth, boundary="nearest")
    chunks = tuple(
        tuple(c * f for c in cs) for cs, f in zip(halo.chunks, factors)
    )
    zoomed = halo.map_blocks(
        lambda block: ndimage.zoom(
            block, zoom, order=1, mode="nearest", grid_mode=True
        ).astype(block.dtype, copy=False),
        chunks=chunks,
        dtype=array.dtype,
    )
    return da.overlap.trim_internal(
        zoomed, {axis: factors[axis] for axis in axes}, boundary="nearest"
    )


def resample_array(
    array: da.Array,
    factors: Sequence[float],
    target_shape: Sequence[Optional[int]],
    method: Method,
) -> da.Array:
    """Resample `array` by per-axis integer `factors` (>1 downscales, <1 upscales).

    `target_shape` gives the exact output size per axis (``None`` = whatever the
    factor yields); the result is cropped or edge-padded to it.
    """
    down, up = [], []
    for f in factors:
        if f >= 1:
            ratio = round(f)
            ok = abs(f - ratio) <= 1e-6
            down.append(ratio)
            up.append(1)
        else:
            ratio = round(1 / f)
            ok = abs(1 / f - ratio) <= 1e-6
            down.append(1)
            up.append(ratio)
        if not ok:
            raise ValueError(
                f"Scale ratio {f} between the levels is not an integer (or its "
                "inverse); cannot resample by an integer factor."
            )
    out = array
    if any(f != 1 for f in down):
        out = _downscale(out, down, "mean" if method == "linear" else method)
    if any(f != 1 for f in up):
        out = _upscale(out, up, "linear" if method == "mean" else method)
    return _fit(out, target_shape)

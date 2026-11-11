"""Tests for Reader.resample."""

import dask.array as da
import numpy as np
import pytest
from scipy import ndimage

from ome_zarr_io import Reader, Writer
from ome_zarr_io.schema_models import Channel, Omero


@pytest.fixture
def reader(tmp_path):
    # 101x75 gives odd sizes at the coarser levels (50x37, 25x18).
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(2, 101, 75), dtype=np.uint16)
    path = tmp_path / "r.zarr"
    Writer(
        path=path,
        image=image,
        dims=["c", "y", "x"],
        axis_units={"y": "micrometer", "x": "micrometer"},
        scale_transformations={"y": 0.5, "x": 0.5},
        downscale_levels=2,
        downscale_factor=2,
        omero_metadata=Omero(channels=[Channel(label="DAPI"), Channel(label="GFP")]),
        overwrite=True,
    ).write()
    return Reader(path)


def test_downscale_matches_stored_level_shape(reader):
    ch = reader.get_channel("DAPI", 0)
    for to in (1, 2):
        out = reader.resample(ch, 0, to)
        assert out.shape == reader.get_channel("DAPI", to).shape
        assert out.dtype == ch.dtype


def test_mean_downscale_matches_stored_level(reader):
    out = reader.resample(reader.get_channel("DAPI", 0, "numpy"), 0, 1)
    np.testing.assert_array_equal(out, reader.get_channel("DAPI", 1, "numpy"))


def test_upscale_matches_level0_shape_for_odd_sizes(reader):
    for src in (1, 2):
        out = reader.resample(reader.get_channel("GFP", src), src, 0)
        assert out.shape == reader.get_channel("GFP", 0).shape


def test_labels_roundtrip_and_no_new_ids(reader):
    lbl = np.random.default_rng(1).integers(0, 5, size=(101, 75)).astype(np.uint8)
    down = reader.resample(lbl, 0, 2, kind="label")
    assert set(np.unique(down)) <= set(np.unique(lbl))
    up = reader.resample(down, 2, 0, kind="label")
    assert up.shape == lbl.shape
    assert set(np.unique(up)) <= set(np.unique(lbl))
    # nearest up then down returns the coarse array
    np.testing.assert_array_equal(reader.resample(up, 0, 2, kind="label"), down)


def test_linear_matches_scipy_zoom_and_dask_equals_numpy(reader):
    # Level 1 -> 0 is a factor of 2; compare against scipy on the overlap.
    a = np.random.default_rng(3).random((50, 37)).astype(np.float32)
    got_np = reader.resample(a, 1, 0)
    got_da = reader.resample(da.from_array(a, chunks=(16, 16)), 1, 0).compute()
    np.testing.assert_allclose(got_da, got_np, rtol=1e-5)
    ref = ndimage.zoom(a, 2, order=1, mode="nearest", grid_mode=True)
    np.testing.assert_allclose(got_np[:100, :74], ref[:100, :74], rtol=1e-5)


def test_dask_stays_lazy_and_as_type(reader):
    lazy = reader.resample(reader.get_channel("DAPI", 0), 0, 1)
    assert isinstance(lazy, da.Array)
    assert isinstance(reader.resample(lazy, 1, 0, as_type="numpy"), np.ndarray)
    arr = reader.get_channel("DAPI", 0, "numpy")
    assert isinstance(reader.resample(arr, 0, 1), np.ndarray)
    assert isinstance(reader.resample(arr, 0, 1, as_type="dask"), da.Array)


def test_full_array_with_channel_axis_and_negative_level(reader):
    full = reader.get_channel("DAPI", 0, "numpy")[None].repeat(2, axis=0)
    out = reader.resample(full, 0, -1)
    assert out.shape == (2, 25, 18)


def test_nearest_upscale_for_images(reader):
    a = np.arange(6, dtype=np.uint8).reshape(2, 3)
    a = np.pad(a, ((0, 48), (0, 34)))  # shape (50, 37)
    out = reader.resample(a, 1, 0, method="nearest")
    np.testing.assert_array_equal(out[:4:2, :6:2], a[:2, :3])


def test_errors(reader):
    a = reader.get_channel("DAPI", 0, "numpy")
    with pytest.raises(ValueError, match="invent label"):
        reader.resample(a, 0, 1, kind="label", method="mean")
    with pytest.raises(ValueError, match="does not apply"):
        reader.resample(a, 0, 1, method="linear")
    with pytest.raises(ValueError, match="does not apply"):
        reader.resample(reader.get_channel("DAPI", 1, "numpy"), 1, 0, method="mean")
    with pytest.raises(ValueError, match="from_level"):
        reader.resample(a, 1, 0)
    with pytest.raises(IndexError):
        reader.resample(a, 0, 7)
    with pytest.raises(TypeError):
        reader.resample(a, 0, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dimensions"):
        reader.resample(np.zeros(5), 0, 1)

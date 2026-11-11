"""Tests for Writer.from_existing(path).add_channels."""

import dask.array as da
import numpy as np
import pytest
import zarr

from ome_zarr_io import Reader, Writer, validate

UNITS = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "img.ome.zarr"
    data = np.random.default_rng(0).integers(0, 4000, (2, 3, 64, 64)).astype(np.uint16)
    Writer(
        path,
        data,
        list("czyx"),
        UNITS,
        scale_transformations={"z": 2.0, "y": 0.5, "x": 0.5},
        downscale_levels=2,
        channels=["DAPI", "GFP"],
    ).write()
    return path, data


def new_data(shape=(1, 3, 64, 64), dtype=np.uint16):
    return np.random.default_rng(1).integers(0, 3000, shape).astype(dtype)


def test_adds_channel_to_every_level(image):
    path, data = image
    new = new_data()
    Writer.from_existing(path).add_channels(
        new, [{"label": "DAPI_bgsub", "active": False}]
    )
    for level, size in enumerate((64, 32, 16)):
        arr = zarr.open_array(str(path / str(level)))
        assert arr.shape == (3, 3, size, size)
        assert arr.dtype == np.uint16
    np.testing.assert_array_equal(zarr.open_array(str(path / "0"))[:2], data)
    np.testing.assert_array_equal(zarr.open_array(str(path / "0"))[2:], new)

    channels = zarr.open_group(str(path)).attrs["ome"]["omero"]["channels"]
    assert [c["label"] for c in channels] == ["DAPI", "GFP", "DAPI_bgsub"]
    assert channels[2]["active"] is False
    assert channels[2]["color"] not in {channels[0]["color"], channels[1]["color"]}
    assert set(channels[2]["window"]) == {"start", "min", "end", "max"}
    assert Reader(path).channel_names == ["DAPI", "GFP", "DAPI_bgsub"]
    validate(path)


def test_pyramid_matches_a_fresh_write(image, tmp_path):
    path, data = image
    new = new_data((2, 3, 64, 64))
    Writer.from_existing(path).add_channels(new, ["A", "B"])
    ref = tmp_path / "ref.ome.zarr"
    Writer(
        ref,
        np.concatenate([data, new]),
        list("czyx"),
        UNITS,
        scale_transformations={"z": 2.0, "y": 0.5, "x": 0.5},
        downscale_levels=2,
    ).write()
    for level in range(3):
        np.testing.assert_array_equal(
            zarr.open_array(str(path / str(level)))[:],
            zarr.open_array(str(ref / str(level)))[:],
        )


def test_accepts_dask_and_dict_channels(image):
    path, _ = image
    Writer.from_existing(path).add_channels(
        da.from_array(new_data()), {"X": {"color": "ff0000", "window": (0, 100)}}
    )
    ch = zarr.open_group(str(path)).attrs["ome"]["omero"]["channels"][-1]
    assert ch["color"] == "FF0000" or ch["color"].lower() == "ff0000"
    assert ch["window"]["start"] == 0 and ch["window"]["end"] == 100


def test_writer_stays_usable_for_labels(image):
    path, _ = image
    writer = Writer.from_existing(path)
    writer.add_channels(new_data(), ["bg"])
    assert writer.image.shape[0] == 3
    writer.add_labels("mask", np.ones((3, 3, 64, 64), dtype=np.uint8))
    writer.add_channels(new_data(), ["bg2"])
    assert zarr.open_array(str(path / "0")).shape[0] == 4


@pytest.mark.parametrize(
    "array, channels, match",
    [
        (new_data(dtype=np.float32), ["x"], "dtype"),
        (new_data((1, 3, 32, 64)), ["x"], "does not match"),
        (new_data((1, 2, 64, 64)), ["x"], "does not match"),
        (new_data(), ["DAPI"], "already in the image"),
        (new_data(), ["a", "b"], "2 channels given"),
        (new_data(), ["a", "a"], "duplicate"),
    ],
)
def test_rejects_bad_input_and_leaves_image_untouched(image, array, channels, match):
    path, data = image
    with pytest.raises(ValueError, match=match):
        Writer.from_existing(path).add_channels(array, channels)
    assert zarr.open_array(str(path / "0")).shape == (2, 3, 64, 64)
    assert len(zarr.open_group(str(path)).attrs["ome"]["omero"]["channels"]) == 2


def test_rejects_non_array(image):
    with pytest.raises(TypeError):
        Writer.from_existing(image[0]).add_channels([[1]], ["x"])


def test_requires_from_existing(image):
    w = Writer(image[0], np.zeros((1, 1, 8, 8), np.uint8), list("czyx"), UNITS)
    with pytest.raises(RuntimeError, match="from_existing"):
        w.add_channels(np.zeros((1, 1, 8, 8), np.uint8), ["x"])


def test_requires_c_axis(tmp_path):
    path = tmp_path / "zyx.ome.zarr"
    Writer(path, np.zeros((2, 16, 16), np.uint8), list("zyx"), UNITS).write()
    with pytest.raises(ValueError, match="no 'c' axis"):
        Writer.from_existing(path).add_channels(np.zeros((2, 16, 16), np.uint8), ["x"])


def test_requires_omero_for_existing_channels(tmp_path):
    path = tmp_path / "plain.ome.zarr"
    Writer(path, np.zeros((2, 16, 16), np.uint8), list("cyx"), UNITS).write()
    with pytest.raises(ValueError, match="omero"):
        Writer.from_existing(path).add_channels(np.zeros((1, 16, 16), np.uint8), ["x"])


def test_failure_mid_write_rolls_back(image):
    path, _ = image

    def boom(block):
        raise RuntimeError("compute failed")

    bad = da.from_array(new_data()).map_blocks(boom, dtype=np.uint16)
    with pytest.raises(RuntimeError, match="compute failed"):
        Writer.from_existing(path).add_channels(bad, ["x"])
    for level in range(3):
        assert zarr.open_array(str(path / str(level))).shape[0] == 2
    assert len(zarr.open_group(str(path)).attrs["ome"]["omero"]["channels"]) == 2

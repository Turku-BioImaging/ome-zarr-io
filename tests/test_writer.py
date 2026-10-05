"""Test the Writer class initialization and basic functionality."""

import pytest
import numpy as np
import dask.array as da
from pathlib import Path
import copy
import zarr
from ome_zarr_io.writer import Writer
from ome_zarr_io.reader import Reader
from ome_zarr_io.schema_models import ScaleTransformation


@pytest.fixture
def sample_2d_image():
    """Create a sample 2D image for testing."""
    return np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)


@pytest.fixture
def sample_3d_image():
    """Create a sample 3D image for testing."""
    return np.random.randint(0, 255, size=(10, 100, 100), dtype=np.uint8)


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return tmp_path


def test_init_with_numpy_array(temp_dir, sample_2d_image):
    """Test initialization with numpy array."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}

    writer = Writer(
        path=path, image=sample_2d_image, dims=dims, axis_units=axis_units
    )

    assert writer.path == Path(path)
    assert isinstance(writer.image, da.Array)  # Should be converted to dask
    assert writer.dims == dims
    assert len(writer.axes) == 2


def test_init_with_dask_array(temp_dir, sample_2d_image):
    """Test initialization with dask array."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}
    dask_image = da.from_array(sample_2d_image, chunks="auto")

    writer = Writer(path=path, image=dask_image, dims=dims, axis_units=axis_units)

    assert writer.path == Path(path)
    assert isinstance(writer.image, da.Array)
    assert writer.dims == dims


def test_init_with_coordinate_transformations(temp_dir, sample_2d_image):
    """Test initialization with coordinate transformations."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}
    # Test with simple scale transformations
    scale_transformations = {"y": 0.1, "x": 0.1}

    writer = Writer(
        path=path,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units,
        scale_transformations=scale_transformations,
    )

    # Should create a single ScaleTransformation object
    assert writer.coordinate_transformations is not None
    assert len(writer.coordinate_transformations) == 1
    assert isinstance(writer.coordinate_transformations[0], ScaleTransformation)
    assert writer.coordinate_transformations[0].scale == [0.1, 0.1]


def test_init_with_overwrite_flag(temp_dir, sample_2d_image):
    """Test initialization with overwrite flag."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}

    writer = Writer(
        path=path,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units,
        overwrite=True,
    )

    assert writer.overwrite is True


def test_path_handling_string_input(sample_2d_image):
    """Test that string paths are converted to Path objects."""
    path_str = "/tmp/test.zarr"
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}

    writer = Writer(
        path=path_str, image=sample_2d_image, dims=dims, axis_units=axis_units
    )

    assert isinstance(writer.path, Path)
    assert str(writer.path) == path_str


def test_add_labels_to_existing_image(temp_dir, sample_2d_image):
    """Test that a label image can be attached to an existing image group."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}

    writer = Writer(
        path=path,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units,
        downscale_levels=1,
    )
    writer.write(chunks=(32, 32))

    label = np.zeros_like(sample_2d_image, dtype=np.uint8)
    label[10:20, 10:20] = 1

    writer.add_labels(
        name="cell_space_segmentation",
        array=label,
        colors=[
            {"label-value": 0, "rgba": [0, 0, 128, 128]},
            {"label-value": 1, "rgba": [0, 128, 0, 128]},
        ],
        properties=[
            {"label-value": 0, "class": "intercellular space"},
            {"label-value": 1, "class": "cell"},
        ],
    )

    root = zarr.open_group(str(path), mode="r")
    assert "labels" in root
    labels_group = root["labels"]
    assert labels_group.attrs["ome"]["version"] == "0.5"
    assert labels_group.attrs["ome"]["labels"] == ["cell_space_segmentation"]
    label_group = labels_group["cell_space_segmentation"]
    assert label_group.attrs["ome"]["image-label"]["version"] == "0.5"
    assert label_group.attrs["ome"]["image-label"]["colors"][0]["label-value"] == 0
    assert "multiscales" in label_group.attrs["ome"]


def test_write_method_works(temp_dir, sample_2d_image):
    """Test that write method works (no longer raises NotImplementedError)."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}

    writer = Writer(
        path=path, image=sample_2d_image, dims=dims, axis_units=axis_units
    )

    # Should not raise an exception
    writer.write()

    # Verify the file was created
    assert path.exists()


def test_3d_image_initialization(temp_dir, sample_3d_image):
    """Test initialization with 3D image."""
    path = temp_dir / "test.zarr"
    dims = ["z", "y", "x"]
    axis_units = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}

    writer = Writer(
        path=path, image=sample_3d_image, dims=dims, axis_units=axis_units
    )

    assert writer.image.shape == sample_3d_image.shape
    assert writer.dims == dims
    assert len(writer.axes) == 3


def test_multichannel_image_dims(temp_dir):
    """Test initialization with multichannel image."""
    multichannel_image = np.random.randint(0, 255, size=(3, 100, 100), dtype=np.uint8)
    path = temp_dir / "test.zarr"
    dims = ["c", "y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}

    writer = Writer(
        path=path, image=multichannel_image, dims=dims, axis_units=axis_units
    )

    assert writer.image.shape == multichannel_image.shape
    assert writer.dims == dims
    assert len(writer.axes) == 3
    # Check that channel axis has no unit
    channel_axis = next(ax for ax in writer.axes if ax.name == "c")
    assert channel_axis.unit is None


def test_write_without_zarr_backend(temp_dir, sample_2d_image):
    """Test that write works without zarr_backend parameter."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}

    writer = Writer(
        path=path, image=sample_2d_image, dims=dims, axis_units=axis_units
    )

    baseline = copy.deepcopy(zarr.config.get("codec_pipeline"))

    writer.write()

    # Ensure global config restored after write
    assert zarr.config.get("codec_pipeline") == baseline


def test_write_preserves_zarr_config(temp_dir, sample_2d_image):
    """Test that write operation preserves zarr global configuration."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}

    writer = Writer(
        path=path,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units,
    )

    baseline = copy.deepcopy(zarr.config.get("codec_pipeline"))
    writer.write()

    assert zarr.config.get("codec_pipeline") == baseline


def test_write_leaves_zarr_config_intact(temp_dir, sample_2d_image):
    """Test that write operation leaves zarr configuration intact."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}

    writer = Writer(
        path=path,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units,
    )

    baseline = copy.deepcopy(zarr.config.get("codec_pipeline"))

    writer.write()

    assert zarr.config.get("codec_pipeline") == baseline


def test_multiple_writes_preserve_config(temp_dir, sample_2d_image):
    """Ensure multiple write operations preserve zarr configuration."""
    path1 = temp_dir / "test1.zarr"
    path2 = temp_dir / "test2.zarr"
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}

    baseline = copy.deepcopy(zarr.config.get("codec_pipeline"))

    writer1 = Writer(
        path=path1,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units,
    )
    writer1.write()
    assert zarr.config.get("codec_pipeline") == baseline

    writer2 = Writer(
        path=path2,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units,
    )
    writer2.write()
    assert zarr.config.get("codec_pipeline") == baseline


def test_add_labels_singleton_channel_axis(temp_dir):
    """A label with c=1 is valid for a multi-channel image (NGFF 0.5)."""
    path = temp_dir / "test.zarr"
    writer = Writer(
        path=path,
        image=np.zeros((2, 3, 4, 32, 32), dtype=np.uint16),
        dims=["t", "c", "z", "y", "x"],
        axis_units={"t": "second", "z": "micrometer", "y": "micrometer", "x": "micrometer"},
        downscale_levels=2,
    )
    writer.write()
    writer.add_labels(name="ch0", array=np.zeros((2, 1, 4, 32, 32), dtype=np.uint8))

    label_group = zarr.open_group(str(path), mode="r")["labels"]["ch0"]
    datasets = label_group.attrs["ome"]["multiscales"][0]["datasets"]
    assert [label_group[d["path"]].shape for d in datasets] == [
        (2, 1, 4, 32, 32),
        (2, 1, 4, 16, 16),
        (2, 1, 4, 8, 8),
    ]
    assert Reader(path).validate()


def test_add_labels_rejects_incompatible_shapes(temp_dir):
    path = temp_dir / "test.zarr"
    writer = Writer(
        path=path,
        image=np.zeros((3, 4, 32, 32), dtype=np.uint16),
        dims=["c", "z", "y", "x"],
        axis_units={"z": "micrometer", "y": "micrometer", "x": "micrometer"},
    )
    writer.write()
    with pytest.raises(ValueError, match="axis 'c' must be 3 or 1"):
        writer.add_labels(name="a", array=np.zeros((2, 4, 32, 32), dtype=np.uint8))
    with pytest.raises(ValueError, match="axis 'x' must be 32"):
        writer.add_labels(name="b", array=np.zeros((3, 4, 32, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="dimensions"):
        writer.add_labels(name="c", array=np.zeros((4, 32, 32), dtype=np.uint8))


class TestDimensionNames:
    """NGFF 0.5: "dimension_names" MUST be in each array's zarr.json and match "axes"."""

    def test_image_and_label_arrays(self, tmp_path):
        import json

        path = tmp_path / "dims.ome.zarr"
        dims = ["c", "z", "y", "x"]
        writer = Writer(
            path=path,
            image=np.zeros((2, 3, 32, 32), dtype=np.uint8),
            dims=dims,
            axis_units={"z": "micrometer", "y": "micrometer", "x": "micrometer"},
            downscale_levels=1,
        )
        writer.write()
        writer.add_labels("mask", np.ones((1, 3, 32, 32), dtype=np.uint8))

        for array in ("0", "1", "labels/mask/0", "labels/mask/1"):
            meta = json.loads((path / array / "zarr.json").read_text())
            assert meta["dimension_names"] == dims, array


class TestLabelLevelCount:
    """NGFF 0.5: a label image MUST have the same number of scale levels as its image."""

    UNITS = {"y": "micrometer", "x": "micrometer"}

    @staticmethod
    def levels(group_path):
        import json

        ome = json.loads((group_path / "zarr.json").read_text())["attributes"]["ome"]
        return len(ome["multiscales"][0]["datasets"])

    def image(self, path, size=64, **kwargs):
        writer = Writer(
            path, np.zeros((size, size), dtype=np.uint8), ["y", "x"], self.UNITS, **kwargs
        )
        writer.write()
        return writer

    def mask(self, size=64):
        return np.ones((size, size), dtype=np.uint8)

    @pytest.mark.parametrize("downscale_levels", [None, 0, 1, 3])
    def test_label_has_as_many_levels_as_the_image(self, tmp_path, downscale_levels):
        path = tmp_path / "img.zarr"
        writer = self.image(path, downscale_levels=downscale_levels)
        writer.add_labels("mask", self.mask())
        assert self.levels(path / "labels" / "mask") == self.levels(path)
        assert self.levels(path) == (downscale_levels or 0) + 1

    def test_matches_an_image_whose_levels_were_capped(self, tmp_path):
        path = tmp_path / "img.zarr"
        with pytest.warns(UserWarning, match="too many"):
            writer = self.image(path, size=16, downscale_levels=10)
        writer.add_labels("mask", self.mask(16))  # must not warn again
        assert self.levels(path) == 5  # 16, 8, 4, 2, 1
        assert self.levels(path / "labels" / "mask") == 5

    def test_count_comes_from_the_image_on_disk_not_the_writer(self, tmp_path):
        """A second Writer with other settings must still write a matching label."""
        path = tmp_path / "img.zarr"
        self.image(path, downscale_levels=3)
        for name, kwargs in (("default", {}), ("other", {"downscale_levels": 1})):
            other = Writer(
                path, np.zeros((64, 64), dtype=np.uint8), ["y", "x"], self.UNITS, **kwargs
            )
            other.add_labels(name, self.mask())
            assert self.levels(path / "labels" / name) == 4

    def test_downscale_levels_argument_is_deprecated(self, tmp_path):
        path = tmp_path / "img.zarr"
        writer = self.image(path, downscale_levels=2)
        with pytest.warns(DeprecationWarning, match="downscale_levels is deprecated"):
            writer.add_labels("mask", self.mask(), downscale_levels=2)
        assert self.levels(path / "labels" / "mask") == 3

    @pytest.mark.parametrize("bad", [0, 1, 3, -1])
    def test_mismatched_downscale_levels_is_rejected(self, tmp_path, bad):
        path = tmp_path / "img.zarr"
        writer = self.image(path, downscale_levels=2)
        with pytest.raises(ValueError, match="same number of levels as its image"):
            writer.add_labels("mask", self.mask(), downscale_levels=bad)
        assert not (path / "labels").exists()  # nothing written by the rejected call

    def test_negative_downscale_levels_matches_a_single_level_image(self, tmp_path):
        path = tmp_path / "img.zarr"
        writer = self.image(path)
        with pytest.warns(DeprecationWarning):
            writer.add_labels("mask", self.mask(), downscale_levels=-1)
        assert self.levels(path / "labels" / "mask") == 1

    def test_downscale_factor_that_keeps_the_count_is_allowed(self, tmp_path):
        path = tmp_path / "img.zarr"
        writer = self.image(path, downscale_levels=2)
        writer.add_labels("mask", self.mask(), downscale_factor=4)
        assert self.levels(path / "labels" / "mask") == 3
        shapes = [zarr.open_array(str(path / "labels/mask" / str(i))).shape for i in range(3)]
        assert shapes == [(64, 64), (16, 16), (4, 4)]

    def test_downscale_factor_that_cannot_reach_the_count_is_rejected(self, tmp_path):
        path = tmp_path / "img.zarr"
        writer = self.image(path, downscale_levels=3)
        with pytest.raises(ValueError, match="can only have 3"):
            writer.add_labels("mask", self.mask(), downscale_factor=8)
        assert not (path / "labels").exists()

    def test_rejected_overwrite_keeps_the_existing_label(self, tmp_path):
        path = tmp_path / "img.zarr"
        writer = self.image(path, downscale_levels=3)
        writer.add_labels("mask", self.mask())
        with pytest.raises(ValueError):
            writer.add_labels("mask", self.mask(), downscale_factor=8, overwrite=True)
        assert self.levels(path / "labels" / "mask") == 4
        assert Reader(path).label_names == ["mask"]

    def test_group_without_image_metadata_is_rejected(self, tmp_path):
        path = tmp_path / "plain.zarr"
        zarr.create_group(str(path), zarr_format=3)
        writer = Writer(
            path, np.zeros((64, 64), dtype=np.uint8), ["y", "x"], self.UNITS
        )
        with pytest.raises(ValueError, match="no multiscales metadata"):
            writer.add_labels("mask", self.mask())

    def test_written_fileset_is_valid(self, tmp_path):
        path = tmp_path / "img.zarr"
        writer = self.image(path, downscale_levels=2)
        writer.add_labels("mask", self.mask())
        assert Reader(path).validate()


class TestLabelDtype:
    """NGFF 0.5: label pixels MUST be one of the eight integer data types."""

    UNITS = {"y": "micrometer", "x": "micrometer"}

    @pytest.fixture
    def writer(self, tmp_path):
        writer = Writer(
            tmp_path / "img.zarr",
            np.zeros((32, 32), dtype=np.uint8),
            ["y", "x"],
            self.UNITS,
            downscale_levels=1,
        )
        writer.write()
        return writer

    @pytest.mark.parametrize(
        "dtype",
        ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64"],
    )
    def test_integer_dtypes_are_accepted(self, writer, dtype):
        writer.add_labels("mask", np.ones((32, 32), dtype=dtype))
        assert str(zarr.open_array(str(writer.path / "labels/mask/0")).dtype) == dtype
        assert Reader(writer.path).validate()

    @pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "complex64"])
    def test_float_and_complex_are_rejected(self, writer, dtype):
        with pytest.raises(ValueError, match=f"integer data type .* got {dtype}"):
            writer.add_labels("mask", np.ones((32, 32), dtype=dtype))
        assert not (writer.path / "labels").exists()  # nothing written

    def test_whole_number_floats_are_still_rejected(self, writer):
        with pytest.raises(ValueError, match="integer data type"):
            writer.add_labels("mask", np.array([[0.0, 1.0]] * 32 * 16).reshape(32, 32))

    def test_bool_is_rejected_with_a_conversion_hint(self, writer):
        mask = np.ones((32, 32), dtype=bool)
        with pytest.raises(ValueError, match=r'got bool\. Convert .*astype\("uint8"\)'):
            writer.add_labels("mask", mask)
        assert not (writer.path / "labels").exists()
        writer.add_labels("mask", mask.astype("uint8"))  # the suggested fix works
        assert Reader(writer.path).label_names == ["mask"]

    def test_dask_arrays_are_checked_too(self, writer):
        with pytest.raises(ValueError, match="got float32"):
            writer.add_labels("mask", da.ones((32, 32), dtype="float32", chunks=16))
        writer.add_labels("mask", da.ones((32, 32), dtype="uint16", chunks=16))

    def test_rejected_overwrite_keeps_the_existing_label(self, writer):
        writer.add_labels("mask", np.ones((32, 32), dtype=np.uint8))
        with pytest.raises(ValueError):
            writer.add_labels("mask", np.ones((32, 32), dtype=bool), overwrite=True)
        assert str(zarr.open_array(str(writer.path / "labels/mask/0")).dtype) == "uint8"

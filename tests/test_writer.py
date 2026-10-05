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

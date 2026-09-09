"""Test the OmeZarrImage class initialization and basic functionality."""

import pytest
import numpy as np
import dask.array as da
from pathlib import Path
import copy
import zarr
from ome_zarr_writer.image import OmeZarrImage
from ome_zarr_writer.schema_models import ScaleTransformation


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

    writer = OmeZarrImage(
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

    writer = OmeZarrImage(path=path, image=dask_image, dims=dims, axis_units=axis_units)

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

    writer = OmeZarrImage(
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

    writer = OmeZarrImage(
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

    writer = OmeZarrImage(
        path=path_str, image=sample_2d_image, dims=dims, axis_units=axis_units
    )

    assert isinstance(writer.path, Path)
    assert str(writer.path) == path_str


def test_add_labels_to_existing_image(temp_dir, sample_2d_image):
    """Test that a label image can be attached to an existing image group."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}

    writer = OmeZarrImage(
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

    writer = OmeZarrImage(
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

    writer = OmeZarrImage(
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

    writer = OmeZarrImage(
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

    writer = OmeZarrImage(
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

    writer = OmeZarrImage(
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

    writer = OmeZarrImage(
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

    writer1 = OmeZarrImage(
        path=path1,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units,
    )
    writer1.write()
    assert zarr.config.get("codec_pipeline") == baseline

    writer2 = OmeZarrImage(
        path=path2,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units,
    )
    writer2.write()
    assert zarr.config.get("codec_pipeline") == baseline

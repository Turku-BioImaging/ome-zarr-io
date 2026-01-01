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


def test_zarr_backend_default(temp_dir, sample_2d_image):
    """Default zarr_backend is 'zarrs' but should not mutate global config."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}

    writer = OmeZarrImage(
        path=path, image=sample_2d_image, dims=dims, axis_units=axis_units
    )

    baseline = copy.deepcopy(zarr.config.get("codec_pipeline"))
    assert writer.zarr_backend == "zarrs"

    writer.write()

    # Ensure global config restored after write
    assert zarr.config.get("codec_pipeline") == baseline


def test_zarr_backend_zarrs_explicit(temp_dir, sample_2d_image):
    """Explicit zarrs backend uses scoped config and leaves globals untouched."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}

    writer = OmeZarrImage(
        path=path,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units,
        zarr_backend="zarrs",
    )

    assert writer.zarr_backend == "zarrs"

    baseline = copy.deepcopy(zarr.config.get("codec_pipeline"))
    writer.write()

    assert zarr.config.get("codec_pipeline") == baseline


def test_zarr_backend_zarr_python(temp_dir, sample_2d_image):
    """Zarr-python backend should leave existing config as-is."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}

    writer = OmeZarrImage(
        path=path,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units,
        zarr_backend="zarr-python",
    )

    assert writer.zarr_backend == "zarr-python"
    baseline = copy.deepcopy(zarr.config.get("codec_pipeline"))

    writer.write()

    assert zarr.config.get("codec_pipeline") == baseline


def test_zarr_config_isolation(temp_dir, sample_2d_image):
    """Ensure backend-specific config is scoped per write and does not leak."""
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
        zarr_backend="zarrs",
    )
    writer1.write()
    assert zarr.config.get("codec_pipeline") == baseline

    writer2 = OmeZarrImage(
        path=path2,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units,
        zarr_backend="zarr-python",
    )
    writer2.write()
    assert zarr.config.get("codec_pipeline") == baseline

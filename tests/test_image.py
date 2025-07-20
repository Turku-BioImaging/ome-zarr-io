"""
Tests for the OmeZarrImage writer class.
"""

import pytest
import numpy as np
import dask.array as da
from pathlib import Path
import tempfile
from ome_zarr_writer.image import OmeZarrImage


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    import shutil
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_2d_image():
    """Create a sample 2D image for testing."""
    return np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)


@pytest.fixture
def sample_3d_image():
    """Create a sample 3D image for testing."""
    return np.random.randint(0, 255, size=(10, 100, 100), dtype=np.uint8)


def test_init_with_numpy_array(temp_dir, sample_2d_image):
    """Test initialization with numpy array."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    
    writer = OmeZarrImage(
        path=path,
        image=sample_2d_image,
        dims=dims
    )
    
    assert writer.path == Path(path)
    assert isinstance(writer.image, da.Array)
    assert writer.dims == dims
    assert writer.coordinate_transformations is None
    assert writer.downscale_levels is None
    assert writer.overwrite is False


def test_init_with_dask_array(temp_dir, sample_2d_image):
    """Test initialization with dask array."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    dask_image = da.from_array(sample_2d_image, chunks="auto")
    
    writer = OmeZarrImage(
        path=path,
        image=dask_image,
        dims=dims
    )
    
    assert writer.path == Path(path)
    assert isinstance(writer.image, da.Array)
    assert writer.image is dask_image  # Should be the same object
    assert writer.dims == dims


def test_init_with_coordinate_transformations(temp_dir, sample_2d_image):
    """Test initialization with coordinate transformations."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    coord_transforms = [{"type": "scale", "scale": [0.1, 0.1]}]
    
    writer = OmeZarrImage(
        path=path,
        image=sample_2d_image,
        dims=dims,
        coordinate_transformations=coord_transforms
    )
    
    assert writer.coordinate_transformations == coord_transforms


def test_init_with_downscale_levels(temp_dir, sample_2d_image):
    """Test initialization with downscale levels."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    downscale_levels = 3
    
    writer = OmeZarrImage(
        path=path,
        image=sample_2d_image,
        dims=dims,
        downscale_levels=downscale_levels
    )
    
    assert writer.downscale_levels == downscale_levels


def test_init_with_overwrite_flag(temp_dir, sample_2d_image):
    """Test initialization with overwrite flag."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    
    writer = OmeZarrImage(
        path=path,
        image=sample_2d_image,
        dims=dims,
        overwrite=True
    )
    
    assert writer.overwrite is True


def test_path_handling_string_input(sample_2d_image):
    """Test that string paths are converted to Path objects."""
    path_str = "/tmp/test.zarr"
    dims = ["y", "x"]
    
    writer = OmeZarrImage(
        path=path_str,
        image=sample_2d_image,
        dims=dims
    )
    
    assert isinstance(writer.path, Path)
    assert str(writer.path) == path_str


def test_create_multiscale_group_not_implemented(temp_dir, sample_2d_image):
    """Test that create_multiscale_group raises NotImplementedError."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    
    writer = OmeZarrImage(
        path=path,
        image=sample_2d_image,
        dims=dims
    )
    
    with pytest.raises(NotImplementedError):
        writer.create_multiscale_group(
            arrays=[sample_2d_image],
            axes=[{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
        )


def test_create_downscaled_arrays_basic_functionality(temp_dir, sample_2d_image):
    """Test that _create_downscaled_arrays creates downscaled versions correctly."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    
    writer = OmeZarrImage(
        path=path,
        image=sample_2d_image,
        dims=dims,
        downscale_levels=2
    )
    
    arrays = writer._create_downscaled_arrays()
    
    # Should have 3 levels: original + 2 downscaled
    assert len(arrays) == 3
    
    # Check shapes
    assert arrays[0].shape == sample_2d_image.shape  # Original
    assert arrays[1].shape == (50, 50)  # Half size
    assert arrays[2].shape == (25, 25)  # Quarter size


def test_create_downscaled_arrays_no_downscaling(temp_dir, sample_2d_image):
    """Test that _create_downscaled_arrays returns only original when no downscaling."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    
    writer = OmeZarrImage(
        path=path,
        image=sample_2d_image,
        dims=dims,
        downscale_levels=None
    )
    
    arrays = writer._create_downscaled_arrays()
    
    # Should have only 1 level: original
    assert len(arrays) == 1
    assert arrays[0].shape == sample_2d_image.shape


def test_write_not_implemented(temp_dir, sample_2d_image):
    """Test that write method raises NotImplementedError."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    
    writer = OmeZarrImage(
        path=path,
        image=sample_2d_image,
        dims=dims
    )
    
    with pytest.raises(NotImplementedError):
        writer.write(
            image=sample_2d_image,
            pixel_size=(0.1, 0.1),
            units=["micrometer", "micrometer"]
        )


def test_3d_image_initialization(temp_dir, sample_3d_image):
    """Test initialization with 3D image."""
    path = temp_dir / "test.zarr"
    dims = ["z", "y", "x"]
    
    writer = OmeZarrImage(
        path=path,
        image=sample_3d_image,
        dims=dims
    )
    
    assert writer.image.shape == sample_3d_image.shape
    assert writer.dims == dims


def test_multichannel_image_dims(temp_dir):
    """Test initialization with multichannel image."""
    multichannel_image = np.random.randint(0, 255, size=(3, 100, 100), dtype=np.uint8)
    path = temp_dir / "test.zarr"
    dims = ["c", "y", "x"]
    
    writer = OmeZarrImage(
        path=path,
        image=multichannel_image,
        dims=dims
    )
    
    assert writer.image.shape == (3, 100, 100)
    assert writer.dims == dims

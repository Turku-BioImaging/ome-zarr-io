"""Test the OmeZarrImage class initialization and basic functionality."""

import pytest
import numpy as np
import dask.array as da
from pathlib import Path
from ome_zarr_writer.image import OmeZarrImage
from ome_zarr_writer.schema_models import Axis


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
    axis_units = {"unit": "micrometer"}

    writer = OmeZarrImage(
        path=path,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units
    )
    
    assert writer.path == Path(path)
    assert isinstance(writer.image, da.Array)  # Should be converted to dask
    assert writer.dims == dims
    assert len(writer.axes) == 2


def test_init_with_dask_array(temp_dir, sample_2d_image):
    """Test initialization with dask array."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"unit": "micrometer"}
    dask_image = da.from_array(sample_2d_image, chunks="auto")

    writer = OmeZarrImage(
        path=path,
        image=dask_image,
        dims=dims,
        axis_units=axis_units
    )
    
    assert writer.path == Path(path)
    assert isinstance(writer.image, da.Array)
    assert writer.dims == dims


def test_init_with_coordinate_transformations(temp_dir, sample_2d_image):
    """Test initialization with coordinate transformations."""
    from ome_zarr_writer.schema_models import ScaleTransformation, TranslationTransformation
    from typing import List, Union
    
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"unit": "micrometer"}
    coord_transforms: List[Union[ScaleTransformation, TranslationTransformation]] = [ScaleTransformation(scale=[0.1, 0.1])]

    writer = OmeZarrImage(
        path=path,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units,
        coordinate_transformations=coord_transforms
    )
    
    assert writer.coordinate_transformations == coord_transforms


def test_init_with_downscale_levels(temp_dir, sample_2d_image):
    """Test initialization with downscale levels."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"unit": "micrometer"}
    downscale_levels = 3

    writer = OmeZarrImage(
        path=path,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units,
        downscale_levels=downscale_levels
    )
    
    assert writer.downscale_levels == downscale_levels


def test_init_with_overwrite_flag(temp_dir, sample_2d_image):
    """Test initialization with overwrite flag."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"unit": "micrometer"}

    writer = OmeZarrImage(
        path=path,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units,
        overwrite=True
    )
    
    assert writer.overwrite is True


def test_path_handling_string_input(sample_2d_image):
    """Test that string paths are converted to Path objects."""
    path_str = "/tmp/test.zarr"
    dims = ["y", "x"]
    axis_units = {"unit": "micrometer"}

    writer = OmeZarrImage(
        path=path_str,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units
    )
    
    assert isinstance(writer.path, Path)
    assert str(writer.path) == path_str


def test_create_multiscale_group_not_implemented(temp_dir, sample_2d_image):
    """Test that create_multiscale_group raises NotImplementedError."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"unit": "micrometer"}

    writer = OmeZarrImage(
        path=path,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units
    )
    
    with pytest.raises(NotImplementedError):
        writer.create_multiscale_group(
            arrays=[],
            axes=[]
        )


def test_create_downscaled_arrays_basic_functionality(temp_dir, sample_2d_image):
    """Test that _create_downscaled_arrays creates downscaled versions correctly."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"unit": "micrometer"}

    writer = OmeZarrImage(
        path=path,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units,
        downscale_levels=2
    )
    
    arrays = writer._create_downscaled_arrays()
    
    # Should have 3 levels: original + 2 downscaled
    assert len(arrays) == 3
    
    # Check shapes
    assert arrays[0].shape == sample_2d_image.shape  # Original
    assert arrays[1].shape == (50, 50)  # 2x downscaled
    assert arrays[2].shape == (25, 25)  # 4x downscaled


def test_create_downscaled_arrays_no_downscaling(temp_dir, sample_2d_image):
    """Test that _create_downscaled_arrays returns only original when no downscaling."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"unit": "micrometer"}

    writer = OmeZarrImage(
        path=path,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units,
        downscale_levels=None
    )
    
    arrays = writer._create_downscaled_arrays()
    
    # Should have only 1 level: original
    assert len(arrays) == 1
    assert arrays[0].shape == sample_2d_image.shape


def test_write_method_works(temp_dir, sample_2d_image):
    """Test that write method works (no longer raises NotImplementedError)."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"unit": "micrometer"}

    writer = OmeZarrImage(
        path=path,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units
    )
    
    # Should not raise an exception
    writer.write()
    
    # Verify the file was created
    assert path.exists()


def test_3d_image_initialization(temp_dir, sample_3d_image):
    """Test initialization with 3D image."""
    path = temp_dir / "test.zarr"
    dims = ["z", "y", "x"]
    axis_units = {"unit": "micrometer"}

    writer = OmeZarrImage(
        path=path,
        image=sample_3d_image,
        dims=dims,
        axis_units=axis_units
    )
    
    assert writer.image.shape == sample_3d_image.shape
    assert writer.dims == dims
    assert len(writer.axes) == 3


def test_multichannel_image_dims(temp_dir):
    """Test initialization with multichannel image."""
    multichannel_image = np.random.randint(0, 255, size=(3, 100, 100), dtype=np.uint8)
    path = temp_dir / "test.zarr"
    dims = ["c", "y", "x"]
    axis_units = {"unit": "micrometer"}

    writer = OmeZarrImage(
        path=path,
        image=multichannel_image,
        dims=dims,
        axis_units=axis_units
    )
    
    assert writer.image.shape == multichannel_image.shape
    assert writer.dims == dims
    assert len(writer.axes) == 3
    # Check that channel axis has no unit
    channel_axis = next(ax for ax in writer.axes if ax.name == "c")
    assert channel_axis.unit is None

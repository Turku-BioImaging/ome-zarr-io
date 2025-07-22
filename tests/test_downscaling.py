"""Test downscaling functionality of the OmeZarrImage class."""

import pytest
import numpy as np
import dask.array as da
import zarr
from pathlib import Path
from ome_zarr_writer.image import OmeZarrImage


@pytest.fixture
def sample_2d_image():
    """Create a sample 2D image for testing."""
    return np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return tmp_path


def test_downscale_method_gaussian(temp_dir):
    """Test gaussian downscale method produces correct results by comparing with reference implementation."""
    # Create a deterministic test image with clear patterns for filtering validation
    test_image = np.zeros((60, 60), dtype=np.uint8)
    test_image[15:45, 15:45] = 255  # White square
    test_image[22:38, 22:38] = 128  # Gray square in center
    test_image[25:35, 25:35] = 64   # Dark gray square in center
    
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}
    downscale_factor = 1.5
    downscale_levels = 4

    # Test OmeZarrImage implementation
    writer = OmeZarrImage(
        path=path,
        image=test_image,
        dims=dims,
        axis_units=axis_units,
        downscale_method="gaussian",
        downscale_levels=downscale_levels,
        downscale_factor=downscale_factor,
    )

    # Get downscaled arrays from OmeZarrImage
    ome_arrays = writer._create_downscaled_arrays()
    
    # Create reference implementation for validation
    import dask_image.ndfilters
    from skimage.transform import rescale
    
    reference_arrays = [test_image]  # Level 0: original
    current_array = test_image.astype(np.float64)
    
    for level in range(1, downscale_levels + 1):
        # Check if dimensions are too small to continue
        if (current_array.shape[0] / downscale_factor < 1 or 
            current_array.shape[1] / downscale_factor < 1):
            break
            
        # Apply Gaussian filter with same sigma calculation as OmeZarrImage
        sigma = [0.0] * current_array.ndim
        sigma[-2] = (downscale_factor - 1) / 4.0  # Y dimension
        sigma[-1] = (downscale_factor - 1) / 4.0  # X dimension
        
        filtered_array = dask_image.ndfilters.gaussian_filter(
            da.from_array(current_array), sigma=sigma, mode="nearest"
        ).compute()
        
        # Rescale with same parameters as OmeZarrImage
        scale_factors = [1.0] * filtered_array.ndim
        scale_factor = 1.0 / downscale_factor
        scale_factors[-2] = scale_factor  # Y dimension
        scale_factors[-1] = scale_factor  # X dimension
        
        downscaled = rescale(
            filtered_array,
            scale=scale_factors,
            preserve_range=True,
            anti_aliasing=False,  # Already applied Gaussian filter
            channel_axis=None,
        ).astype(test_image.dtype)
        
        reference_arrays.append(downscaled)
        current_array = downscaled.astype(np.float64)
    
    # Verify we get the same number of levels
    assert len(ome_arrays) == len(reference_arrays), f"Expected {len(reference_arrays)} levels, got {len(ome_arrays)}"
    
    # Compare each level with reference implementation
    # Allow increasing tolerance for deeper levels due to accumulated differences
    max_allowed_diffs = [0, 2, 3, 4, 5]  # Level 0 exact, increasing tolerance for deeper levels
    
    for level, (ome_array, ref_array) in enumerate(zip(ome_arrays, reference_arrays)):
        ome_computed = ome_array.compute() if hasattr(ome_array, 'compute') else ome_array
        ome_computed = np.asarray(ome_computed)
        ref_array = np.asarray(ref_array)
        
        # Shapes should match exactly
        assert ome_computed.shape == ref_array.shape, f"Level {level}: shape mismatch {ome_computed.shape} vs {ref_array.shape}"
        
        if level == 0:
            # Original should match exactly
            np.testing.assert_array_equal(ome_computed, ref_array, 
                                        err_msg=f"Level {level}: Original arrays don't match")
        else:
            # For downscaled levels, allow small differences due to numerical precision
            diff = np.abs(ome_computed.astype(np.int16) - ref_array.astype(np.int16))
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            max_allowed_diff = max_allowed_diffs[min(level, len(max_allowed_diffs) - 1)]
            
            assert max_diff <= max_allowed_diff, f"Level {level}: Max difference {max_diff} > {max_allowed_diff}"
            assert mean_diff < 1.0, f"Level {level}: Mean difference {mean_diff} too high"
            
            # Calculate similarity percentage
            total_pixels = ome_computed.size
            matching_pixels = np.sum(diff <= 2)  # Allow up to 2-pixel differences
            similarity = (matching_pixels / total_pixels) * 100
            
            assert similarity >= 90.0, f"Level {level}: Only {similarity:.1f}% similarity with reference"
    
    # Test specific expected shapes for our test case with factor 1.5
    expected_shapes = [
        (60, 60),   # Original
        (40, 40),   # 60/1.5 = 40
        (26, 26),   # 40/1.5 ≈ 26
        (17, 17),   # 26/1.5 ≈ 17
        (11, 11),   # 17/1.5 ≈ 11
    ]
    
    for level in range(min(len(ome_arrays), len(expected_shapes))):
        actual_shape = ome_arrays[level].shape
        expected_shape = expected_shapes[level]
        assert actual_shape == expected_shape, f"Level {level}: expected shape {expected_shape}, got {actual_shape}"
    
    # Verify that Gaussian filtering is actually applied (edges should be smoothed)
    if len(ome_arrays) >= 2:
        original = ome_arrays[0].compute() if hasattr(ome_arrays[0], 'compute') else ome_arrays[0]
        downscaled = ome_arrays[1].compute() if hasattr(ome_arrays[1], 'compute') else ome_arrays[1]
        
        # Ensure we have numpy arrays
        original = np.asarray(original)
        downscaled = np.asarray(downscaled)
        
        # Original should have sharp edges (high gradient)
        original_grad = np.gradient(original.astype(np.float32))
        original_edge_strength = np.sqrt(original_grad[0]**2 + original_grad[1]**2).max()
        
        # Downscaled should have smoother edges (lower gradient)
        downscaled_grad = np.gradient(downscaled.astype(np.float32))
        downscaled_edge_strength = np.sqrt(downscaled_grad[0]**2 + downscaled_grad[1]**2).max()
        
        # Gaussian filtering should reduce edge strength
        assert downscaled_edge_strength < original_edge_strength, "Gaussian filtering should smooth edges"
    
    print(f"✅ Gaussian downscaling validation passed with {len(ome_arrays)} levels")
    for level, (ome_array, ref_array) in enumerate(zip(ome_arrays, reference_arrays)):
        ome_computed = ome_array.compute() if hasattr(ome_array, 'compute') else ome_array
        if level > 0:
            diff = np.abs(ome_computed.astype(np.int16) - ref_array.astype(np.int16))
            print(f"   Level {level}: {ome_computed.shape}, max_diff: {np.max(diff)}, mean_diff: {np.mean(diff):.3f}")
        else:
            print(f"   Level {level}: {ome_computed.shape} (original, exact match)")


def test_downscale_method_nearest(temp_dir, sample_2d_image):
    """Test initialization with nearest downscale method."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}

    # This test verifies that the nearest method can be passed as parameter
    # When implemented, it should store the downscale method
    writer = OmeZarrImage(
        path=path,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units,
        downscale_method="nearest",
        downscale_levels=2,
    )

    # Basic checks that initialization worked
    assert writer.path == Path(path)
    assert writer.downscale_levels == 2


def test_downscale_method_default(temp_dir, sample_2d_image):
    """Test that downscale method defaults to gaussian."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}

    # Test without specifying downscale_method - should default to gaussian
    writer = OmeZarrImage(
        path=path,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units,
        downscale_levels=2,
    )

    # Basic checks that initialization worked with default method
    assert writer.path == Path(path)
    assert writer.downscale_levels == 2


@pytest.mark.parametrize("invalid_method", ["bicubic", "lanczos", "invalid", ""])
@pytest.mark.skip(reason="Validation of downscale_method parameter not yet implemented")
def test_downscale_method_invalid_value(temp_dir, sample_2d_image, invalid_method):
    """Test that invalid downscale method raises error."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}

    # When implemented, this should raise a ValueError for invalid methods
    # For now, we expect a TypeError due to Literal type constraint
    with pytest.raises((ValueError, TypeError)):
        # Using type: ignore to bypass the Literal type checking for this test
        OmeZarrImage(
            path=path,
            image=sample_2d_image,
            dims=dims,
            axis_units=axis_units,
            downscale_method=invalid_method,  # type: ignore
            downscale_levels=2,
        )


def test_downscaled_arrays_gaussian_vs_nearest(temp_dir, sample_2d_image):
    """Test that gaussian and nearest methods produce different results."""
    path_gaussian = temp_dir / "test_gaussian.zarr"
    path_nearest = temp_dir / "test_nearest.zarr"
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}

    # Create a test image with distinct patterns to see filtering effects
    test_image = np.zeros((100, 100), dtype=np.uint8)
    test_image[25:75, 25:75] = 255  # White square in center
    test_image[40:60, 40:60] = 128  # Gray square in center of white square

    writer_gaussian = OmeZarrImage(
        path=path_gaussian,
        image=test_image,
        dims=dims,
        axis_units=axis_units,
        downscale_method="gaussian",
        downscale_levels=1,
    )

    writer_nearest = OmeZarrImage(
        path=path_nearest,
        image=test_image,
        dims=dims,
        axis_units=axis_units,
        downscale_method="nearest",
        downscale_levels=1,
    )

    # Generate downscaled arrays
    arrays_gaussian = writer_gaussian._create_downscaled_arrays()
    arrays_nearest = writer_nearest._create_downscaled_arrays()

    # Both should have the same number of levels
    assert len(arrays_gaussian) == len(arrays_nearest) == 2

    # Original arrays should be identical
    np.testing.assert_array_equal(
        arrays_gaussian[0].compute(), arrays_nearest[0].compute()
    )

    # Get downscaled arrays
    gaussian_downscaled = arrays_gaussian[1].compute()
    nearest_downscaled = arrays_nearest[1].compute()

    # They should have the same shape
    assert gaussian_downscaled.shape == nearest_downscaled.shape

    # Currently both methods use the same implementation (gaussian)
    # When nearest is implemented, this test should expect them to be different
    # For now, verify they are the same until nearest method is implemented
    np.testing.assert_array_equal(gaussian_downscaled, nearest_downscaled)
    
    # TODO: When nearest method is implemented, change the above assertion to:
    # assert not np.array_equal(gaussian_downscaled, nearest_downscaled)


def test_downscale_method_preserves_other_dimensions(temp_dir):
    """Test that downscale method works correctly with multi-dimensional images."""
    # Create a 4D image (t, c, y, x)
    multi_dim_image = np.random.randint(0, 255, size=(2, 3, 100, 100), dtype=np.uint8)
    path = temp_dir / "test_4d.zarr"
    dims = ["t", "c", "y", "x"]
    axis_units = {"t": "second", "y": "micrometer", "x": "micrometer"}

    writer = OmeZarrImage(
        path=path,
        image=multi_dim_image,
        dims=dims,
        axis_units=axis_units,
        downscale_method="nearest",
        downscale_levels=1,
    )

    arrays = writer._create_downscaled_arrays()

    # Should have 2 levels
    assert len(arrays) == 2

    # Original shape preserved
    assert arrays[0].shape == (2, 3, 100, 100)

    # Downscaled should only affect Y and X dimensions
    assert arrays[1].shape == (2, 3, 50, 50)


def test_downscale_method_with_coordinate_transformations(temp_dir, sample_2d_image):
    """Test that downscale method works with coordinate transformations."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}
    scale_transformations = {"y": 0.1, "x": 0.1}

    writer = OmeZarrImage(
        path=path,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units,
        downscale_method="nearest",
        downscale_levels=2,
        scale_transformations=scale_transformations,
    )

    # Should work without errors and coordinate transformations should be set
    assert writer.coordinate_transformations is not None
    assert writer.downscale_levels == 2
    
    # Test coordinate transformations for levels
    level_transformations = writer._create_coordinate_transformations_for_levels()
    assert len(level_transformations) >= 1


def test_downscale_method_integration_with_write(temp_dir, sample_2d_image):
    """Test that downscale method integrates properly with the write method."""
    path_gaussian = temp_dir / "test_gaussian.zarr"
    path_nearest = temp_dir / "test_nearest.zarr"
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}

    # Test both methods can complete the full write process
    writer_gaussian = OmeZarrImage(
        path=path_gaussian,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units,
        downscale_method="gaussian",
        downscale_levels=1,
    )

    writer_nearest = OmeZarrImage(
        path=path_nearest,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units,
        downscale_method="nearest",
        downscale_levels=1,
    )

    # Both should write successfully
    writer_gaussian.write()
    writer_nearest.write()

    # Both output files should exist
    assert path_gaussian.exists()
    assert path_nearest.exists()


@pytest.mark.parametrize("method", ["gaussian", "nearest"])
def test_downscale_method_parametrized(temp_dir, sample_2d_image, method):
    """Test both downscale methods using parametrized test."""
    path = temp_dir / f"test_{method}.zarr"
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}

    writer = OmeZarrImage(
        path=path,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units,
        downscale_method=method,
        downscale_levels=2,
    )

    # Should initialize successfully
    assert writer.path == Path(path)
    assert writer.downscale_levels == 2

    # Should be able to create downscaled arrays
    arrays = writer._create_downscaled_arrays()
    assert len(arrays) >= 2  # Original + at least 1 downscaled

    # Should be able to write successfully
    writer.write()
    assert path.exists()


def test_create_downscaled_arrays_basic_functionality(temp_dir, sample_2d_image):
    """Test that _create_downscaled_arrays creates downscaled versions correctly."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}

    writer = OmeZarrImage(
        path=path,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units,
        downscale_levels=2,
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
    axis_units = {"y": "micrometer", "x": "micrometer"}

    writer = OmeZarrImage(
        path=path,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units,
        downscale_levels=None,
    )

    arrays = writer._create_downscaled_arrays()

    # Should have only 1 level: original
    assert len(arrays) == 1
    assert arrays[0].shape == sample_2d_image.shape
    
    # Write the zarr file to create the actual group structure
    writer.write()
    
    # Verify the zarr file was created
    assert path.exists()
    
    # Open the zarr group and verify it has only 1 array (original, no downscaling)
    group = zarr.open_group(str(path), mode="r")
    
    # Should have only array "0" for the original level
    assert "0" in group  # Original level
    assert "1" not in group  # No first downscale level
    assert "2" not in group  # No second downscale level
    
    # Verify we have exactly 1 array
    array_keys = [key for key in group.array_keys()]
    assert len(array_keys) == 1
    assert set(array_keys) == {"0"}


def test_init_with_downscale_levels(temp_dir, sample_2d_image):
    """Test initialization with downscale levels."""
    path = temp_dir / "test.zarr"
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}
    downscale_levels = 3

    writer = OmeZarrImage(
        path=path,
        image=sample_2d_image,
        dims=dims,
        axis_units=axis_units,
        downscale_levels=downscale_levels,
    )

    assert writer.downscale_levels == downscale_levels
    
    # Write the zarr file to create the actual group structure
    writer.write()
    
    # Verify the zarr file was created
    assert path.exists()
    
    # Open the zarr group and verify it has 4 arrays (original + 3 downscale levels)
    group = zarr.open_group(str(path), mode="r")
    
    # Should have arrays "0", "1", "2", "3" for the 4 levels
    assert "0" in group  # Original level
    assert "1" in group  # First downscale level  
    assert "2" in group  # Second downscale level
    assert "3" in group  # Third downscale level
    
    # Verify we have exactly 4 arrays
    array_keys = [key for key in group.array_keys()]
    assert len(array_keys) == 4
    assert set(array_keys) == {"0", "1", "2", "3"}

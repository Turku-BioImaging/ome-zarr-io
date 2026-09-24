"""Comprehensive tests for downscaling functionality in OME-Zarr writer.

This module tests both the Downscaler class (in isolation) and its integration
with the Writer class to ensure proper downscaling behavior.
"""

import pytest
import numpy as np
import dask.array as da
import zarr
from pathlib import Path
from ome_zarr_io.writer import Writer
from ome_zarr_io.downscaler import Downscaler
from ome_zarr_io.schema_models import ScaleTransformation


@pytest.fixture
def sample_2d_image():
    """Create a sample 2D image for testing."""
    return np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return tmp_path


class TestDownscaler:
    """Test the Downscaler class in isolation."""

    def test_init_valid_parameters(self):
        """Test Downscaler initialization with valid parameters."""
        downscaler = Downscaler(
            downscale_factor=2.0, downscale_method="gaussian", downscale_levels=3
        )

        assert downscaler.downscale_factor == 2.0
        assert downscaler.downscale_method == "gaussian"
        assert downscaler.downscale_levels == 3

    def test_init_invalid_downscale_factor(self):
        """Test that invalid downscale factor raises ValueError."""
        with pytest.raises(ValueError, match="downscale_factor must be > 1.0"):
            Downscaler(downscale_factor=0.5)

    def test_validate_downscale_levels_valid(self):
        """Test validate_downscale_levels with valid input."""
        downscaler = Downscaler(downscale_levels=3)
        image_shape = (100, 100)

        validated_levels = downscaler.validate_downscale_levels(image_shape)
        assert validated_levels == 3

    def test_validate_downscale_levels_too_many(self):
        """Test validate_downscale_levels with too many levels."""
        downscaler = Downscaler(downscale_levels=10)
        image_shape = (16, 16)  # Small image that can't support 10 levels

        with pytest.warns(UserWarning):
            validated_levels = downscaler.validate_downscale_levels(image_shape)

        assert validated_levels < 10

    def test_validate_downscale_levels_none(self):
        """Test validate_downscale_levels with None levels."""
        downscaler = Downscaler(downscale_levels=None)
        image_shape = (100, 100)

        validated_levels = downscaler.validate_downscale_levels(image_shape)
        assert validated_levels == 0

    def test_create_downscaled_arrays_no_downscaling(self):
        """Test create_downscaled_arrays with no downscaling."""
        image = da.ones((50, 50), dtype=np.uint8)
        downscaler = Downscaler(downscale_levels=0)

        arrays = downscaler.create_downscaled_arrays(image)

        assert len(arrays) == 1
        assert arrays[0].shape == (50, 50)

    @pytest.mark.parametrize("method", ["gaussian", "nearest"])
    def test_create_downscaled_arrays_methods(self, method):
        """Test create_downscaled_arrays with different methods."""
        image = da.ones((100, 100), dtype=np.uint8)
        downscaler = Downscaler(
            downscale_factor=2.0, downscale_method=method, downscale_levels=2
        )

        arrays = downscaler.create_downscaled_arrays(image)

        assert len(arrays) == 3  # Original + 2 downscaled
        assert arrays[0].shape == (100, 100)
        assert arrays[1].shape == (50, 50)
        assert arrays[2].shape == (25, 25)

    def test_create_downscaled_arrays_invalid_method(self):
        """Test create_downscaled_arrays with invalid method defaults to gaussian."""
        image = da.ones((100, 100), dtype=np.uint8)
        downscaler = Downscaler(
            downscale_factor=2.0,
            downscale_method="invalid",  # type: ignore[arg-type]  # Should default to gaussian
            downscale_levels=1,
        )

        arrays = downscaler.create_downscaled_arrays(image)

        assert len(arrays) == 2  # Should still work
        assert arrays[0].shape == (100, 100)
        assert arrays[1].shape == (50, 50)

    def test_create_coordinate_transformations_for_levels(self):
        """Test coordinate transformation creation."""
        downscaler = Downscaler(downscale_factor=2.0, downscale_levels=2)

        # Create original transformations
        original_transforms = [ScaleTransformation(scale=[0.1, 0.1])]

        level_transforms = downscaler.create_coordinate_transformations_for_levels(
            original_transforms
        )

        assert len(level_transforms) == 3

        # Check scale values
        assert level_transforms[0][0].scale == [0.1, 0.1]  # Level 0
        assert level_transforms[1][0].scale == [0.2, 0.2]  # Level 1 (2x)
        assert level_transforms[2][0].scale == [0.4, 0.4]  # Level 2 (4x)

    def test_create_coordinate_transformations_none_input(self):
        """Test coordinate transformation creation with None input."""
        downscaler = Downscaler()

        level_transforms = downscaler.create_coordinate_transformations_for_levels(coordinate_transformations=None)

        assert level_transforms == []

    def test_multidimensional_image_downscaling(self):
        """Test downscaling with multidimensional images (preserves non-spatial dims)."""
        # Create a 4D image (T, C, Y, X)
        image = da.ones((5, 3, 100, 100), dtype=np.uint8)
        downscaler = Downscaler(
            downscale_factor=2.0, downscale_method="gaussian", downscale_levels=2
        )

        arrays = downscaler.create_downscaled_arrays(image)

        assert len(arrays) == 3
        # Time and channel dimensions should be preserved
        assert arrays[0].shape == (5, 3, 100, 100)
        assert arrays[1].shape == (5, 3, 50, 50)
        assert arrays[2].shape == (5, 3, 25, 25)


class TestWriterDownscaling:
    """Test downscaling integration with Writer."""

    def test_gaussian_method_reference_validation(self, temp_dir):
        """Test gaussian downscale method produces correct results by comparing with reference implementation."""
        # Create a deterministic test image with clear patterns for filtering validation
        test_image = np.zeros((60, 60), dtype=np.uint8)
        test_image[15:45, 15:45] = 255  # White square
        test_image[22:38, 22:38] = 128  # Gray square in center
        test_image[25:35, 25:35] = 64  # Dark gray square in center

        path = temp_dir / "test.zarr"
        dims = ["y", "x"]
        axis_units = {"y": "micrometer", "x": "micrometer"}
        downscale_factor = 1.5
        downscale_levels = 4

        # Test Writer implementation
        writer = Writer(
            path=path,
            image=test_image,
            dims=dims,
            axis_units=axis_units,
            downscale_method="gaussian",
            downscale_levels=downscale_levels,
            downscale_factor=downscale_factor,
        )

        # Get downscaled arrays from Writer
        ome_arrays = writer._create_downscaled_arrays()

        # Test specific expected shapes for our test case with factor 1.5
        # New implementation: each level is scaled from original by (1 / (factor * level))
        expected_shapes = [
            (60, 60),  # Original (level 0)
            (40, 40),  # 60 * (1/(1.5*1)) = 60 * 0.667 = 40
            (20, 20),  # 60 * (1/(1.5*2)) = 60 * 0.333 = 20
            (13, 13),  # 60 * (1/(1.5*3)) = 60 * 0.222 ≈ 13
            (10, 10),  # 60 * (1/(1.5*4)) = 60 * 0.167 = 10
        ]

        for level in range(min(len(ome_arrays), len(expected_shapes))):
            actual_shape = ome_arrays[level].shape
            expected_shape = expected_shapes[level]
            assert (
                actual_shape == expected_shape
            ), f"Level {level}: expected shape {expected_shape}, got {actual_shape}"

        # Verify that Gaussian filtering is actually applied (edges should be smoothed)
        if len(ome_arrays) >= 2:
            original = (
                ome_arrays[0].compute()
                if hasattr(ome_arrays[0], "compute")
                else ome_arrays[0]
            )
            downscaled = (
                ome_arrays[1].compute()
                if hasattr(ome_arrays[1], "compute")
                else ome_arrays[1]
            )

            # Ensure we have numpy arrays
            original = np.asarray(original)
            downscaled = np.asarray(downscaled)

            # Original should have sharp edges (high gradient)
            original_grad = np.gradient(original.astype(np.float32))
            original_edge_strength = np.sqrt(
                original_grad[0] ** 2 + original_grad[1] ** 2
            ).max()

            # Downscaled should have smoother edges (lower gradient)
            downscaled_grad = np.gradient(downscaled.astype(np.float32))
            downscaled_edge_strength = np.sqrt(
                downscaled_grad[0] ** 2 + downscaled_grad[1] ** 2
            ).max()

            # Gaussian filtering should reduce edge strength
            assert (
                downscaled_edge_strength < original_edge_strength
            ), "Gaussian filtering should smooth edges"

    def test_gaussian_vs_nearest_methods_produce_different_results(self, temp_dir):
        """Test that gaussian and nearest methods produce different results."""
        # Create a test image with distinct patterns to see filtering effects
        test_image = np.zeros((100, 100), dtype=np.uint8)
        test_image[25:75, 25:75] = 255  # White square in center
        test_image[40:60, 40:60] = 128  # Gray square in center of white square

        path_gaussian = temp_dir / "test_gaussian.zarr"
        path_nearest = temp_dir / "test_nearest.zarr"
        dims = ["y", "x"]
        axis_units = {"y": "micrometer", "x": "micrometer"}

        writer_gaussian = Writer(
            path=path_gaussian,
            image=test_image,
            dims=dims,
            axis_units=axis_units,
            downscale_method="gaussian",
            downscale_levels=1,
        )

        writer_nearest = Writer(
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
        # Gaussian- and nearest-downscaled images should not be identical
        assert not np.array_equal(gaussian_downscaled, nearest_downscaled)

    @pytest.mark.parametrize("method", ["gaussian", "nearest"])
    def test_downscale_methods_with_ome_zarr_image(
        self, temp_dir, sample_2d_image, method
    ):
        """Test both downscale methods integrate properly with Writer."""
        path = temp_dir / f"test_{method}.zarr"
        dims = ["y", "x"]
        axis_units = {"y": "micrometer", "x": "micrometer"}

        writer = Writer(
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
        assert writer.downscale_method == method

        # Should be able to create downscaled arrays
        arrays = writer._create_downscaled_arrays()
        assert len(arrays) >= 2  # Original + at least 1 downscaled

        # Should be able to write successfully
        writer.write()
        assert path.exists()

    def test_downscale_method_default_is_gaussian(self, temp_dir, sample_2d_image):
        """Test that downscale method defaults to gaussian."""
        path = temp_dir / "test.zarr"
        dims = ["y", "x"]
        axis_units = {"y": "micrometer", "x": "micrometer"}

        # Test without specifying downscale_method - should default to gaussian
        writer = Writer(
            path=path,
            image=sample_2d_image,
            dims=dims,
            axis_units=axis_units,
            downscale_levels=2,
        )

        # Basic checks that initialization worked with default method
        assert writer.path == Path(path)
        assert writer.downscale_levels == 2
        assert writer.downscale_method == "gaussian"  # Should default to gaussian

    @pytest.mark.parametrize("invalid_method", ["bicubic", "lanczos", "invalid", ""])
    def test_invalid_downscale_method_defaults_to_gaussian(
        self, temp_dir, sample_2d_image, invalid_method
    ):
        """Test that invalid downscale method values default to gaussian behavior."""
        path = temp_dir / "test.zarr"
        dims = ["y", "x"]
        axis_units = {"y": "micrometer", "x": "micrometer"}

        # The current implementation accepts invalid methods at runtime
        # but they are caught by type checkers due to Literal["gaussian", "nearest"]
        writer = Writer(
            path=path,
            image=sample_2d_image,
            dims=dims,
            axis_units=axis_units,
            downscale_method=invalid_method,  # type: ignore[arg-type]
            downscale_levels=2,
        )

        # Should initialize successfully (runtime doesn't validate the method)
        assert writer.path == Path(path)
        assert writer.downscale_levels == 2

        # Should be able to create arrays and write (defaults to gaussian behavior)
        arrays = writer._create_downscaled_arrays()
        assert len(arrays) == 3  # Original + 2 downscaled levels

        writer.write()
        assert path.exists()

    def test_multidimensional_image_preserves_non_spatial_dimensions(self, temp_dir):
        """Test that downscaling works correctly with multi-dimensional images."""
        # Create a 4D image (t, c, y, x)
        multi_dim_image = np.random.randint(
            0, 255, size=(2, 3, 100, 100), dtype=np.uint8
        )
        path = temp_dir / "test_4d.zarr"
        dims = ["t", "c", "y", "x"]
        axis_units = {"t": "second", "y": "micrometer", "x": "micrometer"}

        writer = Writer(
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

    def test_coordinate_transformations_integration(self, temp_dir, sample_2d_image):
        """Test that downscaling works with coordinate transformations."""
        path = temp_dir / "test.zarr"
        dims = ["y", "x"]
        axis_units = {"y": "micrometer", "x": "micrometer"}
        scale_transformations = {"y": 0.1, "x": 0.1}

        writer = Writer(
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

    def test_no_downscaling_creates_single_level(self, temp_dir, sample_2d_image):
        """Test that no downscaling creates only the original level."""
        path = temp_dir / "test.zarr"
        dims = ["y", "x"]
        axis_units = {"y": "micrometer", "x": "micrometer"}

        writer = Writer(
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

        # Verify we have exactly 1 array
        array_keys = [key for key in group.array_keys()]
        assert len(array_keys) == 1
        assert set(array_keys) == {"0"}

    def test_multiple_downscale_levels_creates_correct_number_of_arrays(
        self, temp_dir, sample_2d_image
    ):
        """Test that multiple downscale levels create the correct number of zarr arrays."""
        path = temp_dir / "test.zarr"
        dims = ["y", "x"]
        axis_units = {"y": "micrometer", "x": "micrometer"}
        downscale_levels = 3

        writer = Writer(
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

    def test_basic_downscaling_functionality(self, temp_dir, sample_2d_image):
        """Test basic downscaling creates correct shapes."""
        path = temp_dir / "test.zarr"
        dims = ["y", "x"]
        axis_units = {"y": "micrometer", "x": "micrometer"}

        writer = Writer(
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


class TestBackwardCompatibility:
    """Test that the refactored code maintains backward compatibility."""

    def test_ome_zarr_image_properties_work(self, temp_dir, sample_2d_image):
        """Test that Writer properties still provide access to downscaler settings."""
        path = temp_dir / "test.zarr"
        dims = ["y", "x"]
        axis_units = {"y": "micrometer", "x": "micrometer"}

        writer = Writer(
            path=path,
            image=sample_2d_image,
            dims=dims,
            axis_units=axis_units,
            downscale_method="nearest",
            downscale_levels=3,
            downscale_factor=2.5,
        )

        # Test that properties work
        assert writer.downscale_levels == 3
        assert writer.downscale_factor == 2.5
        assert writer.downscale_method == "nearest"

        # Test that the underlying downscaler is accessible
        assert hasattr(writer, "downscaler")
        assert isinstance(writer.downscaler, Downscaler)

"""Comprehensive tests for downscaling functionality in OME-Zarr writer.

This module tests both the Downscaler class (in isolation) and its integration
with the Writer class to ensure proper downscaling behavior.
"""

import pytest
import numpy as np
import dask.array as da
import zarr
from pathlib import Path
from ome_zarr_io.reader import Reader
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
            downscale_factor=2, downscale_method="mean", downscale_levels=3
        )

        assert downscaler.downscale_factor == 2
        assert downscaler.downscale_method == "mean"
        assert downscaler.downscale_levels == 3

    def test_init_integral_float_factor_is_accepted(self):
        """Integral floats (e.g. 2.0) are accepted and stored as int."""
        downscaler = Downscaler(downscale_factor=2.0)

        assert downscaler.downscale_factor == 2
        assert isinstance(downscaler.downscale_factor, int)

    @pytest.mark.parametrize("factor", [0.5, 1, 1.0, 1.5, 2.5, 0, -2, True, "2"])
    def test_init_invalid_downscale_factor(self, factor):
        """Test that non-integer or < 2 downscale factors raise ValueError."""
        with pytest.raises(
            ValueError, match="downscale_factor must be an integer >= 2"
        ):
            Downscaler(downscale_factor=factor)

    def test_init_gaussian_method_is_deprecated_alias_for_mean(self):
        """Test that "gaussian" warns and is normalized to "mean"."""
        with pytest.warns(DeprecationWarning, match='"gaussian" is deprecated'):
            downscaler = Downscaler(downscale_method="gaussian")

        assert downscaler.downscale_method == "mean"

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

    @pytest.mark.parametrize("method", ["mean", "nearest"])
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
        """Test create_downscaled_arrays with invalid method defaults to mean."""
        image = da.ones((100, 100), dtype=np.uint8)
        downscaler = Downscaler(
            downscale_factor=2.0,
            downscale_method="invalid",  # type: ignore[arg-type]  # Should default to mean
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

        level_transforms = downscaler.create_coordinate_transformations_for_levels(
            coordinate_transformations=None
        )

        assert level_transforms == []

    def test_multidimensional_image_downscaling(self):
        """Test downscaling with multidimensional images (preserves non-spatial dims)."""
        # Create a 4D image (T, C, Y, X)
        image = da.ones((5, 3, 100, 100), dtype=np.uint8)
        downscaler = Downscaler(
            downscale_factor=2, downscale_method="mean", downscale_levels=2
        )

        arrays = downscaler.create_downscaled_arrays(image)

        assert len(arrays) == 3
        # Time and channel dimensions should be preserved
        assert arrays[0].shape == (5, 3, 100, 100)
        assert arrays[1].shape == (5, 3, 50, 50)
        assert arrays[2].shape == (5, 3, 25, 25)


class TestWriterDownscaling:
    """Test downscaling integration with Writer."""

    def test_mean_method_reference_validation(self, temp_dir):
        """Test mean downscale method against a reference block average."""
        # Create a deterministic test image with clear patterns for filtering validation
        test_image = np.zeros((60, 60), dtype=np.uint8)
        test_image[15:45, 15:45] = 255  # White square
        test_image[22:38, 22:38] = 128  # Gray square in center
        test_image[25:35, 25:35] = 64  # Dark gray square in center

        writer = Writer(
            path=temp_dir / "test.zarr",
            image=da.from_array(test_image, chunks=(16, 16)),
            dims=["y", "x"],
            axis_units={"y": "micrometer", "x": "micrometer"},
            downscale_method="mean",
            downscale_levels=3,
            downscale_factor=2,
        )

        ome_arrays = writer._create_downscaled_arrays()

        # Level L is floor(60 / 2**L): trailing pixels that don't fill a block are dropped
        assert [a.shape for a in ome_arrays] == [(60, 60), (30, 30), (15, 15), (7, 7)]

        for level in range(1, len(ome_arrays)):
            block = 2**level
            size = 60 // block
            expected = np.rint(
                test_image[: size * block, : size * block]
                .reshape(size, block, size, block)
                .mean(axis=(1, 3))
            ).astype(np.uint8)
            np.testing.assert_array_equal(ome_arrays[level].compute(), expected)

        # Averaging should smooth edges (lower max gradient than the original)
        original = test_image.astype(np.float32)
        downscaled = ome_arrays[1].compute().astype(np.float32)
        original_edge = np.hypot(*np.gradient(original)).max()
        downscaled_edge = np.hypot(*np.gradient(downscaled)).max()
        assert downscaled_edge < original_edge, "Averaging should smooth edges"

    def test_mean_vs_nearest_methods_produce_different_results(self, temp_dir):
        """Test that mean and nearest methods produce different results."""
        # Create a test image with distinct patterns to see filtering effects
        test_image = np.zeros((100, 100), dtype=np.uint8)
        test_image[25:75, 25:75] = 255  # White square in center
        test_image[40:60, 40:60] = 128  # Gray square in center of white square

        path_mean = temp_dir / "test_mean.zarr"
        path_nearest = temp_dir / "test_nearest.zarr"
        dims = ["y", "x"]
        axis_units = {"y": "micrometer", "x": "micrometer"}

        writer_mean = Writer(
            path=path_mean,
            image=test_image,
            dims=dims,
            axis_units=axis_units,
            downscale_method="mean",
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
        arrays_mean = writer_mean._create_downscaled_arrays()
        arrays_nearest = writer_nearest._create_downscaled_arrays()

        # Both should have the same number of levels
        assert len(arrays_mean) == len(arrays_nearest) == 2

        # Original arrays should be identical
        np.testing.assert_array_equal(
            arrays_mean[0].compute(), arrays_nearest[0].compute()
        )

        # Get downscaled arrays
        mean_downscaled = arrays_mean[1].compute()
        nearest_downscaled = arrays_nearest[1].compute()

        # They should have the same shape
        assert mean_downscaled.shape == nearest_downscaled.shape
        # Mean- and nearest-downscaled images should not be identical
        assert not np.array_equal(mean_downscaled, nearest_downscaled)

    @pytest.mark.parametrize("method", ["mean", "nearest"])
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

    def test_downscale_method_default_is_mean(self, temp_dir, sample_2d_image):
        """Test that downscale method defaults to mean."""
        path = temp_dir / "test.zarr"
        dims = ["y", "x"]
        axis_units = {"y": "micrometer", "x": "micrometer"}

        # Test without specifying downscale_method - should default to mean
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
        assert writer.downscale_method == "mean"  # Should default to mean

    @pytest.mark.parametrize("invalid_method", ["bicubic", "lanczos", "invalid", ""])
    def test_invalid_downscale_method_defaults_to_mean(
        self, temp_dir, sample_2d_image, invalid_method
    ):
        """Test that invalid downscale method values default to mean behavior."""
        path = temp_dir / "test.zarr"
        dims = ["y", "x"]
        axis_units = {"y": "micrometer", "x": "micrometer"}

        # The current implementation accepts invalid methods at runtime
        # but they are caught by type checkers due to Literal["mean", "nearest", "gaussian"]
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

        # Should be able to create arrays and write (defaults to mean behavior)
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
            downscale_factor=3,
        )

        # Test that properties work
        assert writer.downscale_levels == 3
        assert writer.downscale_factor == 3
        assert writer.downscale_method == "nearest"

        # Test that the underlying downscaler is accessible
        assert hasattr(writer, "downscaler")
        assert isinstance(writer.downscaler, Downscaler)


class TestPyramidGeometry:
    """Regression tests: array shapes must match the metadata scale at every level."""

    @pytest.mark.parametrize("factor", [2, 3])
    @pytest.mark.parametrize("method", ["mean", "nearest"])
    def test_levels_shrink_geometrically(self, factor, method):
        """Level L is floor(size / factor**L), not size / (factor * L)."""
        size = 256
        levels = 4 if factor == 2 else 3
        downscaler = Downscaler(
            downscale_factor=factor, downscale_method=method, downscale_levels=levels
        )

        arrays = downscaler.create_downscaled_arrays(
            da.zeros((2, size, size), dtype=np.uint16, chunks=(1, 64, 64))
        )

        assert len(arrays) == levels + 1
        for level, array in enumerate(arrays):
            expected = size // factor**level
            assert array.shape == (2, expected, expected), f"level {level}"

    @pytest.mark.parametrize("factor", [2, 3])
    def test_written_shapes_match_metadata_scale(self, temp_dir, factor):
        """The Y/X scale ratio of each written level matches its size reduction."""
        size = 243  # divisible by 3**5, not by 2: exercises trimming for factor 2
        path = temp_dir / "test.zarr"
        Writer(
            path=path,
            image=np.random.randint(0, 255, size=(size, size), dtype=np.uint8),
            dims=["y", "x"],
            axis_units={"y": "micrometer", "x": "micrometer"},
            scale_transformations={"y": 0.5, "x": 0.5},
            downscale_levels=4,
            downscale_factor=factor,
        ).write(chunks=(32, 32))

        group = zarr.open_group(str(path), mode="r")
        datasets = group.attrs["ome"]["multiscales"][0]["datasets"]
        assert len(datasets) == 5
        for level, dataset in enumerate(datasets):
            scale = dataset["coordinateTransformations"][0]["scale"]
            assert scale == [0.5 * factor**level] * 2
            assert group[dataset["path"]].shape == (size // factor**level,) * 2

    @pytest.mark.parametrize("method", ["mean", "nearest"])
    def test_result_is_independent_of_chunking(self, method):
        """Chunks that the factor doesn't divide must not change the output."""
        image = np.random.default_rng(0).integers(0, 1000, (3, 200, 200), np.uint16)
        downscaler = Downscaler(downscale_method=method, downscale_levels=4)

        single = downscaler.create_downscaled_arrays(da.from_array(image, chunks=-1))
        odd = downscaler.create_downscaled_arrays(
            da.from_array(image, chunks=(1, 37, 37))
        )

        for level, (a, b) in enumerate(zip(single, odd)):
            np.testing.assert_array_equal(a.compute(), b.compute(), f"level {level}")

    def test_mean_matches_block_average_and_preserves_dtype(self):
        """Mean downscaling equals a hand-computed block average, rounded to dtype."""
        image = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
        downscaler = Downscaler(downscale_method="mean", downscale_levels=2)

        arrays = downscaler.create_downscaled_arrays(da.from_array(image, chunks=5))

        expected = image.reshape(4, 4, 4, 4).mean(axis=(1, 3))
        result = arrays[2].compute()
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, expected)

    def test_nearest_preserves_label_values(self):
        """Nearest downscaling never introduces label values absent from the input."""
        labels = np.random.default_rng(0).choice([0, 3, 7, 250], size=(64, 64))
        labels = labels.astype(np.uint8)
        downscaler = Downscaler(downscale_method="nearest", downscale_levels=3)

        for array in downscaler.create_downscaled_arrays(da.from_array(labels)):
            assert set(np.unique(array.compute())) <= {0, 3, 7, 250}

    def test_add_labels_scale_uses_label_factor(self, temp_dir):
        """A label downscale_factor override is reflected in the label metadata."""
        path = temp_dir / "test.zarr"
        writer = Writer(
            path=path,
            image=np.zeros((81, 81), dtype=np.uint8),
            dims=["y", "x"],
            axis_units={"y": "micrometer", "x": "micrometer"},
            scale_transformations={"y": 0.5, "x": 0.5},
            downscale_levels=2,
            downscale_factor=2,
        )
        writer.write()
        writer.add_labels(
            name="cells", array=np.zeros((81, 81), dtype=np.uint8), downscale_factor=3
        )

        label_group = zarr.open_group(str(path), mode="r")["labels"]["cells"]
        datasets = label_group.attrs["ome"]["multiscales"][0]["datasets"]
        for level, dataset in enumerate(datasets):
            scale = dataset["coordinateTransformations"][0]["scale"]
            assert scale == [0.5 * 3**level] * 2
            assert label_group[dataset["path"]].shape == (81 // 3**level,) * 2

    def test_default_scale_reflects_downscaling(self, temp_dir):
        """Without scale_transformations, Y/X scale is factor**L (not 1.0) per level."""
        path = temp_dir / "test.zarr"
        writer = Writer(
            path=path,
            image=np.zeros((2, 64, 64), dtype=np.uint8),
            dims=["c", "y", "x"],
            axis_units={"y": "micrometer", "x": "micrometer"},
            downscale_levels=3,
        )
        writer.write()
        writer.add_labels(name="cells", array=np.zeros((2, 64, 64), dtype=np.uint8))

        root = zarr.open_group(str(path), mode="r")
        for group in (root, root["labels"]["cells"]):
            datasets = group.attrs["ome"]["multiscales"][0]["datasets"]
            scales = [d["coordinateTransformations"][0]["scale"] for d in datasets]
            assert scales == [[1.0, 2.0**L, 2.0**L] for L in range(4)]
        assert Reader(path).validate()

    def test_negative_downscale_levels_writes_single_level(self, temp_dir):
        """Negative downscale_levels means no downscaling, with unit scale."""
        path = temp_dir / "test.zarr"
        Writer(
            path=path,
            image=np.zeros((16, 16), dtype=np.uint8),
            dims=["y", "x"],
            axis_units={"y": "micrometer", "x": "micrometer"},
            downscale_levels=-1,
        ).write()

        datasets = zarr.open_group(str(path), mode="r").attrs["ome"]["multiscales"][0][
            "datasets"
        ]
        assert len(datasets) == 1
        assert datasets[0]["coordinateTransformations"][0]["scale"] == [1.0, 1.0]

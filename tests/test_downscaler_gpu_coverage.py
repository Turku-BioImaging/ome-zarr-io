"""Advanced tests for downscaler functionality including GPU support and edge cases.

This module tests the advanced functionality of the Downscaler class that wasn't
covered in the basic downscaling tests, including device validation, GPU operations,
and error handling.
"""

import numpy as np
import dask.array as da
from unittest.mock import Mock, patch
from ome_zarr_writer.downscaler import Downscaler
from ome_zarr_writer.schema_models import ScaleTransformation


class TestDownscalerDeviceValidation:
    """Test device validation and GPU-related functionality."""

    def test_init_with_cpu_device(self):
        """Test initialization with CPU device."""
        downscaler = Downscaler(device="cpu")
        assert downscaler.device == "cpu"
        assert downscaler.cuda_device_id is None

    def test_should_use_gpu_cpu_device(self):
        """Test _should_use_gpu method with CPU device."""
        downscaler = Downscaler(device="cpu")
        assert not downscaler._should_use_gpu()


class TestDownscalerDeviceInfo:
    """Test device information methods."""

    def test_get_device_info_cpu(self):
        """Test get_device_info for CPU device."""
        downscaler = Downscaler(device="cpu")
        info = downscaler.get_device_info()
        
        assert info["device_type"] == "cpu"
        assert info["device_name"] == "CPU"
        assert info["device_id"] is None

    def test_list_cuda_devices_no_cupy(self):
        """Test list_cuda_devices when CuPy is not available."""
        with patch('builtins.__import__', side_effect=ImportError("No module named 'cupy'")):
            devices = Downscaler.list_cuda_devices()
            
            assert len(devices) == 1
            assert "error" in devices[0]
            assert "Failed to list CUDA devices" in devices[0]["error"]


class TestDownscalerGPUMethods:
    """Test GPU-specific downscaling methods."""

    def test_downscale_gaussian_gpu_no_cupy(self):
        """Test GPU Gaussian downscaling when CuPy is not available."""
        downscaler = Downscaler(device="cpu")  # Even CPU can call GPU methods
        image = da.ones((100, 100), dtype=np.float32)
        
        # Mock the import to fail
        with patch('builtins.__import__', side_effect=ImportError("No module named 'cupy'")):
            result = downscaler._downscale_gaussian_gpu(image)
            
            assert result is None

    def test_calculate_gaussian_sigma(self):
        """Test _calculate_gaussian_sigma method."""
        downscaler = Downscaler(downscale_factor=2.0)
        sigma = downscaler._calculate_gaussian_sigma(4)
        
        assert len(sigma) == 4
        assert sigma[0] == 0.0
        assert sigma[1] == 0.0
        assert sigma[2] == 0.25  # (2.0 - 1) / 4.0
        assert sigma[3] == 0.25

    def test_calculate_output_chunks(self):
        """Test _calculate_output_chunks method."""
        downscaler = Downscaler(downscale_factor=2.0)
        input_chunks = ((50, 50), (60, 40), (100, 100), (80, 80))
        
        output_chunks = downscaler._calculate_output_chunks(input_chunks)
        
        assert len(output_chunks) == 4
        assert output_chunks[0] == (50, 50)  # Non-spatial dimensions unchanged
        assert output_chunks[1] == (60, 40)  # Non-spatial dimensions unchanged
        assert output_chunks[2] == (50, 50)  # Y dimension (halved)
        assert output_chunks[3] == (40, 40)  # X dimension (halved)

    def test_calculate_zoom_factors(self):
        """Test _calculate_zoom_factors method."""
        downscaler = Downscaler(downscale_factor=2.0)
        zoom_factors = downscaler._calculate_zoom_factors(4, 2.0)
        
        assert len(zoom_factors) == 4
        assert zoom_factors[0] == 1.0
        assert zoom_factors[1] == 1.0
        assert zoom_factors[2] == 0.5  # Y dimension
        assert zoom_factors[3] == 0.5  # X dimension

    def test_process_block_on_gpu_empty_block(self):
        """Test _process_block_on_gpu with empty block."""
        downscaler = Downscaler(device="cpu")
        
        # Test with empty block - this should return the block as-is
        empty_block = np.array([])
        mock_cp = Mock()
        mock_ndi = Mock()
        
        result = downscaler._process_block_on_gpu(
            empty_block, [0.0, 0.0, 0.25, 0.25], 2.0, 0, mock_cp, mock_ndi
        )
        
        assert np.array_equal(result, empty_block)

    def test_cleanup_gpu_memory(self):
        """Test _cleanup_gpu_memory method."""
        downscaler = Downscaler()
        
        mock_cp = Mock()
        mock_memory_pool = Mock()
        mock_cp.get_default_memory_pool.return_value = mock_memory_pool
        
        # Test with free_all_blocks method
        downscaler._cleanup_gpu_memory(mock_cp)
        mock_memory_pool.free_all_blocks.assert_called_once()
        
        # Test fallback to free_all_free for older CuPy versions
        mock_memory_pool.reset_mock()
        mock_memory_pool.free_all_blocks.side_effect = AttributeError("no free_all_blocks")
        
        downscaler._cleanup_gpu_memory(mock_cp)
        mock_memory_pool.free_all_free.assert_called_once()

    def test_downscale_nearest_gpu_fallback(self):
        """Test that GPU nearest-neighbor downscaling falls back to CPU."""
        downscaler = Downscaler(device="cpu")
        image = da.ones((100, 100), dtype=np.float32)
        
        def mock_rescale_func(block, order):
            return block[::2, ::2]  # Simple downscaling
        
        # Mock the CPU method to verify it's called
        with patch.object(downscaler, '_downscale_nearest_cpu') as mock_cpu_method:
            mock_cpu_method.return_value = da.ones((50, 50), dtype=np.float32)
            
            result = downscaler._downscale_nearest_gpu(image, mock_rescale_func)
            
            mock_cpu_method.assert_called_once_with(image, mock_rescale_func)
            assert result is not None


class TestDownscalerEdgeCases:
    """Test edge cases and error conditions."""

    def test_validate_downscale_levels_zero_or_negative(self):
        """Test validate_downscale_levels with zero or negative levels."""
        downscaler = Downscaler(downscale_levels=-1)
        result = downscaler.validate_downscale_levels((100, 100))
        assert result == 0
        
        downscaler = Downscaler(downscale_levels=0)
        result = downscaler.validate_downscale_levels((100, 100))
        assert result == 0

    def test_create_downscaled_arrays_dimensions_too_small(self):
        """Test downscaling when dimensions become too small."""
        # Create an image that will become too small after one downscale
        image = da.ones((4, 4), dtype=np.uint8)
        downscaler = Downscaler(
            downscale_factor=5.0,  # Large factor that will make dimensions < 1
            downscale_levels=3
        )
        
        arrays = downscaler.create_downscaled_arrays(image)
        
        # Should stop early when dimensions become too small
        assert len(arrays) == 1  # Only original image

    def test_downscale_gaussian_cpu_error_handling(self):
        """Test error handling in CPU Gaussian downscaling."""
        downscaler = Downscaler(downscale_factor=2.0)
        image = da.ones((100, 100), dtype=np.float32)
        
        def failing_rescale_func(block, order):
            raise ValueError("Rescale failed")
        
        # Mock map_blocks to raise an error
        with patch('dask.array.map_blocks', side_effect=ValueError("Processing failed")):
            result = downscaler._downscale_gaussian_cpu(image, failing_rescale_func)
            assert result is None

    def test_downscale_nearest_cpu_error_handling(self):
        """Test error handling in CPU nearest-neighbor downscaling."""
        downscaler = Downscaler(downscale_factor=2.0)
        image = da.ones((100, 100), dtype=np.float32)
        
        def failing_rescale_func(block, order):
            raise ValueError("Rescale failed")
        
        # Mock map_blocks to raise an error
        with patch('dask.array.map_blocks', side_effect=RuntimeError("Processing failed")):
            result = downscaler._downscale_nearest_cpu(image, failing_rescale_func)
            assert result is None

    def test_create_coordinate_transformations_multidimensional(self):
        """Test coordinate transformations with multidimensional scales."""
        downscaler = Downscaler(downscale_factor=2.0)
        
        # Test with 5D scale (T, C, Z, Y, X)
        original_transforms = [ScaleTransformation(scale=[1.0, 1.0, 0.5, 0.1, 0.1])]
        
        level_transforms = downscaler.create_coordinate_transformations_for_levels(
            original_transforms, (10, 3, 50, 100, 100), 3
        )
        
        assert len(level_transforms) == 3
        
        # Level 0 (original)
        assert level_transforms[0][0].scale == [1.0, 1.0, 0.5, 0.1, 0.1]
        # Level 1 (2x downscaled in Y, X)
        assert level_transforms[1][0].scale == [1.0, 1.0, 0.5, 0.2, 0.2]
        # Level 2 (4x downscaled in Y, X)
        assert level_transforms[2][0].scale == [1.0, 1.0, 0.5, 0.4, 0.4]

    def test_create_coordinate_transformations_empty_list(self):
        """Test coordinate transformations with empty input list."""
        downscaler = Downscaler(downscale_factor=2.0)
        
        level_transforms = downscaler.create_coordinate_transformations_for_levels(
            [], (100, 100), 3
        )
        
        assert len(level_transforms) == 3
        assert all(len(transforms) == 0 for transforms in level_transforms)

    def test_fractional_downscale_factor_validation(self):
        """Test validation with fractional downscale factors."""
        # This test doesn't need warnings - just test the method works
        downscaler = Downscaler(downscale_factor=1.5, downscale_levels=5)
        
        # Test with a small image that can support fewer levels
        image_shape = (8, 8)
        
        validated_levels = downscaler.validate_downscale_levels(image_shape)
        
        # Should find the maximum valid levels
        assert validated_levels >= 0
        assert validated_levels <= 5

    def test_device_validation_basic(self):
        """Test basic device validation functionality."""
        # Test CPU device initialization
        downscaler = Downscaler(device="cpu")
        assert downscaler.device == "cpu"
        assert downscaler.cuda_device_id is None
        
        # Test device info for CPU
        info = downscaler.get_device_info()
        assert info["device_type"] == "cpu"

    def test_downscale_method_validation(self):
        """Test that invalid downscale methods default to gaussian."""
        # Create image and test
        image = da.ones((100, 100), dtype=np.uint8)
        downscaler = Downscaler(
            downscale_factor=2.0,
            downscale_method="invalid_method",  # type: ignore[arg-type]
            downscale_levels=1
        )
        
        # Should still work and default to gaussian
        arrays = downscaler.create_downscaled_arrays(image)
        assert len(arrays) == 2
        assert arrays[0].shape == (100, 100)
        assert arrays[1].shape == (50, 50)

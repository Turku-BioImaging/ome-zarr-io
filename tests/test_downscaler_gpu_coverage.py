"""
This module tests the advanced functionality of the Downscaler class that wasn't
covered in the basic downscaling tests, including device validation, GPU operations,
and error handling.
"""

import numpy as np
import dask.array as da
import pytest
from unittest.mock import Mock, patch
from ome_zarr_writer.downscaler import Downscaler
from ome_zarr_writer.schema_models import ScaleTransformation


@pytest.mark.gpu
class TestDownscalerDeviceValidation:
    """
    This is the documentation for TestDownscalerDeviceValidation class,
    used for device validation and GPU-related functionality.
    """

    def test_init_with_cpu_device(self):
        downscaler = Downscaler(device="cpu")
        assert downscaler.device == "cpu"
        assert downscaler.cuda_device_id is None

    def test_use_gpu_cpu_device(self):
        """Test _use_gpu method with CPU device."""
        downscaler = Downscaler(device="cpu")
        assert not downscaler._use_gpu()


@pytest.mark.gpu
class TestDownscalerDeviceInfo:
    """Test device information methods."""

    def test_get_device_info__if_cpu(self):
        downscaler = Downscaler(device="cpu")
        info = downscaler.get_device_info()
        
        assert info["device_type"] == "cpu"
        assert info["device_name"] == "CPU"
        assert info["device_id"] is None

    def test_list_cuda_devices__if_no_cupy(self):
        with patch('builtins.__import__', side_effect=ImportError("No module named 'cupy'")):
            devices = Downscaler.list_cuda_devices()
            
            assert len(devices) == 1
            assert "error" in devices[0]
            assert "Failed to list CUDA devices" in devices[0]["error"]


@pytest.mark.gpu
class TestDownscalerGPUMethods:
    """Test GPU-specific downscaling methods."""
    
    @pytest.fixture
    def create_2d_sample_image(self):
        return da.ones((100, 100), dtype=np.float32)

    def test_gpu_gaus_downscale__if_no_cupy(self, create_2d_sample_image):
        downscaler = Downscaler(device="cpu")  # Even CPU can call GPU methods

        # Mock the import to fail
        with patch('builtins.__import__', side_effect=ImportError("No module named 'cupy'")):
            result = downscaler._downscale_gaussian_gpu(create_2d_sample_image)
            assert result is None

    def test_calculate_gaussian_sigma(self):
        downscaler = Downscaler(downscale_factor=2.0)
        sigma = downscaler._calculate_gaussian_sigma(4)
        
        assert len(sigma) == 4
        assert sigma[0] == 0.0
        assert sigma[1] == 0.0
        assert sigma[2] == 0.25  # (2.0 - 1) / 4.0
        assert sigma[3] == 0.25

    def test_calculate_output_chunk__xy_only(self):
        downscaler = Downscaler(downscale_factor=2.0)
        input_chunks = ((50, 50), (60, 40), (100, 100), (80, 80))
        
        output_chunks = downscaler._calculate_output_chunks(input_chunks)
        
        assert len(output_chunks) == 4
        assert output_chunks[0] == (50, 50)  
        assert output_chunks[1] == (60, 40)  
        assert output_chunks[2] == (50, 50)  
        assert output_chunks[3] == (40, 40)  

    def test_calculate_zoom_factors__xy_only(self):
        downscaler = Downscaler(downscale_factor=2.0)
        zoom_factors = downscaler._calculate_zoom_factors(4, 2.0)
        
        assert len(zoom_factors) == 4
        assert zoom_factors[0] == 1.0
        assert zoom_factors[1] == 1.0
        assert zoom_factors[2] == 0.5  
        assert zoom_factors[3] == 0.5  

    def test_process_block_on_gpu__with_empty_block(self):
        downscaler = Downscaler(device="cpu")
        
        empty_block = np.array([])
        mock_cp = Mock()
        mock_ndi = Mock()
        
        result = downscaler._process_block_on_gpu(
            empty_block, [0.0, 0.0, 0.25, 0.25], 2.0, 0, mock_cp, mock_ndi
        )
        
        assert np.array_equal(result, empty_block)

    def test_cleanup_gpu_memory(self):
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

    def test_downscale_nearest_gpu__import_err_no_cupy_fallback(self, create_2d_sample_image):
        downscaler = Downscaler(device="cuda")
        
        def mock_rescale_func(block, order):
            return block[::2, ::2]  
        
        with patch('builtins.__import__', side_effect=ImportError("No module named 'cupy'")):
            result = downscaler._downscale_nearest_gpu(create_2d_sample_image, mock_rescale_func)
            assert result is None

    def test_downscale_nearest_gpu__completion_to_dask_arr(self, create_2d_sample_image):
        downscaler = Downscaler(device="cuda", downscale_factor=2.0)        
        result = downscaler._downscale_nearest_gpu(create_2d_sample_image, None)
        
        if result is not None: 
            assert result.shape == (32, 32)
            assert result.dtype == np.float32
            assert hasattr(result, 'compute')


@pytest.mark.gpu
class TestDownscalerEdgeCases:
    """Test edge cases and error conditions."""

    def test_validate_downscale_levels__zero_or_neg_levels(self):
        downscaler = Downscaler(downscale_levels=-1)
        result = downscaler.validate_downscale_levels((100, 100))
        assert result == 0
        
        downscaler = Downscaler(downscale_levels=0)
        result = downscaler.validate_downscale_levels((100, 100))
        assert result == 0

    def test_create_downscaled_arrays_for_multiscale__if_img_dimensions_small(self):
        image = da.ones((4, 4), dtype=np.uint8)
        downscaler = Downscaler(
            downscale_factor=5.0,  
            downscale_levels=3
        )
        
        arrays = downscaler.create_downscaled_arrays_for_multiscale(image)
        
        assert len(arrays) == 1  

    def test_downscale_gaussian_cpu__error_handling(self, create_2d_sample_image):
        downscaler = Downscaler(downscale_factor=2.0)
        
        def failing_rescale_func(block, order):
            raise ValueError("Rescale failed")
        
        with patch('dask.array.map_blocks', side_effect=ValueError("Processing failed")):
            result = downscaler._downscale_gaussian_cpu(create_2d_sample_image, failing_rescale_func)
            assert result is None

    def test_downscale_nearest_cpu__error_handling(self, create_2d_sample_image):
        downscaler = Downscaler(downscale_factor=2.0)
        
        def failing_rescale_func(block, order):
            raise ValueError("Rescale failed")
        
        with patch('dask.array.map_blocks', side_effect=RuntimeError("Processing failed")):
            result = downscaler._downscale_nearest_cpu(create_2d_sample_image, failing_rescale_func)
            assert result is None

    def test_create_coordinate_transformations_for_multiscales(self):
        downscaler = Downscaler(downscale_factor=2.0)
        
        original_transforms = [ScaleTransformation(scale=[1.0, 1.0, 0.5, 0.1, 0.1])]
        
        level_transforms = downscaler.create_coordinate_transformations_for_multiscales(
            original_transforms, (10, 3, 50, 100, 100), 3
        )
        
        assert len(level_transforms) == 3
        
        assert level_transforms[0][0].scale == [1.0, 1.0, 0.5, 0.1, 0.1]
        assert level_transforms[1][0].scale == [1.0, 1.0, 0.5, 0.2, 0.2]
        assert level_transforms[2][0].scale == [1.0, 1.0, 0.5, 0.4, 0.4]

    def test_create_coordinate_transformations_for_multiscales__empty_list(self):
        downscaler = Downscaler(downscale_factor=2.0)
        
        level_transforms = downscaler.create_coordinate_transformations_for_multiscales(
            [], (100, 100), 3
        )
        
        assert len(level_transforms) == 3
        assert all(len(transforms) == 0 for transforms in level_transforms)

    def test_validate_downscale_levels__fractional_downscale_factors(self):
        downscaler = Downscaler(downscale_factor=1.5, downscale_levels=5)
        image_shape = (8, 8)
        
        validated_levels = downscaler.validate_downscale_levels(image_shape)
        
        assert validated_levels >= 0
        assert validated_levels <= 5

    # Rediundant test, as the device validation is already covered above
    # def test_device_validation_basic(self):
    #     """Test basic device validation functionality."""
    #     # Test CPU device initialization
    #     downscaler = Downscaler(device="cpu")
    #     assert downscaler.device == "cpu"
    #     assert downscaler.cuda_device_id is None

    #     # Test device info for CPU
    #     info = downscaler.get_device_info()
    #     assert info["device_type"] == "cpu"

    def test_downscale_method_validation__invalid_defaults_to_gaus(self, create_2d_sample_image):
        downscaler = Downscaler(
            downscale_factor=2.0,
            downscale_method="invalid_method",  
            downscale_levels=1
        )
        
        arrays = downscaler.create_downscaled_arrays_for_multiscale(create_2d_sample_image)
        assert len(arrays) == 2
        assert arrays[0].shape == (100, 100)
        assert arrays[1].shape == (50, 50)

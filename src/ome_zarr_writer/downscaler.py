"""Downscaling functionality for OME-Zarr multiscale pyramids."""

from typing import List, Literal, Optional
import numpy as np
import dask.array as da
import dask_image.ndfilters
from skimage.transform import rescale
from .schema_models import ScaleTransformation


class Downscaler:
    """
    This Downscaler class handles operations for:
    - Creating downscaled versions of image arrays using interpolation methods.
    - Managing coordinate transformations for each resolution level.
    """

    def __init__(
        self,
        downscale_factor: float = 2.0,
        downscale_method: Literal["gaussian", "nearest"] = "gaussian",
        downscale_levels: Optional[int] = None,
        device: Literal["cpu", "cuda"] = "cpu",
        cuda_device_id: Optional[int] = None,
    ):
        """Initialize the downscaler.

        Args:
            downscale_factor: Factor by which to downscale each level. Default is 2.0.
            downscale_method: Method used for downscaling. The options are:
                - "gaussian" (default): applies a Gaussian blur before resizing.
                - "nearest": uses nearest-neighbor downsampling.  
            downscale_levels: Number of downscaled levels to create. If None, no downscaled levels are generated beyong the original.
            device: Device to use for computation. The options are:
                - "cpu" (default): uses CPU for processing.
                - "cuda": uses GPU with CUDA support for processing.
            cuda_device_id: ID of the specific CUDA device to use. If None, uses default device.
        """
        if downscale_factor <= 1.0:
            raise ValueError(f"downscale_factor must be > 1.0, got {downscale_factor}")

        self.downscale_factor = downscale_factor
        self.downscale_method = downscale_method
        self.downscale_levels = downscale_levels
        self.device = device
        self.cuda_device_id = cuda_device_id

        self._validate_requested_device_availability()

    def _validate_requested_device_availability(self) -> None:
        if self.device == "cuda":
            try:
                import cupy as cp
                num_devices = cp.cuda.runtime.getDeviceCount()
                
                if self.cuda_device_id is not None:
                    if self.cuda_device_id < 0 or self.cuda_device_id >= num_devices:
                        raise ValueError(
                            f"Invalid CUDA device ID {self.cuda_device_id}. "
                            f"Available devices: 0-{num_devices-1}"
                        )
                    with cp.cuda.Device(self.cuda_device_id):
                        cp.cuda.runtime.getDeviceProperties(self.cuda_device_id)
                else:
                    self.cuda_device_id = 0
                    
            except (ImportError, Exception) as e:
                import warnings
                warnings.warn(
                    f"CUDA device requested but not available: {e}. "
                    "Falling back to CPU computation.",
                    UserWarning,
                )
                self.device = "cpu"
                self.cuda_device_id = None

    def _use_gpu(self) -> bool:
        return self.device == "cuda"

    def _device_info(self) -> dict:
        if not self._should_use_gpu():
            return {
                "device_type": "cpu",
                "device_name": "CPU",
                "device_id": None,
            }
        
        try:
            import cupy as cp
            with cp.cuda.Device(self.cuda_device_id):
                props = cp.cuda.runtime.getDeviceProperties(self.cuda_device_id)
                meminfo = cp.cuda.runtime.memGetInfo()
                free_mem = meminfo[0] / 1024**3
                total_mem = meminfo[1] / 1024**3
                used_mem = total_mem - free_mem
                
                return {
                    "device_type": "cuda",
                    "device_id": self.cuda_device_id,
                    "device_name": props['name'].decode(),
                    "compute_capability": f"{props['major']}.{props['minor']}",
                    "total_memory_gb": round(total_mem, 2),
                    "used_memory_gb": round(used_mem, 2),
                    "free_memory_gb": round(free_mem, 2),
                    "memory_usage_percent": round(used_mem/total_mem*100, 1),
                    "multiprocessors": props['multiProcessorCount'],
                }
        except Exception as e:
            return {
                "device_type": "cuda",
                "device_id": self.cuda_device_id,
                "error": f"Failed to get device info: {e}"
            }

    @staticmethod
    def _cuda_devices_available() -> List[dict]:
        try:
            import cupy as cp
            num_devices = cp.cuda.runtime.getDeviceCount()
            devices = []
            
            for i in range(num_devices):
                with cp.cuda.Device(i):
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    meminfo = cp.cuda.runtime.memGetInfo()
                    free_mem = meminfo[0] / 1024**3
                    total_mem = meminfo[1] / 1024**3
                    used_mem = total_mem - free_mem
                    
                    devices.append({
                        "device_id": i,
                        "device_name": props['name'].decode(),
                        "compute_capability": f"{props['major']}.{props['minor']}",
                        "total_memory_gb": round(total_mem, 2),
                        "used_memory_gb": round(used_mem, 2),
                        "free_memory_gb": round(free_mem, 2),
                        "memory_usage_percent": round(used_mem/total_mem*100, 1),
                        "multiprocessors": props['multiProcessorCount'],
                    })
            
            return devices
            
        except Exception as e:
            return [{"error": f"Failed to list CUDA devices: {e}"}]

    def validate_or_adjust_downscale_levels(self, image_shape: tuple) -> int:
        """"
        Args:
            image_shape: Shape of the input image array.
        Returns:
            Validated number of downscale levels.
        """
        if self.downscale_levels is None or self.downscale_levels <= 0:
            return 0

        min_spatial_dim = min(image_shape[-2], image_shape[-1])

        max_levels = 0
        test_size = min_spatial_dim
        while test_size >= self.downscale_factor:
            test_size = test_size / self.downscale_factor
            max_levels += 1

        if self.downscale_levels > max_levels:
            import warnings

            suggested_levels = max_levels
            suggested_factor = self.downscale_factor

            for factor in [1.5, 1.25, 1.1]:
                if factor >= self.downscale_factor:
                    continue
                test_levels = 0
                test_size = min_spatial_dim
                while test_size >= factor:
                    test_size = test_size / factor
                    test_levels += 1
                if test_levels >= self.downscale_levels:
                    suggested_factor = factor
                    break

            warning_msg = (
                f"Requested {self.downscale_levels} downscale levels with factor {self.downscale_factor} "
                f"is too many for image with spatial dimensions {image_shape[-2:]}. "
                f"Maximum possible levels: {max_levels}. "
                f"Suggestions: reduce downscale_levels to {suggested_levels} "
                f"or reduce downscale_factor to {suggested_factor}."
            )
            warnings.warn(warning_msg, UserWarning)

            return max_levels

        return self.downscale_levels
        

    def create_downscaled_arrays_for_multiscale(self, image: da.Array) -> List[da.Array]:
        """
        Creates a list of dask arrays where each subsequent array is downscaled
        by the specified downscale_factor in the last two dimensions (Y and X axes),
        while preserving all other dimensions (time, channel, Z). Uses Gaussian
        filtering before downscaling to prevent aliasing artifacts.

        Args:
            image: The input dask array to downscale.

        Returns:
            List of downscaled dask arrays. The first array is the original image,
            followed by progressively downscaled versions.
        """
        validated_levels = self.validate_or_adjust_downscale_levels(image.shape)

        if validated_levels <= 0:
            return [image]

        arrays = [image]  
        current_array = image

        def rescale_yx_block_nearest_neighbor_gpu(block, order: int):
            """
            Args:
                block: The dask array block to rescale.
                order: the order of interpolation.
                    0 = nearest-neighbor,1 = bilinear.
                    See skimage.transform.warp documentation for details.
            """
            scale_factors = [1.0] * block.ndim
            scale_factor = 1.0 / self.downscale_factor
            scale_factors[-2] = scale_factor  
            scale_factors[-1] = scale_factor  

            return rescale(
                block,
                scale=scale_factors,
                order=order,
                preserve_range=True,
                anti_aliasing=False, 
                channel_axis=None,
            ).astype(block.dtype)

        for _ in range(1, validated_levels + 1):
            if (
                current_array.shape[-2] / self.downscale_factor < 1
                or current_array.shape[-1] / self.downscale_factor < 1
            ):
                break

            method = self.downscale_method
            if method not in {"gaussian", "nearest"}:
                method = "gaussian"

            if method == "gaussian":
                current_array = self._downscale_gaussian(
                    current_array, rescale_yx_block_nearest_neighbor_gpu
                )
            elif method == "nearest":
                current_array = self._downscale_nearest(current_array, rescale_yx_block_nearest_neighbor_gpu)

            if current_array is None:
                break

            arrays.append(current_array)

        return arrays

    def _downscale_gaussian(
        self, current_array: da.Array, rescale_func
    ) -> Optional[da.Array]:
        """Apply Gaussian filtering followed by downscaling.

        Args:
            current_array: Current array to downscale.
            rescale_func: Function to use for rescaling blocks.

        Returns:
            Downscaled array or None if operation failed.
        """
        if self._should_use_gpu():
            return self._downscale_gaussian_gpu(current_array)
        else:
            return self._downscale_gaussian_cpu(current_array, rescale_func)

    def _downscale_gaussian_cpu(
        self, current_array: da.Array, rescale_func
    ) -> Optional[da.Array]:
        """Apply Gaussian filtering followed by downscaling on CPU.

        Args:
            current_array: Current array to downscale.
            rescale_func: Function to use for rescaling blocks.

        Returns:
            Downscaled array or None if operation failed.
        """
        # Apply Gaussian filter to prevent aliasing
        # Sigma is proportional to the downscale factor
        # For factor=2, sigma=0.5; for factor=4, sigma=1.0, etc.
        sigma = [0.0] * current_array.ndim
        sigma[-2] = (self.downscale_factor - 1) / 4.0  # Y dimension
        sigma[-1] = (self.downscale_factor - 1) / 4.0  # X dimension

        filtered_array = dask_image.ndfilters.gaussian_filter(
            current_array, sigma=sigma, mode="nearest"
        )

        # Calculate new chunk sizes (also downscaled for Y and X dimensions)
        new_chunks = list(filtered_array.chunks)
        new_chunks[-2] = tuple(
            max(1, int(chunk_size / self.downscale_factor))
            for chunk_size in new_chunks[-2]
        )
        new_chunks[-1] = tuple(
            max(1, int(chunk_size / self.downscale_factor))
            for chunk_size in new_chunks[-1]
        )

        try:
            return da.map_blocks(
                rescale_func,
                filtered_array,
                dtype=filtered_array.dtype,
                chunks=new_chunks,
                drop_axis=None,
                new_axis=None,
                meta=np.array([], dtype=filtered_array.dtype),
                order=1,  # use bicubic interpolation on Gaussian-filtered data
            )
        except (ValueError, RuntimeError):
            return None

    def _downscale_gaussian_gpu(self, current_array: da.Array) -> Optional[da.Array]:
        """Apply Gaussian filtering followed by downscaling on GPU.
        
        Args:
            current_array: Current array to downscale.
            
        Returns:
            Downscaled array or None if operation failed.
        """
        try:
            import cupy as cp
            import cupyx.scipy.ndimage as ndi
        except ImportError:
            return None
            
        # Calculate Gaussian filter sigma for anti-aliasing
        sigma = self._calculate_gaussian_sigma(current_array.ndim)
        
        # Optimize array chunking for GPU processing
        current_array = current_array.rechunk("100MB")
        
        # Calculate output chunk sizes
        new_chunks = self._calculate_output_chunks(current_array.chunks)

        def filter_and_rescale(block, sigma_list, downscale_factor, cuda_device_id):
            """Apply Gaussian filtering and rescaling in a single GPU operation."""
            return self._process_block_on_gpu(
                block, sigma_list, downscale_factor, cuda_device_id, cp, ndi
            )

        return da.map_blocks(
            filter_and_rescale,
            current_array,
            sigma_list=sigma,
            downscale_factor=self.downscale_factor,
            cuda_device_id=self.cuda_device_id,
            dtype=current_array.dtype,
            chunks=new_chunks,
            drop_axis=None,
            new_axis=None,
            meta=np.array([], dtype=current_array.dtype),
        )

    def _calculate_gaussian_sigma(self, ndim: int) -> List[float]:
        """Calculate Gaussian filter sigma values for anti-aliasing.
        
        Args:
            ndim: Number of dimensions in the array.
            
        Returns:
            List of sigma values, with non-zero values only for Y and X dimensions.
        """
        sigma = [0.0] * ndim
        sigma_value = (self.downscale_factor - 1) / 4.0
        sigma[-2] = sigma_value  # Y dimension
        sigma[-1] = sigma_value  # X dimension
        return sigma

    def _calculate_output_chunks(self, input_chunks: tuple) -> List[tuple]:
        """Calculate output chunk sizes after downscaling.
        
        Args:
            input_chunks: Input array chunk sizes.
            
        Returns:
            List of output chunk sizes.
        """
        new_chunks = list(input_chunks)
        # Downscale Y and X dimensions (last two)
        for dim_idx in [-2, -1]:
            new_chunks[dim_idx] = tuple(
                max(1, int(chunk_size / self.downscale_factor))
                for chunk_size in new_chunks[dim_idx]
            )
        return new_chunks

    def _process_block_on_gpu(self, block, sigma_list, downscale_factor, cuda_device_id, cp, ndi):
        """
        Args:
            block: Input block to process.
            sigma_list: Gaussian filter sigma values.
            downscale_factor: Factor by which to downscale.
            cuda_device_id: CUDA device ID to use.
            cp: CuPy module.
            ndi: CuPy scipy ndimage module.
            
        Returns:
            Processed block as NumPy array.
        """
        if block.size == 0:
            return block
        
        with cp.cuda.Device(cuda_device_id):
            gpu_block = cp.asarray(block)
            
            try:
                # Gaussian filter
                filtered_gpu = ndi.gaussian_filter(
                    gpu_block, sigma=sigma_list, mode="nearest"
                )
                
                # Calculate zoom factors for rescaling
                zoom_factors = self._calculate_zoom_factors_for_rescaling(block.ndim, downscale_factor)
                
                # rescaling with bilinear interpolation
                rescaled_gpu = ndi.zoom(
                    filtered_gpu, zoom=zoom_factors, order=1, prefilter=False
                )
                
                # Convert back to NumPy with original dtype
                result = cp.asnumpy(rescaled_gpu).astype(block.dtype)
                
                # cleanup
                del gpu_block, filtered_gpu, rescaled_gpu
                
                return result
                
            except Exception as e:
                self._cleanup_gpu_memory(cp)
                raise RuntimeError(
                    f"GPU operation failed on device {cuda_device_id}: {e}"
                ) from e

    def _calculate_zoom_factors_for_rescaling(self, ndim: int, downscale_factor: float) -> List[float]:
        """
        Args:
            ndim: Number of dimensions in the block.
            downscale_factor: Factor by which to downscale the xy dimensions.
            
        Returns:
            List of zoom factors for each dimension.
        """
        zoom_factors = [1.0] * ndim
        zoom_factor = 1.0 / downscale_factor
        zoom_factors[-2] = zoom_factor  
        zoom_factors[-1] = zoom_factor  
        return zoom_factors

    def _cleanup_gpu_memory(self, cp):
        """       
        Args:
            cp: CuPy module.
        """
        import gc
        gc.collect()
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except AttributeError:
            # Fallback for older CuPy versions
            cp.get_default_memory_pool().free_all_free()


    def _downscale_nearest(
        self, current_array: da.Array, rescale_func
    ) -> Optional[da.Array]:
        """
        Args:
            current_array: Current array to downscale.
            rescale_func: Function to use for rescaling blocks.
        Returns:
            Downscaled array or None if operation failed.
        """
        if self._should_use_gpu():
            return self._downscale_nearest_neighbor_gpu(current_array, rescale_func)
        else:
            return self._downscale_nearest_neighbor_cpu(current_array, rescale_func)

    def _downscale_nearest_neighbor_cpu(
        self, current_array: da.Array, rescale_func
    ) -> Optional[da.Array]:
        """
        Args:
            current_array: Current array to downscale.
            rescale_func: Function to use for rescaling blocks.
        Returns:
            Downscaled array or None if operation failed.
        """
        new_chunks = list(current_array.chunks)
        new_chunks[-2] = tuple(
            max(1, int(chunk_size / self.downscale_factor))
            for chunk_size in new_chunks[-2]
        )
        new_chunks[-1] = tuple(
            max(1, int(chunk_size / self.downscale_factor))
            for chunk_size in new_chunks[-1]
        )

        try:
            return da.map_blocks(
                rescale_func,
                current_array,
                dtype=current_array.dtype,
                chunks=tuple(new_chunks),
                drop_axis=None,
                new_axis=None,
                meta=np.array([], dtype=current_array.dtype),
                order=0,  # nearest-neighbor interpolation
            )
        except (ValueError, RuntimeError):
            return None

    def _downscale_nearest_neighbor_gpu(
        self, current_array: da.Array, rescale_func
    ) -> Optional[da.Array]:
        """
        Args:
            current_array: Current array to downscale.
            rescale_func: Function to use for rescaling blocks.
        Returns:
            Downscaled array or None if operation failed.
        """
        try:
            import cupy as cp
            import cupyx.scipy.ndimage as ndi
        except ImportError:
            return None
        
        def rescale_yx_block_nearest_neighbor_gpu(block, cuda_device_id):
            if block.size == 0:
                return block
                
            with cp.cuda.Device(cuda_device_id):
                try:
                    array_cp = cp.asarray(block)
                    # Calculate zoom factors 
                    zoom_factors = [1.0] * array_cp.ndim
                    zoom_factor = 1.0 / self.downscale_factor
                    zoom_factors[-2] = zoom_factor  
                    zoom_factors[-1] = zoom_factor 
                    
                    rescaled_gpu = ndi.zoom(array_cp, zoom_factors, order=0, prefilter=False)
                    result = cp.asnumpy(rescaled_gpu).astype(block.dtype)
                    
                    # Cleanup
                    del array_cp, rescaled_gpu
                    
                    return result
                    
                except Exception as e:
                    self._cleanup_gpu_memory(cp)
                    raise RuntimeError(
                        f"GPU nearest-neighbor operation failed on device {cuda_device_id}: {e}"
                    ) from e

        
        current_array = current_array.rechunk("100MB")
        new_chunks = self._calculate_output_chunks(current_array.chunks)

        return da.map_blocks(
            rescale_yx_block_nearest_neighbor_gpu,
            current_array,
            cuda_device_id=self.cuda_device_id,
            dtype=current_array.dtype,
            chunks=new_chunks,
            drop_axis=None,
            new_axis=None,
            meta=np.array([], dtype=current_array.dtype),
        )


    def create_coordinate_transformations_for_multiscales(
        self,
        coordinate_transformations: Optional[List[ScaleTransformation]],
        image_shape: tuple,
        num_levels: int,
    ) -> List[List[ScaleTransformation]]:
        """
        Scale transformations provided for the original image are adjusted for each downscale level
        by multiplying the spatial dimensions by the downscale factor for each level.

        Args:
            coordinate_transformations: Original coordinate transformations.
            image_shape: Shape of the original image.
            num_levels: Number of resolution levels created.

        Returns:
            List of scale transformation lists, one for each downscaled level.
            The first list corresponds to the original image, subsequent lists
            correspond to progressively downscaled levels.
        """
        if coordinate_transformations is None:
            return []

        all_transformations = []

        for level in range(num_levels):
            level_transformations = []

            for transform in coordinate_transformations:
                new_scale = transform.scale.copy()

                scale_factor = self.downscale_factor**level
                new_scale[-2] *= scale_factor 
                new_scale[-1] *= scale_factor 

                level_transformations.append(ScaleTransformation(scale=new_scale))

            all_transformations.append(level_transformations)

        return all_transformations

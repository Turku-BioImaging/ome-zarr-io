"""Downscaling functionality for OME-Zarr multiscale pyramids."""

from typing import List, Literal, Optional
import numpy as np
import dask.array as da
import dask_image.ndfilters
from skimage.transform import rescale
from .schema_models import ScaleTransformation


class Downscaler:
    """Handles downscaling operations for creating multiscale image pyramids.

    This class is responsible for creating downscaled versions of image arrays
    using different interpolation methods (Gaussian filtering or nearest-neighbor)
    and managing coordinate transformations for each resolution level.
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
            downscale_factor: Factor by which to downscale each level (default: 2.0).
            downscale_method: Method to use for downscaling ("gaussian" or "nearest").
            downscale_levels: Number of downscale levels to create. If None, no downscaling.
            device: Device to use for computation ("cpu" or "cuda"). Default: "cpu".
            cuda_device_id: Specific CUDA device ID to use. If None, uses default device.
        """
        if downscale_factor <= 1.0:
            raise ValueError(f"downscale_factor must be > 1.0, got {downscale_factor}")

        self.downscale_factor = downscale_factor
        self.downscale_method = downscale_method
        self.downscale_levels = downscale_levels
        self.device = device
        self.cuda_device_id = cuda_device_id

        # Validate device availability
        self._validate_device()

    def _validate_device(self) -> None:
        """Validate that the requested device is available.

        Raises:
            ValueError: If CUDA is requested but not available.
        """
        if self.device == "cuda":
            try:
                import cupy as cp
                # Test if CUDA is actually available
                num_devices = cp.cuda.runtime.getDeviceCount()
                
                if self.cuda_device_id is not None:
                    if self.cuda_device_id < 0 or self.cuda_device_id >= num_devices:
                        raise ValueError(
                            f"Invalid CUDA device ID {self.cuda_device_id}. "
                            f"Available devices: 0-{num_devices-1}"
                        )
                    # Test if the specific device is accessible
                    with cp.cuda.Device(self.cuda_device_id):
                        cp.cuda.runtime.getDeviceProperties(self.cuda_device_id)
                else:
                    # Use default device (0)
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

    def _should_use_gpu(self) -> bool:
        """Check if GPU should be used for computation.

        Returns:
            True if GPU should be used, False otherwise.
        """
        return self.device == "cuda"

    def get_device_info(self) -> dict:
        """Get information about the current computing device.
        
        Returns:
            Dictionary containing device information.
        """
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
    def list_cuda_devices() -> List[dict]:
        """List all available CUDA devices.
        
        Returns:
            List of dictionaries containing information about each CUDA device.
        """
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

    def validate_downscale_levels(self, image_shape: tuple) -> int:
        """Validate and adjust downscale levels based on image dimensions.

        Args:
            image_shape: Shape of the input image array.

        Returns:
            Validated number of downscale levels.
        """
        if self.downscale_levels is None or self.downscale_levels <= 0:
            return 0

        # Get the size of the smallest spatial dimension (Y and X are last two)
        min_spatial_dim = min(image_shape[-2], image_shape[-1])

        # Calculate maximum possible levels
        max_levels = 0
        test_size = min_spatial_dim
        while test_size >= self.downscale_factor:
            test_size = test_size / self.downscale_factor
            max_levels += 1

        if self.downscale_levels > max_levels:
            import warnings

            suggested_levels = max_levels
            suggested_factor = self.downscale_factor

            # Try to find a smaller factor that would work
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

    def create_downscaled_arrays(self, image: da.Array) -> List[da.Array]:
        """Create downscaled arrays for multiscale representation.

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
        # Validate downscale levels
        validated_levels = self.validate_downscale_levels(image.shape)

        if validated_levels <= 0:
            return [image]

        arrays = [image]  # Level 0: original resolution
        current_array = image

        # Create a rescaling function that only operates on Y and X dimensions
        def rescale_yx_block(block, order: int):
            """
            Rescale only the last two dimensions of a block by the downscale factor.
            Parameter `order` is the order of interpolation. 0 = nearest-neighbor,
            1 = bilinear. See skimage.transform.warp documentation for details.
            """
            # Create scale factors: 1 for all dimensions except last two
            scale_factors = [1.0] * block.ndim
            scale_factor = 1.0 / self.downscale_factor
            scale_factors[-2] = scale_factor  # Y dimension
            scale_factors[-1] = scale_factor  # X dimension

            return rescale(
                block,
                scale=scale_factors,
                order=order,
                preserve_range=True,
                anti_aliasing=False,  # Already applied Gaussian filter
                channel_axis=None,
            ).astype(block.dtype)

        for _ in range(1, validated_levels + 1):
            # Check if Y or X dimensions are too small to downscale further
            if (
                current_array.shape[-2] / self.downscale_factor < 1
                or current_array.shape[-1] / self.downscale_factor < 1
            ):
                # Stop creating more levels if dimensions become too small
                break

            # Validate downscale_method, if not set silently to default Gaussian
            method = self.downscale_method
            if method not in {"gaussian", "nearest"}:
                method = "gaussian"

            # Apply the chosen downscale method
            if method == "gaussian":
                current_array = self._downscale_gaussian(
                    current_array, rescale_yx_block
                )
            elif method == "nearest":
                current_array = self._downscale_nearest(current_array, rescale_yx_block)

            if current_array is None:
                # If rescaling failed, stop here
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
            return self._downscale_gaussian_gpu(current_array, rescale_func)
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

    def _downscale_gaussian_gpu(self, current_array: da.Array, rescale_func) -> Optional[da.Array]:
        """Apply Gaussian filtering followed by downscaling on GPU.
        
        Args:
            current_array: Current array to downscale.
            rescale_func: Function to use for rescaling blocks (unused in GPU implementation).
            
        Returns:
            Downscaled array or None if operation failed.
        """
        try:
            import cupy as cp
            import cupyx.scipy.ndimage as ndi
        except ImportError:
            return None
            
        sigma = [0.0] * current_array.ndim
        sigma[-2] = (self.downscale_factor - 1) / 4.0  # Y dimension
        sigma[-1] = (self.downscale_factor - 1) / 4.0  # X dimension

        # Rechunk for better GPU performance - use larger chunks for better throughput
        # Use a memory-based chunking strategy that dask can understand
        current_array = current_array.rechunk("100MB")

        # Calculate new chunk sizes for the output
        new_chunks = list(current_array.chunks)
        new_chunks[-2] = tuple(
            max(1, int(chunk_size / self.downscale_factor))
            for chunk_size in new_chunks[-2]
        )
        new_chunks[-1] = tuple(
            max(1, int(chunk_size / self.downscale_factor))
            for chunk_size in new_chunks[-1]
        )

        def filter_and_rescale(block, sigma_list, downscale_factor, cuda_device_id):
            """Apply both Gaussian filtering and rescaling in a single GPU operation.
            
            This combines both operations to minimize GPU memory transfers and
            maximize GPU utilization.
            """
            if block.size == 0:  # Handle empty blocks
                return block
            
            # Use the specified CUDA device
            with cp.cuda.Device(cuda_device_id):
                # Convert to CuPy array for GPU processing (single conversion)
                gpu_block = cp.asarray(block)
                
                try:
                    # Step 1: Apply Gaussian filter on GPU
                    filtered_gpu = ndi.gaussian_filter(gpu_block, sigma=sigma_list, mode="nearest")
                    
                    # Step 2: Apply rescaling on GPU (same GPU memory, no transfer)
                    zoom_factors = [1.0] * block.ndim
                    zoom_factor = 1.0 / downscale_factor
                    zoom_factors[-2] = zoom_factor  # Y dimension
                    zoom_factors[-1] = zoom_factor  # X dimension
                    
                    # Use order=1 for bicubic interpolation, prefilter=False since we already filtered
                    rescaled_gpu = ndi.zoom(filtered_gpu, zoom=zoom_factors, order=1, prefilter=False)
                    
                    # Single conversion back to NumPy
                    result = cp.asnumpy(rescaled_gpu).astype(block.dtype)
                    
                    # Clear GPU memory explicitly for better memory management
                    del gpu_block, filtered_gpu, rescaled_gpu
                    
                    return result
                    
                except Exception as e:
                    # Clean up GPU memory before re-raising the error
                    import gc
                    gc.collect()
                    try:
                        cp.get_default_memory_pool().free_all_blocks()
                    except AttributeError:
                        # Older CuPy versions
                        cp.get_default_memory_pool().free_all_free()
                    
                    # Re-raise the original GPU error instead of falling back to CPU
                    raise RuntimeError(f"GPU operation failed on device {cuda_device_id}: {e}") from e

        try:
            # Single map_blocks operation combining both filtering and rescaling
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
        except (ValueError, RuntimeError):
            return None


    def _downscale_nearest(
        self, current_array: da.Array, rescale_func
    ) -> Optional[da.Array]:
        """Apply nearest-neighbor downscaling.

        Args:
            current_array: Current array to downscale.
            rescale_func: Function to use for rescaling blocks.

        Returns:
            Downscaled array or None if operation failed.
        """
        if self._should_use_gpu():
            return self._downscale_nearest_gpu(current_array, rescale_func)
        else:
            return self._downscale_nearest_cpu(current_array, rescale_func)

    def _downscale_nearest_cpu(
        self, current_array: da.Array, rescale_func
    ) -> Optional[da.Array]:
        """Apply nearest-neighbor downscaling on CPU.

        Args:
            current_array: Current array to downscale.
            rescale_func: Function to use for rescaling blocks.

        Returns:
            Downscaled array or None if operation failed.
        """
        # Calculate new chunk sizes (also downscaled for Y and X dimensions)
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

    def _downscale_nearest_gpu(
        self, current_array: da.Array, rescale_func
    ) -> Optional[da.Array]:
        """Apply nearest-neighbor downscaling on GPU.

        Args:
            current_array: Current array to downscale.
            rescale_func: Function to use for rescaling blocks.

        Returns:
            Downscaled array or None if operation failed.
        """
        # TODO: Implement GPU-accelerated nearest-neighbor downscaling
        # For now, fall back to CPU implementation
        return self._downscale_nearest_cpu(current_array, rescale_func)

    def create_coordinate_transformations_for_levels(
        self,
        coordinate_transformations: Optional[List[ScaleTransformation]],
        image_shape: tuple,
        num_actual_levels: int,
    ) -> List[List[ScaleTransformation]]:
        """Create coordinate transformations for each downscale level.

        Takes the scale transformations provided for the original image and
        adjusts them appropriately for each downscale level. The spatial dimensions
        (Y and X, which are the last two dimensions) are multiplied by the
        downscale factor for each level.

        Args:
            coordinate_transformations: Original coordinate transformations.
            image_shape: Shape of the original image.
            num_actual_levels: Number of actual resolution levels created.

        Returns:
            List of scale transformation lists, one for each resolution level.
            The first list corresponds to the original image, subsequent lists
            correspond to progressively downscaled levels.
        """
        if coordinate_transformations is None:
            return []

        # Create coordinate transformations for each level
        all_transformations = []

        for level in range(num_actual_levels):
            level_transformations = []

            for transform in coordinate_transformations:
                # For scale transformations, adjust spatial dimensions (Y, X)
                new_scale = transform.scale.copy()

                # The last two dimensions are always Y, X in our schema
                # Multiply by downscale_factor^level for these dimensions
                scale_factor = self.downscale_factor**level
                new_scale[-2] *= scale_factor  # Y dimension
                new_scale[-1] *= scale_factor  # X dimension

                level_transformations.append(ScaleTransformation(scale=new_scale))

            all_transformations.append(level_transformations)

        return all_transformations

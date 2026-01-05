"""Downscaling functionality for OME-Zarr multiscale pyramids."""

from typing import List, Literal, Optional

import dask.array as da
import dask_image.ndfilters
import numpy as np
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
    ):
        """Initialize the downscaler.

        Args:
            downscale_factor: Factor by which to downscale each level (default: 2.0).
            downscale_method: Method to use for downscaling ("gaussian" or "nearest").
            downscale_levels: Number of downscale levels to create. If None, no downscaling.
        """
        if downscale_factor <= 1.0:
            raise ValueError(f"downscale_factor must be > 1.0, got {downscale_factor}")

        self.downscale_factor = downscale_factor
        self.downscale_method = downscale_method
        self.downscale_levels = downscale_levels

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

        for i in range(1, validated_levels + 1):
            scale_factor = 1 / (self.downscale_factor * i)

            # Check if Y or X dimensions are too small to downscale further
            if image.shape[-2] * scale_factor < 1 or image.shape[-1] * scale_factor < 1:
                break

            # Get the downscaled array
            downscaled: da.Array = self._downscale_gaussian(
                image, scale_factor=scale_factor
            ) if self.downscale_method == "gaussian" else self._downscale_nearest(
                image, scale_factor=scale_factor
            )

            arrays.append(downscaled)

        return arrays

    def __rescale_yx_block(
        self, block: np.ndarray, order: int, scale_factor: float
    ) -> np.ndarray:
        """
        Rescale only the last two dimensions of a block by the downscale factor.
        Parameter `order` is the order of interpolation. 0 = nearest-neighbor,
        1 = bilinear. See skimage.transform.warp documentation for details.
        """
        # Create scale factors: 1 for all dimensions except last two
        scale_factors = [1.0] * block.ndim
        scale_factors[-2] = scale_factor  # Y dimension
        scale_factors[-1] = scale_factor  # X dimension

        rescaled: np.ndarray = rescale(
            block,
            scale=scale_factors,
            order=order,
            preserve_range=True,
            anti_aliasing=False,  # Already applied Gaussian filter
            channel_axis=None,
        ).astype(block.dtype)

        return rescaled

    def _downscale_gaussian(
        self,
        original_array: da.Array,
        scale_factor: float,
    ) -> da.Array:
        """Apply Gaussian filtering followed by downscaling.

        Args:
            current_array: Current array to downscale.
            rescale_func: Function to use for rescaling blocks.

        Returns:
            Downscaled array or None if operation failed.
        """
        # Apply Gaussian filter before downscaling to prevent aliasing.
        # Use sigma=1 for Y and X dimensions. Other dimensions have sigma=0 (no filtering).
        sigma = [0.0] * original_array.ndim
        sigma[-2] = 1
        sigma[-1] = 1

        filtered_array = dask_image.ndfilters.gaussian_filter(
            original_array, sigma=sigma, mode="nearest"
        )

        # Calculate new chunk sizes (also downscaled for Y and X dimensions)
        new_chunks = list(filtered_array.chunks)
        for dim in [-2, -1]:
            new_chunks[dim] = tuple(
                max(1, int(chunk_size / (1 / scale_factor)))
                for chunk_size in new_chunks[dim]
            )

        downscaled: da.Array = da.map_blocks(
            self.__rescale_yx_block,
            filtered_array,
            dtype=filtered_array.dtype,
            chunks=new_chunks,
            drop_axis=None,
            new_axis=None,
            meta=np.array([], dtype=filtered_array.dtype),
            order=1,  # use bicubic interpolation on Gaussian-filtered data
            scale_factor=scale_factor,
        )

        return downscaled

    def _downscale_nearest(
        self, original_array: da.Array, scale_factor: float
    ) -> da.Array:
        """Apply nearest-neighbor downscaling.

        Args:
            current_array: Current array to downscale.
            rescale_func: Function to use for rescaling blocks.

        Returns:
            Downscaled dask array.
        """
        # Calculate new chunk sizes (also downscaled for Y and X dimensions)
        new_chunks = list(original_array.chunks)
        for dim in [-2, -1]:
            new_chunks[dim] = tuple(
                max(1, int(chunk_size / (1 / scale_factor)))
                for chunk_size in new_chunks[dim]
            )

        downscaled: da.Array = da.map_blocks(
            self.__rescale_yx_block,
            original_array,
            dtype=original_array.dtype,
            chunks=tuple(new_chunks),
            drop_axis=None,
            new_axis=None,
            meta=np.array([], dtype=original_array.dtype),
            order=0,  # nearest-neighbor interpolation,
            scale_factor=scale_factor,
        )

        return downscaled

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

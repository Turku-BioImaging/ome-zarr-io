"""Downscaling functionality for OME-Zarr multiscale pyramids."""

import warnings
from typing import Any, List, Literal, Optional

import dask.array as da
import numpy as np

from .schema_models import ScaleTransformation


def _validate_downscale_factor(downscale_factor: Any) -> int:
    """Return ``downscale_factor`` as an int, or raise if it is not an integer >= 2.

    Integral floats (e.g. ``2.0``) are accepted for backward compatibility.
    """
    valid = (
        isinstance(downscale_factor, (int, float, np.integer, np.floating))
        and float(downscale_factor).is_integer()
        and downscale_factor >= 2
    )
    if not valid:
        raise ValueError(
            f"downscale_factor must be an integer >= 2, got {downscale_factor!r}"
        )
    return int(downscale_factor)


class Downscaler:
    """Handles downscaling operations for creating multiscale image pyramids.

    Level ``L`` of the pyramid is ``downscale_factor ** L`` times smaller than the
    original image in Y and X. Intensity images are downscaled by averaging each
    block of pixels ("mean"); label images by taking one pixel per block
    ("nearest"), which preserves discrete label values.
    """

    def __init__(
        self,
        downscale_factor: int = 2,
        downscale_method: Literal["mean", "nearest", "gaussian"] = "mean",
        downscale_levels: Optional[int] = None,
    ):
        """Initialize the downscaler.

        Args:
            downscale_factor: Integer factor (>= 2) by which each level is downscaled
                relative to the previous one (default: 2).
            downscale_method: "mean" (block average, i.e. a box filter followed by
                subsampling; for intensity images) or
                "nearest" (for label images). "gaussian" is a deprecated alias for "mean".
            downscale_levels: Number of downscale levels to create. If None, no downscaling.
        """
        self.downscale_factor = _validate_downscale_factor(downscale_factor)

        if downscale_method == "gaussian":
            warnings.warn(
                'downscale_method="gaussian" is deprecated; use "mean" instead.',
                DeprecationWarning,
                # Downscaler is created by Writer; point at the user's Writer call.
                stacklevel=3,
            )
            downscale_method = "mean"
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

        # Calculate maximum possible levels (each level must be at least 1 pixel)
        max_levels = 0
        test_size = min_spatial_dim // self.downscale_factor
        while test_size >= 1:
            max_levels += 1
            test_size //= self.downscale_factor

        if self.downscale_levels > max_levels:
            warnings.warn(
                f"Requested {self.downscale_levels} downscale levels with factor "
                f"{self.downscale_factor} is too many for image with spatial "
                f"dimensions {image_shape[-2:]}. Using the maximum possible "
                f"number of levels: {max_levels}.",
                UserWarning,
            )
            return max_levels

        return self.downscale_levels

    def create_downscaled_arrays(self, image: da.Array) -> List[da.Array]:
        """Create downscaled arrays for multiscale representation.

        Creates a list of dask arrays where level ``L`` is downscaled by
        ``downscale_factor ** L`` in the spatial dimensions (Y and X axes), while
        preserving all other dimensions (time, channel, Z). Each level is computed
        directly from the original image; for block averaging and striding this
        gives the same result as downscaling the previous level.

        Spatial sizes are ``floor(size / downscale_factor ** L)``: pixels that do
        not fill a whole block at the bottom/right edge are dropped.

        Args:
            image: The input dask array to downscale.

        Returns:
            List of downscaled dask arrays. The first array is the original image,
            followed by progressively downscaled versions. The number of levels
            is determined by downscale_levels parameter, validated against the
            minimum spatial dimension of the input image.
        """
        validated_levels = self.validate_downscale_levels(image.shape)

        arrays = [image]  # Level 0: original resolution

        for level in range(1, validated_levels + 1):
            block = self.downscale_factor**level
            downscaled = (
                self._downscale_nearest(image, block)
                if self.downscale_method == "nearest"
                else self._downscale_mean(image, block)
            )
            arrays.append(downscaled)

        return arrays

    @staticmethod
    def _downscale_mean(image: da.Array, block: int) -> da.Array:
        """Average each ``block`` x ``block`` tile of the Y/X dimensions.

        Equivalent to a ``block`` x ``block`` box filter followed by taking every
        ``block``-th pixel (box / area downsampling).

        Args:
            image: Input dask array.
            block: Block edge length in pixels.

        Returns:
            Downscaled dask array with the same dtype as the input.
        """
        averaged: da.Array = da.coarsen(
            np.mean,
            image,
            {image.ndim - 2: block, image.ndim - 1: block},
            trim_excess=True,
        )
        if np.issubdtype(image.dtype, np.integer):
            averaged = da.rint(averaged)
        downscaled: da.Array = averaged.astype(image.dtype)
        return downscaled

    @staticmethod
    def _downscale_nearest(image: da.Array, block: int) -> da.Array:
        """Take the top-left pixel of each ``block`` x ``block`` tile of Y/X.

        Args:
            image: Input dask array.
            block: Block edge length in pixels.

        Returns:
            Downscaled dask array; values are a subset of the input's values.
        """
        height = (image.shape[-2] // block) * block
        width = (image.shape[-1] // block) * block
        strided: da.Array = image[..., :height:block, :width:block]
        return strided

    def create_coordinate_transformations_for_levels(
        self,
        coordinate_transformations: Optional[List[ScaleTransformation]],
    ) -> List[List[ScaleTransformation]]:
        """Create coordinate transformations for each downscale level.

        Takes the scale transformations provided for the original image and
        adjusts them appropriately for each downscale level. The spatial dimensions
        (Y and X, which are the last two dimensions) are scaled by the cumulative
        downscale factor for each level (downscale_factor^level).

        Args:
            coordinate_transformations: Original coordinate transformations for level 0.
            If None, returns empty list.

        Returns:
            List of scale transformation lists, one for each resolution level.
            The first list corresponds to the original image (level 0), subsequent
            lists correspond to progressively downscaled levels. Returns empty list
            if coordinate_transformations is None.
        """
        if coordinate_transformations is None:
            return []

        # Create coordinate transformations for each level
        all_transformations = []

        num_downscale_levels = max(0, self.downscale_levels or 0)

        for level in range(num_downscale_levels + 1):
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

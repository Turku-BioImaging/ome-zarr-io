"""Main OME-Zarr writer implementation."""

import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, cast

import dask.array as da
import numpy as np
import zarr
from zarr.core.array import CompressorsLike

from .downscaler import Downscaler
from .schema_models import (
    Axis,
    Dataset,
    Multiscale,
    OMEMetadata,
    Omero,
    OMEZarrImageMetadata,
    ScaleTransformation,
    TranslationTransformation,
)


class OmeZarrImage:
    """Class for writing valid OME-Zarr 0.5 multiscale images.

    Provides functionality to write image data in the OME-Zarr format,
    a cloud-optimized format for bioimaging data supporting multiscale pyramids,
    metadata, and coordinate transformations.
    """

    def __init__(
        self,
        path: Union[str, Path],
        image: Union[da.Array, np.ndarray],
        dims: List[str],
        axis_units: Union[List[Axis], Dict[str, Any]],
        downscale_method: Literal["gaussian", "nearest"] = "gaussian",
        scale_transformations: Optional[Dict[str, Any]] = None,
        downscale_levels: Optional[int] = None,
        downscale_factor: float = 2.0,
        overwrite: bool = False,
        omero_metadata: Optional[Omero] = None,
    ):
        """Initialize the OME-Zarr writer.

        Args:
            path: Path where the OME-Zarr will be written.
            image: Dask or NumPy array representing the image data.
            dims: List of dimension names (e.g., ["t", "c", "z", "y", "x"]).
            axis_units: Either a list of Axis objects whose length corresponds to the number
                of dimensions of the input image, or a dictionary that specifies units for each
                dimension. Dictionary format:
                - Per-dimension units: {"t": "second", "z": "micrometer", "y": "micrometer", "x": "micrometer"}
            downscale_method: Method to use for downscaling. Either Gaussian filter or nearest-neighbor interpolation;
            default "gaussian". Use Gaussian filtering for intensity images to avoid aliasing artifacts in downscaled images.
            Label images should be downscaled using nearest-neighbor interpolation.
            scale_transformations: Optional dictionary specifying scale values for dimensions.
                Examples:
                - {"z": 0.25, "y": 0.1, "x": 0.1} for spatial dimensions
                - {"t": 0.5, "z": 0.25, "y": 0.1, "x": 0.1} including time axis
                Units are determined by the axis_units parameter. Scale values will be
                automatically adjusted for each downscale level.
            downscale_levels: Optional number of downscale levels to create. If `None`, no downscaling is performed.
            downscale_factor: Factor by which to downscale each level (default: 2.0).
            overwrite: Whether to overwrite existing files.
            omero_metadata: Optional OMERO metadata for channel display configuration.
                Must be an Omero object containing channel information for image visualization.
            zarr_backend: Zarr backend to use for writing. Either "zarr-python" or "zarrs"
                (default: "zarrs").
        """
        self.path = Path(path)
        # Convert numpy array to dask array if necessary
        if isinstance(image, np.ndarray):
            self.image = da.from_array(image, chunks="auto")
        else:
            self.image = image
        self.dims = dims
        self.overwrite = overwrite

        # Process and validate axis_units
        self.axes = self._process_axis_units(axis_units, dims, self.image.shape)

        # Process scale transformations (convert dict format to ScaleTransformation)
        self.coordinate_transformations = self._process_scale_transformations(
            scale_transformations, dims
        )

        # Create downscaler instance
        self.downscaler = Downscaler(
            downscale_factor=downscale_factor,
            downscale_method=downscale_method,
            downscale_levels=downscale_levels,
        )

        # Store OMERO metadata
        self.omero_metadata = omero_metadata

    @property
    def downscale_levels(self) -> Optional[int]:
        """Get the number of downscale levels from the downscaler."""
        return self.downscaler.downscale_levels

    @property
    def downscale_factor(self) -> float:
        """Get the downscale factor from the downscaler."""
        return self.downscaler.downscale_factor

    @property
    def downscale_method(self) -> Literal["gaussian", "nearest"]:
        """Get the downscale method from the downscaler."""

        return cast(Literal["gaussian", "nearest"], self.downscaler.downscale_method)

    def _create_downscaled_arrays(self) -> List[da.Array]:
        """Create downscaled arrays for multiscale representation.

        Uses the Downscaler instance to create a list of dask arrays where each
        subsequent array is downscaled by the specified downscale_factor in the
        last two dimensions (Y and X axes), while preserving all other dimensions.

        Returns:
            List of downscaled dask arrays. The first array is the original image,
            followed by progressively downscaled versions.
        """
        return self.downscaler.create_downscaled_arrays(da.asarray(self.image))

    def _create_coordinate_transformations_for_levels(
        self,
    ) -> List[List[ScaleTransformation]]:
        """Create coordinate transformations for each downscale level.

        Uses the Downscaler instance to create coordinate transformations for each
        resolution level, adjusting spatial dimensions appropriately.

        Returns:
            List of scale transformation lists, one for each resolution level.
        """
        # First, get the actual arrays that will be created to determine num_levels
        arrays = self._create_downscaled_arrays()
        num_levels = len(arrays)

        return self.downscaler.create_coordinate_transformations_for_levels(
            self.coordinate_transformations, self.image.shape, num_levels
        )

    def _process_axis_units(
        self,
        axis_units: Union[List[Axis], Dict[str, Any]],
        dims: List[str],
        image_shape: Any,
    ) -> List[Axis]:
        """Process axis_units parameter and create a list of Axis objects.

        Args:
            axis_units: Either a list of Axis objects or a dictionary with unit specifications
            dims: List of dimension names
            image_shape: Shape of the input image

        Returns:
            List of Axis objects corresponding to the image dimensions

        Raises:
            ValueError: If axis_units doesn't match the expected format or dimensions
        """
        # Validate that dims length matches image dimensions
        if len(dims) != len(image_shape):
            raise ValueError(
                f"Length of dims ({len(dims)}) must match number of image dimensions ({len(image_shape)})"
            )

        if isinstance(axis_units, list):
            # Case 1: List of Axis objects
            if len(axis_units) != len(dims):
                raise ValueError(
                    f"Length of axis_units list ({len(axis_units)}) must match number of dimensions ({len(dims)})"
                )

            # Validate that all items are Axis objects
            for i, axis in enumerate(axis_units):
                if not isinstance(axis, Axis):
                    raise ValueError(
                        f"axis_units[{i}] must be an Axis object, got {type(axis)}"
                    )

            return axis_units.copy()

        elif isinstance(axis_units, dict):
            # Case 2: Dictionary that can be mapped to create Axis objects
            return self._create_axes_from_dict(axis_units, dims)

        else:
            raise ValueError(
                f"axis_units must be either a list of Axis objects or a dictionary, got {type(axis_units)}"
            )

    def _create_axes_from_dict(
        self, axis_dict: Dict[str, Any], dims: List[str]
    ) -> List[Axis]:
        """Create Axis objects from a dictionary specification.

        Args:
            axis_dict: Dictionary containing unit specifications. Per-dimension format: {"t": "second", "z": "micrometer", "y": "micrometer", "x": "micrometer"}
            dims: List of dimension names

        Returns:
            List of Axis objects
        """
        # Per-dimension format
        return self._create_axes_from_per_dimension_dict(axis_dict, dims)

    def _create_axes_from_per_dimension_dict(
        self, axis_dict: Dict[str, Any], dims: List[str]
    ) -> List[Axis]:
        """Create Axis objects from per-dimension unit specification.

        Args:
            axis_dict: Dictionary with dimension names as keys and units as values
                Example: {"t": "second", "z": "micrometer", "y": "micrometer", "x": "micrometer"}
            dims: List of dimension names

        Returns:
            List of Axis objects

        Raises:
            ValueError: If required dimensions are missing or invalid units are provided
        """
        # Create axes based on dimension names
        axes = []
        for dim in dims:
            dim_lower = dim.lower()

            # Get unit for this dimension (case insensitive lookup)
            unit = None
            for key, value in axis_dict.items():
                if key.lower() == dim_lower:
                    unit = value
                    break

            if dim_lower == "t":
                if unit is None:
                    raise ValueError(
                        f"Time dimension '{dim}' requires a unit specification"
                    )
                axes.append(Axis(name=dim, type="time", unit=unit))
            elif dim_lower == "c":
                # Channel dimension - no unit needed, can be omitted from axis_units
                axes.append(Axis(name=dim, type="channel", unit=None))
            elif dim_lower in ["x", "y", "z"]:
                if unit is None:
                    raise ValueError(
                        f"Spatial dimension '{dim}' requires a unit specification"
                    )
                axes.append(Axis(name=dim, type="space", unit=unit))
            else:
                raise ValueError(
                    f"Unknown dimension '{dim}'. Valid dimensions are: t, c, z, y, x"
                )

        return axes

    def _process_scale_transformations(
        self,
        scale_transformations: Optional[Dict[str, Any]],
        dims: List[str],
    ) -> Optional[List[ScaleTransformation]]:
        """Process scale transformations and convert dictionary format to ScaleTransformation objects.

        Args:
            scale_transformations: Dictionary specifying pixel sizes for dimensions
            dims: List of dimension names

        Returns:
            List containing a single ScaleTransformation object or None

        Raises:
            ValueError: If the dictionary format is invalid
        """
        if scale_transformations is None:
            return None

        if isinstance(scale_transformations, dict):
            return self._create_scale_transformation_from_dict(
                scale_transformations, dims
            )
        else:
            raise ValueError(
                f"scale_transformations must be a dictionary, got {type(scale_transformations)}"
            )

    def _create_scale_transformation_from_dict(
        self,
        transform_dict: Dict[str, Any],
        dims: List[str],
    ) -> List[ScaleTransformation]:
        """Create a ScaleTransformation object from a dictionary specification.

        Args:
            transform_dict: Dictionary with dimension names as keys and scale values as values.
                Values must be numbers (int or float).
                Examples:
                - {"z": 0.25, "y": 0.1, "x": 0.1}
                - {"t": 0.5, "z": 0.25, "y": 0.1, "x": 0.1}
            dims: List of dimension names

        Returns:
            List containing a single ScaleTransformation object

        Raises:
            ValueError: If the dictionary format is invalid
        """
        # Initialize scale array with 1.0 for all dimensions
        scale = [1.0] * len(dims)

        # Process each dimension in the dictionary
        for dim_name, value in transform_dict.items():
            dim_name_lower = dim_name.lower()

            # Find the dimension index
            try:
                dim_index = [d.lower() for d in dims].index(dim_name_lower)
            except ValueError:
                raise ValueError(
                    f"Dimension '{dim_name}' not found in dims {dims}. "
                    f"Valid dimensions are: {', '.join(dims)}"
                )

            # Extract scale value - only accept numbers
            if isinstance(value, (int, float)):
                scale_value = float(value)
            else:
                raise ValueError(
                    f"Invalid value for dimension '{dim_name}': {value}. "
                    f"Expected a number (int or float)"
                )

            # Validate scale value
            if scale_value <= 0:
                raise ValueError(
                    f"Scale value for dimension '{dim_name}' must be positive, got {scale_value}"
                )

            scale[dim_index] = scale_value

        return [ScaleTransformation(scale=scale)]

    def write(
        self,
        chunks: Optional[Union[int, tuple, str]] = None,
        shards: Optional[Union[int, tuple]] = None,
        compressors: Optional[CompressorsLike] = None,
    ) -> None:
        """Write the image as OME-Zarr.

        Creates a complete OME-Zarr file with:
        - Multiscale pyramid datasets
        - Proper OME-Zarr 0.5 metadata
        - Coordinate transformations for each level
        - Zarr arrays for each resolution level

        Args:
            chunks: Chunk shape for zarr arrays. Can be an int, tuple, or "auto".
                If None, zarr will determine chunk size automatically.
            shards: Shard shape for zarr arrays (zarr v3 feature). Can be an int or tuple.
                If None, no sharding is applied.
            compressors: List of compressors to apply to zarr arrays. Can be codec objects
                from zarr.codecs (e.g., BloscCodec, GzipCodec, ZstdCodec) or a single
                compressor. If None, zarr will use default compression.
        """

        # Remove existing file if overwrite is True
        if self.overwrite and self.path.exists():
            if self.path.is_dir():
                shutil.rmtree(self.path)
            else:
                self.path.unlink()

        # Create the root zarr group
        root_group = zarr.create_group(
            str(self.path), overwrite=self.overwrite, zarr_format=3
        )

        # Generate downscaled arrays
        arrays: List[da.Array] = self._create_downscaled_arrays()

        # Generate coordinate transformations for each level
        level_transformations = self._create_coordinate_transformations_for_levels()

        # Create datasets for each resolution level
        datasets = []
        for level, array in enumerate(arrays):

            # Prepare args for zarr.create_array
            zarr_kwargs = {
                "name": str(level),
                "shape": array.shape,
                "dtype": array.dtype,
            }

            if chunks is not None:
                zarr_kwargs["chunks"] = chunks

            if shards is not None:
                zarr_kwargs["shards"] = shards

            if compressors is not None:
                zarr_kwargs["compressors"] = compressors

            # Create zarr array for this level and store the data
            zarr_array = root_group.create_array(**zarr_kwargs)
            zarr_array[:] = array # type: ignore

            # Get coordinate transformations for this level
            if level_transformations and level < len(level_transformations):
                transformations = level_transformations[level]
            else:
                # Create default scale transformation if none provided
                transformations = [ScaleTransformation(scale=[1.0] * len(self.dims))]

            # Create dataset metadata
            dataset = Dataset(
                path=str(level),
                coordinateTransformations=cast(
                    List[Union[ScaleTransformation, TranslationTransformation]],
                    transformations,
                ),
            )
            datasets.append(dataset)

        # Create multiscale metadata
        multiscale = Multiscale(
            datasets=datasets,
            axes=self.axes,
            name=self.path.stem,  # Use filename as name
        )

        # Create OME metadata
        ome_metadata = OMEMetadata(
            multiscales=[multiscale], version="0.5", omero=self.omero_metadata
        )

        # Create final metadata container
        metadata = OMEZarrImageMetadata(ome=ome_metadata)

        # Write metadata to zarr attributes
        root_group.attrs.update(metadata.to_dict())

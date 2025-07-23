"""Main OME-Zarr writer implementation."""

from typing import Any, Dict, List, Optional, Union, Literal
import zarr
import numpy as np
from pathlib import Path
import dask.array as da
import dask_image.ndfilters
from skimage.transform import rescale, resize
from zarr.core.array import CompressorsLike
from .schema_models import (
    ScaleTransformation,
    Axis,
    Dataset,
    Multiscale,
    OMEMetadata,
    OMEZarrImageMetadata,
    Omero,
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
        downscale_method: Literal["gaussian", "nearest"] = 'gaussian',
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
            downscale_method: Method to use for downscaling. Either Gaussian filter or nearest-neighbor interpolation; default "gaussian". Use Gaussian filtering for intensity images to avoid aliasing artifacts in downscaled images. Label images should be downscaled using nearest-neighbor interpolation.
                Another option like "nearest-neighbor" is being added.
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
        """
        import warnings

        self.path = Path(path)
        # Convert numpy array to dask array if necessary
        if isinstance(image, np.ndarray):
            self.image = da.from_array(image, chunks="auto")
        else:
            self.image = image
        self.dims = dims
        self.downscale_factor = downscale_factor
        self.downscale_method = downscale_method
        self.overwrite = overwrite

        # Process and validate axis_units
        self.axes = self._process_axis_units(axis_units, dims, self.image.shape)

        # Process scale transformations (convert dict format to ScaleTransformation)
        self.coordinate_transformations = self._process_scale_transformations(
            scale_transformations, dims
        )

        # Validate downscale_factor
        if downscale_factor <= 1.0:
            raise ValueError(f"downscale_factor must be > 1.0, got {downscale_factor}")

        # Validate downscale_levels against image dimensions
        if downscale_levels is not None and downscale_levels > 0:
            # Get the size of the smallest spatial dimension (Y and X are last two)
            min_spatial_dim = min(self.image.shape[-2], self.image.shape[-1])

            # Calculate maximum possible levels
            max_levels = 0
            test_size = min_spatial_dim
            while test_size >= downscale_factor:
                test_size = test_size / downscale_factor
                max_levels += 1

            if downscale_levels > max_levels:
                suggested_levels = max_levels
                suggested_factor = downscale_factor

                # Try to find a smaller factor that would work
                for factor in [1.5, 1.25, 1.1]:
                    if factor >= downscale_factor:
                        continue
                    test_levels = 0
                    test_size = min_spatial_dim
                    while test_size >= factor:
                        test_size = test_size / factor
                        test_levels += 1
                    if test_levels >= downscale_levels:
                        suggested_factor = factor
                        break

                warning_msg = (
                    f"Requested {downscale_levels} downscale levels with factor {downscale_factor} "
                    f"is too many for image with spatial dimensions {self.image.shape[-2:]}. "
                    f"Maximum possible levels: {max_levels}. "
                    f"Suggestions: reduce downscale_levels to {suggested_levels} "
                    f"or reduce downscale_factor to {suggested_factor}."
                )
                warnings.warn(warning_msg, UserWarning)

                # Automatically adjust to maximum possible levels
                downscale_levels = max_levels

        self.downscale_levels = downscale_levels

        # Store OMERO metadata
        self.omero_metadata = omero_metadata

    def _create_downscaled_arrays(self) -> List[da.Array]:
        """Create downscaled arrays for multiscale representation.

        Creates a list of dask arrays where each subsequent array is downscaled
        by the specified downscale_factor in the last two dimensions (Y and X axes),
        while preserving all other dimensions (time, channel, Z). Uses Gaussian
        filtering before downscaling to prevent aliasing artifacts.

        Returns:
            List of downscaled dask arrays. The first array is the original image,
            followed by progressively downscaled versions.
        """
        if self.downscale_levels is None or self.downscale_levels <= 0:
            return [self.image]

        arrays = [self.image]  # Level 0: original resolution
        current_array = self.image

        for level in range(1, self.downscale_levels + 1):
            
            downscale_factor = self.downscale_factor  # Capture for closure

            # Check if Y or X dimensions are too small to downscale further
            if (
                current_array.shape[-2] / self.downscale_factor < 1
                or current_array.shape[-1] / self.downscale_factor < 1
            ):
                # Stop creating more levels if dimensions become too small
                break
            
            # If Y and X dimensions are not too small for downscaling,
            # apply the chosen downscale method

            # If downscale_method is default/Gaussian:
            elif self.downscale_method == 'gaussian':

                # Apply Gaussian filter to prevent aliasing
                # Sigma is proportional to the downscale factor
                # For factor=2, sigma=0.5; for factor=4, sigma=1.0, etc.
                sigma = [0.0] * current_array.ndim
                sigma[-2] = (self.downscale_factor - 1) / 4.0  # Y dimension
                sigma[-1] = (self.downscale_factor - 1) / 4.0  # X dimension

                filtered_array = dask_image.ndfilters.gaussian_filter(
                    current_array, sigma=sigma, mode="nearest"
                )

                # Create a rescaling function that only operates on Y and X dimensions
                def rescale_yx_block(block, block_id=None):
                    """Rescale only the last two dimensions of a block by the downscale factor."""
                    
                    # Create scale factors: 1 for all dimensions except last two
                    scale_factors = [1.0] * block.ndim
                    scale_factor = 1.0 / downscale_factor
                    scale_factors[-2] = scale_factor  # Y dimension
                    scale_factors[-1] = scale_factor  # X dimension

                    return rescale(
                        block,
                        scale=scale_factors,
                        preserve_range=True,
                        anti_aliasing=False,  # Already applied Gaussian filter
                        channel_axis=None,
                    ).astype(block.dtype)

                # Calculate the expected output shape
                new_shape = list(current_array.shape)
                new_shape[-2] = int(new_shape[-2] / self.downscale_factor)  # Y dimension
                new_shape[-1] = int(new_shape[-1] / self.downscale_factor)  # X dimension

                # Calculate new chunk sizes (also downscaled for Y and X dimensions)
                new_chunks = list(filtered_array.chunks)
                new_chunks[-2] = tuple(
                    max(1, int(chunk_size / self.downscale_factor)) for chunk_size in new_chunks[-2]
                )
                new_chunks[-1] = tuple(
                    max(1, int(chunk_size / self.downscale_factor)) for chunk_size in new_chunks[-1]
                )

                try:
                    current_array = da.map_blocks(
                        rescale_yx_block,
                        filtered_array,
                        dtype=filtered_array.dtype,
                        chunks=new_chunks,
                        drop_axis=None,
                        new_axis=None,
                        meta=np.array([], dtype=filtered_array.dtype),
                    )

                    arrays.append(current_array)

                except (ValueError, RuntimeError):
                    # If rescaling fails, stop here
                    break
            
            # If downscale_method is nearest-neighbor:
            elif self.downscale_method == "nearest-neighbor":

                def resize_yx_block(block, block_info = None, block_id=None):
                    """Resizes the N-dimensional images using nearest neighbor interpolation (order == 0)."""
                    
                    # Extract the target shape of the output chunk from the block-info 
                    output_shape = block_info[None]['chunk-shape']

                    return resize(
                        block,
                        output_shape=output_shape,
                        order=0,  # nearest-neighbor interpolation
                        # anti_aliasing=False,  # By default because data type is Bool
                    ).astype(block.dtype)
                    
                # Calculate the expected output shape
                new_shape = list(current_array.shape)
                new_shape[-2] = int(new_shape[-2] / self.downscale_factor)  # Y dimension
                new_shape[-1] = int(new_shape[-1] / self.downscale_factor)  # X dimension
                
                # Calculate new chunk sizes (also downscaled for Y and X dimensions)
                new_chunks = list(current_array.chunks)
                new_chunks[-2] = tuple(
                    max(1, int(chunk_size / self.downscale_factor)) for chunk_size in new_chunks[-2]
                )
                new_chunks[-1] = tuple(
                    max(1, int(chunk_size / self.downscale_factor)) for chunk_size in new_chunks[-1]
                )

                try:
                    current_array = da.map_blocks(
                        resize_yx_block,
                        current_array,
                        dtype=current_array.dtype,
                        chunks=tuple(new_chunks),
                        drop_axis=None,
                        new_axis=None,
                        meta=np.array([], dtype=current_array.dtype),
                    )

                    arrays.append(current_array)

                except (ValueError, RuntimeError):
                    # If rescaling fails, stop here
                    break

        return arrays

    def _create_coordinate_transformations_for_levels(
        self,
    ) -> List[List[ScaleTransformation]]:
        """Create coordinate transformations for each downscale level.

        Takes the scale transformations provided for the original image and
        adjusts them appropriately for each downscale level. The spatial dimensions 
        (Y and X, which are the last two dimensions) are multiplied by the 
        downscale factor for each level.

        Returns:
            List of scale transformation lists, one for each resolution level.
            The first list corresponds to the original image, subsequent lists
            correspond to progressively downscaled levels.
        """
        if self.coordinate_transformations is None:
            return []

        # Determine the number of levels we'll actually create
        num_levels = 1  # At least the original level
        if self.downscale_levels is not None and self.downscale_levels > 0:
            # Check how many levels we can actually create based on image dimensions
            min_spatial_dim = min(self.image.shape[-2], self.image.shape[-1])

            max_possible_levels = 0
            test_size = min_spatial_dim
            while test_size >= self.downscale_factor:
                test_size = test_size / self.downscale_factor
                max_possible_levels += 1

            num_levels += min(self.downscale_levels, max_possible_levels)

        # Create coordinate transformations for each level
        all_transformations = []

        for level in range(num_levels):
            level_transformations = []

            for transform in self.coordinate_transformations:
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
                    raise ValueError(f"Time dimension '{dim}' requires a unit specification")
                axes.append(Axis(name=dim, type="time", unit=unit))
            elif dim_lower == "c":
                # Channel dimension - no unit needed, can be omitted from axis_units
                axes.append(Axis(name=dim, type="channel", unit=None))
            elif dim_lower in ["x", "y", "z"]:
                if unit is None:
                    raise ValueError(f"Spatial dimension '{dim}' requires a unit specification")
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
            return self._create_scale_transformation_from_dict(scale_transformations, dims)
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
        import shutil

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
        arrays = self._create_downscaled_arrays()

        # Generate coordinate transformations for each level
        level_transformations = self._create_coordinate_transformations_for_levels()

        # Create datasets for each resolution level
        datasets = []
        for level, array in enumerate(arrays):
            # Convert dask array to numpy for zarr storage
            array_data: np.ndarray = np.asarray(
                array.compute() if hasattr(array, "compute") else array
            )

            # Prepare zarr array creation arguments
            zarr_kwargs = {
                "name": str(level),
                "shape": array_data.shape,
                "dtype": array_data.dtype,
            }

            # Add chunks parameter if specified
            if chunks is not None:
                zarr_kwargs["chunks"] = chunks

            # Add shards parameter if specified (zarr v3 feature)
            if shards is not None:
                zarr_kwargs["shards"] = shards

            # Add compressors parameter if specified
            if compressors is not None:
                zarr_kwargs["compressors"] = compressors

            # Create zarr array for this level and store the data
            zarr_array = root_group.create_array(**zarr_kwargs)
            zarr_array[:] = array_data

            # Get coordinate transformations for this level
            if level_transformations and level < len(level_transformations):
                transformations = level_transformations[level]
            else:
                # Create default scale transformation if none provided
                transformations = [ScaleTransformation(scale=[1.0] * len(self.dims))]

            # Create dataset metadata
            from typing import cast
            from .schema_models import TranslationTransformation
            dataset = Dataset(
                path=str(level), 
                coordinateTransformations=cast(
                    List[Union[ScaleTransformation, TranslationTransformation]], 
                    transformations
                )
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
            multiscales=[multiscale], 
            version="0.5",
            omero=self.omero_metadata
        )

        # Create final metadata container
        metadata = OMEZarrImageMetadata(ome=ome_metadata)

        # Write metadata to zarr attributes
        root_group.attrs.update(metadata.to_dict())

        zarr.consolidate_metadata(str(self.path))

        print(f"✅ Successfully wrote OME-Zarr to: {self.path}")
        print(f"   - {len(arrays)} resolution levels")
        print(f"   - Shape: {self.image.shape}")
        print(f"   - Dimensions: {self.dims}")
        print(f"   - Axes: {[(ax.name, ax.type, ax.unit) for ax in self.axes]}")

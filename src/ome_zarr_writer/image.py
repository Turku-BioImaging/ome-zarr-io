"""Main OME-Zarr writer implementation."""

from typing import Any, Dict, List, Optional, Tuple, Union
import zarr
import numpy as np
from pathlib import Path
import dask.array as da
import dask_image.ndfilters
from skimage.transform import rescale
from .schema_models import ScaleTransformation, TranslationTransformation, Axis


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
        coordinate_transformations: Optional[List[Union[ScaleTransformation, TranslationTransformation]]] = None,
        downscale_levels: Optional[int] = None,
        downscale_factor: int = 2,
        overwrite: bool = False,
    ):
        """Initialize the OME-Zarr writer.

        Args:
            path: Path where the OME-Zarr will be written.
            image: Dask or NumPy array representing the image data.
            dims: List of dimension names (e.g., ["t", "c", "z", "y", "x"]).
            axis_units: Either a list of Axis objects whose length corresponds to the number 
                of dimensions of the input image, or a dictionary that can be mapped to 
                create Axis objects (e.g., {"unit": "micrometer", "time_unit": "second"}).
            coordinate_transformations: Optional list of coordinate transformations for the original image.
                These will be automatically adjusted for each downscale level.
            downscale_levels: Optional number of downscale levels to create. If `None`, no downscaling is performed.
            downscale_factor: Factor by which to downscale each level (default: 2).
            overwrite: Whether to overwrite existing files.
        """
        import warnings
        
        self.path = Path(path)
        # Convert numpy array to dask array if necessary
        if isinstance(image, np.ndarray):
            self.image = da.from_array(image, chunks="auto")
        else:
            self.image = image
        self.dims = dims
        self.coordinate_transformations = coordinate_transformations
        self.downscale_factor = downscale_factor
        self.overwrite = overwrite
        
        # Process and validate axis_units
        self.axes = self._process_axis_units(axis_units, dims, self.image.shape)
        
        # Validate downscale_factor
        if downscale_factor < 2:
            raise ValueError(f"downscale_factor must be >= 2, got {downscale_factor}")
        
        # Validate downscale_levels against image dimensions
        if downscale_levels is not None and downscale_levels > 0:
            # Get the size of the smallest spatial dimension (Y and X are last two)
            min_spatial_dim = min(self.image.shape[-2], self.image.shape[-1])
            
            # Calculate maximum possible levels
            max_levels = 0
            test_size = min_spatial_dim
            while test_size >= downscale_factor:
                test_size = test_size // downscale_factor
                max_levels += 1
            
            if downscale_levels > max_levels:
                suggested_levels = max_levels
                suggested_factor = downscale_factor
                
                # Try to find a smaller factor that would work
                for factor in range(2, downscale_factor):
                    test_levels = 0
                    test_size = min_spatial_dim
                    while test_size >= factor:
                        test_size = test_size // factor
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

    def create_multiscale_group(
        self,
        arrays: List[np.ndarray],
        axes: List[Dict[str, Any]],
        coordinate_transformations: Optional[List[List[Union[ScaleTransformation, TranslationTransformation]]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> zarr.Group:
        """Create a multiscale OME-Zarr group.

        Args:
            arrays: List of arrays representing different resolution levels
            axes: List of axis metadata
            coordinate_transformations: Optional coordinate transformations for each level
            metadata: Optional additional metadata

        Returns:
            The created zarr group
        """
        # Implementation will go here
        raise NotImplementedError("This method will be implemented")
    
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
            # Check if Y or X dimensions are too small to downscale further
            if (current_array.shape[-2] < self.downscale_factor or 
                current_array.shape[-1] < self.downscale_factor):
                # Stop creating more levels if dimensions become too small
                break
            
            # Apply Gaussian filter to prevent aliasing
            # Sigma is proportional to the downscale factor
            # For factor=2, sigma=0.5; for factor=4, sigma=1.0, etc.
            sigma = [0.0] * current_array.ndim
            sigma[-2] = (self.downscale_factor - 1) / 4.0  # Y dimension
            sigma[-1] = (self.downscale_factor - 1) / 4.0  # X dimension
            
            filtered_array = dask_image.ndfilters.gaussian_filter(
                current_array, 
                sigma=sigma,
                mode='nearest'
            )
            
            # Create a rescaling function that only operates on Y and X dimensions
            downscale_factor = self.downscale_factor  # Capture for closure
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
                    channel_axis=None
                ).astype(block.dtype)
            
            # Calculate the expected output shape
            new_shape = list(current_array.shape)
            new_shape[-2] = new_shape[-2] // self.downscale_factor  # Y dimension
            new_shape[-1] = new_shape[-1] // self.downscale_factor  # X dimension
            
            # Calculate new chunk sizes (also downscaled for Y and X dimensions)
            new_chunks = list(filtered_array.chunks)
            new_chunks[-2] = tuple(chunk_size // self.downscale_factor for chunk_size in new_chunks[-2])
            new_chunks[-1] = tuple(chunk_size // self.downscale_factor for chunk_size in new_chunks[-1])
            
            try:
                current_array = da.map_blocks(
                    rescale_yx_block,
                    filtered_array,
                    dtype=filtered_array.dtype,
                    chunks=new_chunks,
                    drop_axis=None,
                    new_axis=None,
                    meta=np.array([], dtype=filtered_array.dtype)
                )
                
                arrays.append(current_array)
                
            except (ValueError, RuntimeError):
                # If rescaling fails, stop here
                break
        
        return arrays
    
    def _create_coordinate_transformations_for_levels(self) -> List[List[Union[ScaleTransformation, TranslationTransformation]]]:
        """Create coordinate transformations for each downscale level.
        
        Takes the coordinate transformations provided for the original image and 
        adjusts them appropriately for each downscale level. For scale transformations,
        the spatial dimensions (Y and X, which are the last two dimensions) are 
        multiplied by the downscale factor for each level. Translation transformations 
        remain unchanged across levels.
        
        Returns:
            List of coordinate transformation lists, one for each resolution level.
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
                test_size = test_size // self.downscale_factor
                max_possible_levels += 1
            
            num_levels += min(self.downscale_levels, max_possible_levels)
        
        # Create coordinate transformations for each level
        all_transformations = []
        
        for level in range(num_levels):
            level_transformations = []
            
            for transform in self.coordinate_transformations:
                if isinstance(transform, ScaleTransformation):
                    # For scale transformations, adjust spatial dimensions (Y, X)
                    new_scale = transform.scale.copy()
                    
                    # The last two dimensions are always Y, X in our schema
                    # Multiply by downscale_factor^level for these dimensions
                    scale_factor = self.downscale_factor ** level
                    new_scale[-2] *= scale_factor  # Y dimension
                    new_scale[-1] *= scale_factor  # X dimension
                    
                    level_transformations.append(ScaleTransformation(scale=new_scale))
                
                elif isinstance(transform, TranslationTransformation):
                    # Translation transformations remain the same across levels
                    level_transformations.append(TranslationTransformation(
                        translation=transform.translation.copy()
                    ))
                else:
                    # For any other transformation type, keep it unchanged
                    level_transformations.append(transform)
            
            all_transformations.append(level_transformations)
        
        return all_transformations
    
    def _process_axis_units(self, axis_units: Union[List[Axis], Dict[str, Any]], dims: List[str], image_shape: Any) -> List[Axis]:
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
                    raise ValueError(f"axis_units[{i}] must be an Axis object, got {type(axis)}")
            
            return axis_units.copy()
        
        elif isinstance(axis_units, dict):
            # Case 2: Dictionary that can be mapped to create Axis objects
            return self._create_axes_from_dict(axis_units, dims)
        
        else:
            raise ValueError(
                f"axis_units must be either a list of Axis objects or a dictionary, got {type(axis_units)}"
            )
    
    def _create_axes_from_dict(self, axis_dict: Dict[str, Any], dims: List[str]) -> List[Axis]:
        """Create Axis objects from a dictionary specification.
        
        Args:
            axis_dict: Dictionary containing unit specifications (e.g., {"unit": "micrometer", "time_unit": "second"})
            dims: List of dimension names
            
        Returns:
            List of Axis objects
        """
        # Get default units
        space_unit = axis_dict.get("unit", "micrometer")
        time_unit = axis_dict.get("time_unit", None)
        
        # Create axes based on dimension names
        axes = []
        for dim in dims:
            dim_lower = dim.lower()
            
            if dim_lower == "t":
                axes.append(Axis(name=dim, type="time", unit=time_unit))
            elif dim_lower == "c":
                axes.append(Axis(name=dim, type="channel", unit=None))
            elif dim_lower in ["x", "y", "z"]:
                axes.append(Axis(name=dim, type="space", unit=space_unit))
            else:
                raise ValueError(
                    f"Unknown dimension '{dim}'. Valid dimensions are: t, c, z, y, x"
                )
        
        return axes
    


    def write(
        self,
        image: np.ndarray,
        pixel_size: Tuple[float, ...],
        units: Optional[List[str]] = None,
        channel_names: Optional[List[str]] = None,
    ) -> None:
        """Write an image as OME-Zarr.

        Args:
            image: The image array to write
            pixel_size: Physical pixel sizes for each spatial dimension
            units: Units for each spatial dimension
            channel_names: Names for each channel (if applicable)
        """
        # Implementation will go here
        raise NotImplementedError("This method will be implemented")

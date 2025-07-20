"""Main OME-Zarr writer implementation."""

from typing import Any, Dict, List, Optional, Tuple, Union
import zarr
import numpy as np
from pathlib import Path
import dask.array as da


class OMEZarrImage:
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
        coordinate_transformations: Optional[List[Dict[str, Any]]] = None,
        downscale_levels: Optional[int] = None,
        overwrite: bool = False,
    ):
        """Initialize the OME-Zarr writer.

        Args:
            path: Path where the OME-Zarr will be written.
            image: Dask or NumPy array representing the image data.
            dims: List of dimension names (e.g., ["t", "c", "z", "y", "x"]).
            coordinate_transformations: Optional list of coordinate transformation dicts.
            downscale_levels: Optional number of downscale levels to create. If `None`, no downscaling is performed.
            overwrite: Whether to overwrite existing files.
        """
        self.path = Path(path)
        # Convert numpy array to dask array if necessary
        if isinstance(image, np.ndarray):
            self.image = da.from_array(image, chunks="auto")
        else:
            self.image = image
        self.dims = dims
        self.coordinate_transformations = coordinate_transformations
        self.downscale_levels = downscale_levels
        self.overwrite = overwrite

    def create_multiscale_group(
        self,
        arrays: List[np.ndarray],
        axes: List[Dict[str, Any]],
        coordinate_transformations: Optional[List[List[Dict[str, Any]]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> zarr.Group:
        """Create a multiscale OME-Zarr group.

        Args:
            arrays: List of arrays representing different resolution levels
            axes: List of axis metadata
            coordinate_transformations: Optional coordinate transformations
            metadata: Optional additional metadata

        Returns:
            The created zarr group
        """
        # Implementation will go here
        raise NotImplementedError("This method will be implemented")
    
    def _create_downscaled_arrays(self) -> List[da.Array]:
        """Create downscaled arrays for multiscale representation.

        Returns:
            List of downscaled dask arrays
        """
        # Implementation will go here
        raise NotImplementedError("This method will be implemented")

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

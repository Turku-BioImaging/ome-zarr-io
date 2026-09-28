"""Main OME-Zarr writer implementation."""

import warnings
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, cast

import dask.array as da
import numpy as np
import zarr
from zarr.core.array import CompressorsLike

from .channels import ChannelSpec, any_needs_stats, parse_channels, resolve_channels
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


class Writer:
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
        downscale_method: Literal["mean", "nearest", "gaussian"] = "mean",
        scale_transformations: Optional[Dict[str, Any]] = None,
        downscale_levels: Optional[int] = None,
        downscale_factor: int = 2,
        overwrite: bool = False,
        omero_metadata: Optional[Omero] = None,
        channels: Optional[Union[Dict[str, Any], List[Any], Omero]] = None,
        colors: Optional[Literal["random"]] = None,
        color_seed: int = 0,
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
            downscale_method: Method to use for downscaling; default "mean". "mean" averages each
                block of pixels (a box filter followed by subsampling, also called area
                downsampling) and is intended for intensity images. "nearest" takes one pixel per
                block and should be used for label images. "gaussian" is a deprecated alias for "mean".
            scale_transformations: Optional dictionary specifying scale values for dimensions.
                Examples:
                - {"z": 0.25, "y": 0.1, "x": 0.1} for spatial dimensions
                - {"t": 0.5, "z": 0.25, "y": 0.1, "x": 0.1} including time axis
                Units are determined by the axis_units parameter. Scale values will be
                automatically adjusted for each downscale level.
            downscale_levels: Optional number of downscale levels to create. If `None`, no downscaling is performed.
            downscale_factor: Integer factor (>= 2) by which each level is downscaled relative
                to the previous one (default: 2).
            overwrite: Whether to overwrite existing files.
            omero_metadata: Deprecated; use ``channels`` instead.
                Optional OMERO metadata for channel display configuration.
                Must be an Omero object containing channel information for image visualization.
            channels: Plain-Python channel description, an alternative to ``omero_metadata``.
                A dict keyed by label (``{"DAPI": {"color": "0000FF", "window": (0, 4095)}}``),
                a list of labels, a list of dicts, or an Omero object. Per-channel keys:
                ``color`` (hex without "#", or "random"), ``window`` ("auto" (default),
                "minmax", (start, end), a Window, or None), ``family`` and ``active``.
                Window min/max are the data's min/max; "auto" start/end follow Fiji's
                auto-contrast. Requires a "c" axis with one entry per channel.
            colors: "random" assigns a distinct color to every channel without one.
            color_seed: Changes the automatic palette; the same seed gives the same colors.
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
        if omero_metadata is not None:
            self._warn_omero_deprecated("omero_metadata")
        self._channel_specs: Optional[List[ChannelSpec]] = None
        self.color_seed = color_seed
        if channels is not None:
            if omero_metadata is not None:
                raise ValueError("pass either channels or omero_metadata, not both")
            self._init_channels(channels, colors)
        elif colors is not None:
            raise ValueError("colors requires channels")

    @staticmethod
    def _warn_omero_deprecated(what: str) -> None:
        warnings.warn(
            f"{what} is deprecated; describe channels with a dict or list via "
            "channels=... instead.",
            DeprecationWarning,
            stacklevel=3,
        )

    def _init_channels(self, channels: Any, colors: Optional[str]) -> None:
        if "c" not in self.dims:
            raise ValueError("channels requires a 'c' axis in dims")
        parsed = parse_channels(channels, colors)
        n_axis = self.image.shape[self.dims.index("c")]
        n = len(parsed.channels) if isinstance(parsed, Omero) else len(parsed)
        if n != n_axis:
            raise ValueError(
                f"{n} channels given but the 'c' axis has {n_axis} entries"
            )
        if isinstance(parsed, Omero):
            self._warn_omero_deprecated("an Omero object in channels")
            self.omero_metadata = parsed
        else:
            self._channel_specs = parsed

    @property
    def downscale_levels(self) -> Optional[int]:
        """Get the number of downscale levels from the downscaler."""
        return self.downscaler.downscale_levels

    @property
    def downscale_factor(self) -> int:
        """Get the downscale factor from the downscaler."""
        return self.downscaler.downscale_factor

    @property
    def downscale_method(self) -> Literal["mean", "nearest"]:
        """Get the downscale method from the downscaler."""

        return cast(Literal["mean", "nearest"], self.downscaler.downscale_method)

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

        return self.downscaler.create_coordinate_transformations_for_levels(
            self._base_scale_transformations()
        )

    def _base_scale_transformations(self) -> List[ScaleTransformation]:
        """Level-0 scale transformations; unit scale if none were provided.

        Downscaled levels are derived from these, so even without user-provided
        pixel sizes each level's Y/X scale reflects its size relative to level 0.
        """
        if self.coordinate_transformations is not None:
            return self.coordinate_transformations
        return [ScaleTransformation(scale=[1.0] * len(self.dims))]

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

    def add_labels(
        self,
        name: str,
        array: Union[da.Array, np.ndarray],
        colors: Optional[List[Dict[str, Any]]] = None,
        properties: Optional[List[Dict[str, Any]]] = None,
        source_image: str = "../../",
        overwrite: bool = False,
        downscale_method: Optional[Literal["mean", "nearest", "gaussian"]] = "nearest",
        downscale_levels: Optional[int] = None,
        downscale_factor: Optional[int] = None,
        chunks: Optional[Union[int, tuple, str]] = None,
        shards: Optional[Union[int, tuple]] = None,
        compressors: Optional[CompressorsLike] = None,
    ) -> zarr.Group:
        """Attach a label image to an existing OME-Zarr image group.

        This creates the NGFF 0.5 structure expected for labeled segmentation masks:
        - parent image group: ome.labels = ["<name>"]
        - labels/<name>/zarr.json: ome.multiscales + ome.image-label metadata
        - label image arrays written as a multiscale pyramid

        See `https://ngff.openmicroscopy.org/specifications/0.5/index.html#labels-metadata` for details.

        Args:
            name: Name of the label image as it will appear under the parent labels group.
            array: Label array to attach. Should match the source image shape.
            colors: Optional list of {"label-value": ..., "rgba": [...]} entries.
            properties: Optional list of label metadata entries.
            source_image: Relative path from the label image group to the source image group.
            overwrite: If True, overwrite an existing label image with the same name.
            downscale_method: Label images should typically use nearest-neighbor downsampling.
            downscale_levels: Optional override for number of label pyramid levels.
            downscale_factor: Optional override for the label downscaling factor.
            chunks: Chunk specification for label arrays.
            shards: Optional sharding settings.
            compressors: Optional compressor configuration.

        Returns:
            The newly created label group.
        """
        if not self.path.exists():
            raise ValueError(
                f"Cannot add labels to image at '{self.path}' because the image group does not exist. "
                "Write the image first with .write()."
            )

        if not isinstance(array, (np.ndarray, da.Array)):
            raise TypeError(
                f"array must be a numpy array or dask array, got {type(array)}"
            )

        label_array = (
            da.from_array(array, chunks="auto")
            if isinstance(array, np.ndarray)
            else array
        )
        if label_array.ndim != len(self.dims):
            raise ValueError(
                f"Label array dimensions ({label_array.ndim}) do not match source image dimensions ({len(self.dims)})"
            )

        if label_array.shape != self.image.shape:
            raise ValueError(
                f"Label array shape {label_array.shape} does not match source image shape {self.image.shape}"
            )

        root_group = zarr.open_group(str(self.path), mode="a")

        labels_group = root_group.require_group("labels")
        labels_group_ome_attrs = dict(labels_group.attrs.get("ome", {}))  # type: ignore
        labels = labels_group_ome_attrs.get("labels", [])  # type: ignore
        if name in labels and not overwrite:
            raise ValueError(f"Label '{name}' already exists in this image group")

        if overwrite and name in labels_group.group_keys():
            del labels_group[name]

        label_group = labels_group.require_group(name)

        scaler = Downscaler(
            downscale_factor=(
                downscale_factor
                if downscale_factor is not None
                else self.downscale_factor
            ),
            downscale_method=(
                downscale_method if downscale_method is not None else "nearest"
            ),
            downscale_levels=(
                downscale_levels
                if downscale_levels is not None
                else (self.downscale_levels or 0)
            ),
        )
        label_arrays = scaler.create_downscaled_arrays(da.asarray(label_array))

        # Base scale/axes come from the parent image; the per-level multiplier uses
        # the label's own downscale factor.
        label_level_transformations = (
            scaler.create_coordinate_transformations_for_levels(
                self._base_scale_transformations()
            )
        )

        datasets = []
        for level, level_array in enumerate(label_arrays):
            zarr_kwargs = {
                "name": str(level),
                "shape": level_array.shape,
                "dtype": level_array.dtype,
            }
            if chunks is not None:
                zarr_kwargs["chunks"] = chunks
            if shards is not None:
                zarr_kwargs["shards"] = shards
            if compressors is not None:
                zarr_kwargs["compressors"] = compressors

            zarr_array = label_group.create_array(**zarr_kwargs)
            zarr_array[:] = np.asarray(level_array)  # type: ignore

            transformations = label_level_transformations[level]

            dataset = Dataset(
                path=str(level),
                coordinateTransformations=cast(
                    List[Union[ScaleTransformation, TranslationTransformation]],
                    transformations,
                ),
            )
            datasets.append(dataset)

        multiscale = Multiscale(
            datasets=datasets,
            axes=self.axes,
            name=name,
        )

        image_label = {
            "version": "0.5",
            "source": {"image": source_image},
        }
        if colors is not None:
            image_label["colors"] = colors
        if properties is not None:
            image_label["properties"] = properties

        label_group.attrs.update(
            {
                "ome": {
                    "version": "0.5",
                    "multiscales": [multiscale.to_dict()],
                    "image-label": image_label,
                }
            }
        )

        if name not in labels:
            labels.append(name)

        labels_group_ome_attrs["labels"] = labels  # type: ignore
        labels_group.attrs["ome"] = {  # type: ignore
            **labels_group_ome_attrs,
            "version": "0.5",
        }

        return label_group

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
        level0: Optional[np.ndarray] = None
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
            data = np.asarray(array)
            zarr_array[:] = data  # type: ignore
            if (
                level == 0
                and self._channel_specs
                and any_needs_stats(self._channel_specs)
            ):
                level0 = data

            # Get coordinate transformations for this level
            transformations = level_transformations[level]

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

        omero = self.omero_metadata
        if self._channel_specs is not None:
            omero = resolve_channels(
                self._channel_specs, level0, self.dims.index("c"), self.color_seed
            )

        # Create OME metadata
        ome_metadata = OMEMetadata(multiscales=[multiscale], version="0.5", omero=omero)

        # Create final metadata container
        metadata = OMEZarrImageMetadata(ome=ome_metadata)

        # Write metadata to zarr attributes
        root_group.attrs.update(metadata.to_dict())

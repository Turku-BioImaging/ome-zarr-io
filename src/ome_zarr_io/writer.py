"""Main OME-Zarr writer implementation."""

import math
import warnings
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, cast

import dask.array as da
import numpy as np
import zarr
from zarr.core.array import CompressorsLike

from .channels import (
    ChannelSpec,
    any_needs_stats,
    complete_omero,
    normalize_color,
    omero_needs_stats,
    parse_channels,
    random_colors,
    resolve_channels,
)
from .downscaler import Downscaler
from .reader import Reader
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
        name: Optional[str] = None,
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
                A channel without a color or window gets an automatic color or the
                full data range, because the spec requires both.
            channels: Plain-Python channel description, an alternative to ``omero_metadata``.
                A dict keyed by label (``{"DAPI": {"color": "0000FF", "window": (0, 4095)}}``),
                a list of labels, a list of dicts, or an Omero object. Per-channel keys:
                ``color`` (hex without "#", or "random"), ``window`` ("auto" (default),
                "minmax", (start, end), a Window, or None, which is written as the
                full data range like "minmax"), ``family`` and ``active``.
                Window min/max are the data's min/max; "auto" start/end follow Fiji's
                auto-contrast. Requires a "c" axis with one entry per channel.
                A channel without a ``color`` gets a distinct automatic one, because
                the spec requires every channel to have a color.
            colors: Kept for compatibility. Channels without a color always get an
                automatic one now, so "random" has no further effect.
            color_seed: Changes the automatic palette; the same seed gives the same colors.
            name: Name stored in the multiscale metadata. Defaults to the file name
                of ``path`` without its extension.
        """
        self._from_existing = False
        self.path = Path(path)
        # Convert numpy array to dask array if necessary
        if isinstance(image, np.ndarray):
            self.image = da.from_array(image, chunks="auto")
        else:
            self.image = image
        self.dims = dims
        self.overwrite = overwrite
        self.name = name

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
        if self.omero_metadata is not None:
            # Reject invalid colors now, before any pixel data is written.
            for channel in self.omero_metadata.channels:
                if channel.color:
                    normalize_color(channel.color)

    @classmethod
    def from_existing(cls, path: Union[str, Path], **unexpected: Any) -> "Writer":
        """Attach a Writer to an OME-Zarr image that is already on disk.

        The returned Writer is only for adding to the image, with
        :meth:`add_labels` or :meth:`add_channels`; :meth:`write` is disabled so the image cannot be
        overwritten. Everything ``add_labels`` needs is read from the image's
        metadata: its axes, shape, pixel sizes, number of levels and downscale
        factor. No pixel data is read.

        The image's pyramid must be one this library could have written: only Y
        and X downscaled, by a constant integer factor, with scales that follow,
        and no translation or multiscale-level coordinateTransformations. Anything
        else raises a ValueError, because labels written with it would not line up
        with the image.

        Args:
            path: Path to the image group (a standalone image or a field inside a
                plate), i.e. the directory that contains the ``multiscales``
                metadata.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If ``path`` is not an OME-Zarr 0.5 image, its group holds
                more than one ``multiscales`` entry, or its pyramid is not one that
                labels can be written for.
            TypeError: If any other argument is passed. ``from_existing`` never
                overwrites the image, so it has no ``overwrite`` argument; to
                replace a label image, pass ``overwrite=True`` to :meth:`add_labels`.
        """
        if "overwrite" in unexpected:
            raise TypeError(
                "Writer.from_existing() has no 'overwrite' argument because it never "
                "modifies the image itself. To replace an existing label image, "
                "pass overwrite=True to add_labels(), e.g. "
                "Writer.from_existing(path).add_labels(name, array, overwrite=True)."
            )
        if unexpected:
            raise TypeError(
                "Writer.from_existing() got an unexpected keyword argument "
                f"'{next(iter(unexpected))}'"
            )
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No OME-Zarr image found at '{path}'")
        try:
            reader = Reader(path)
        except (
            zarr.errors.GroupNotFoundError,
            zarr.errors.ContainsArrayError,
            KeyError,
            IndexError,
            TypeError,
            AttributeError,
            ValueError,
        ) as e:
            raise ValueError(
                f"'{path}' is not an OME-Zarr 0.5 image group (it has no valid "
                f"'multiscales' metadata): {e!r}. For a plate, open one of its "
                "fields, e.g. 'plate.zarr/A/1/0'."
            ) from e

        entries = reader._metadata.ome.multiscales
        if len(entries) > 1:
            names = ", ".join(repr(m.name) if m.name else "(unnamed)" for m in entries)
            raise ValueError(
                f"The image group at '{path}' holds {len(entries)} multiscale images "
                f"({names}). Labels are stored once per image group, and the NGFF "
                "0.5 specification does not say which of several multiscale images "
                "they belong to, so from_existing() cannot add labels to this "
                "group."
            )

        multiscale = reader._multiscale
        base_scale, factor = cls._read_pyramid(path, reader)
        writer = cls(
            path,
            reader._base_level_array(0),
            reader.dims,
            reader.axes,
            downscale_factor=factor,
            downscale_levels=len(multiscale.datasets) - 1,
            name=multiscale.name,
        )
        # Use the exact level-0 scale from disk, including the non-spatial axes.
        writer.coordinate_transformations = [ScaleTransformation(scale=base_scale)]
        writer._from_existing = True
        return writer

    @staticmethod
    def _read_pyramid(path: Path, reader: Reader) -> "tuple[List[float], int]":
        """Level-0 scale and downscale factor of an image on disk.

        Raises a ValueError unless every level is exactly what this library would
        have written for that factor (see :meth:`from_existing`).
        """
        multiscale = reader._multiscale
        n_axes = len(multiscale.axes)

        def unsupported(detail: str) -> ValueError:
            return ValueError(
                f"Cannot add labels to '{path}': {detail}. Labels can only be added "
                "to images whose pyramid downscales only Y and X by a constant "
                "integer factor, with no translation or multiscale-level "
                "coordinateTransformations."
            )

        if multiscale.coordinateTransformations:
            raise unsupported("the multiscale has its own coordinateTransformations")

        shapes: List[tuple] = []
        scales: List[List[float]] = []
        for level, dataset in enumerate(multiscale.datasets):
            if any(
                isinstance(t, TranslationTransformation)
                for t in dataset.coordinateTransformations
            ):
                raise unsupported(f"level {level} has a translation transformation")
            scale = next(
                t.scale
                for t in dataset.coordinateTransformations
                if isinstance(t, ScaleTransformation)
            )
            try:
                array = reader._root[dataset.path]
            except KeyError:
                raise unsupported(
                    f"the array '{dataset.path}' listed for level {level} does not exist"
                ) from None
            if not isinstance(array, zarr.Array):
                raise unsupported(f"'{dataset.path}' (level {level}) is not an array")
            if len(array.shape) != n_axes or len(scale) != n_axes:
                raise unsupported(
                    f"level {level} has {len(array.shape)} dimensions and "
                    f"{len(scale)} scale values for {n_axes} axes"
                )
            shapes.append(tuple(array.shape))
            scales.append([float(v) for v in scale])

        if len(shapes) == 1:
            return scales[0], 2  # nothing to infer; the factor is unused

        if scales[0][-1] <= 0:
            raise unsupported("the level-0 scale is not positive")
        ratio = scales[1][-1] / scales[0][-1]
        factor = round(ratio)
        if factor < 2 or not math.isclose(ratio, factor, rel_tol=1e-6):
            raise unsupported(
                f"the X scale changes by {ratio:g} between level 0 and level 1, "
                "which is not an integer factor of at least 2"
            )

        for level in range(len(shapes)):
            step = factor**level
            expected_shape = shapes[0][:-2] + (
                shapes[0][-2] // step,
                shapes[0][-1] // step,
            )
            if shapes[level] != expected_shape:
                raise unsupported(
                    f"level {level} has shape {shapes[level]}, but a Y/X-only "
                    f"downscale by {factor} gives {expected_shape}"
                )
            expected_scale = scales[0][:-2] + [
                scales[0][-2] * step,
                scales[0][-1] * step,
            ]
            if not all(
                math.isclose(a, b, rel_tol=1e-6)
                for a, b in zip(scales[level], expected_scale)
            ):
                raise unsupported(
                    f"level {level} has scale {scales[level]}, but a Y/X-only "
                    f"downscale by {factor} gives {expected_scale}"
                )
        return scales[0], factor

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

    def _image_level_count(self, image_group: zarr.Group) -> int:
        """Number of resolution levels of the image on disk, from its metadata."""
        ome = dict(image_group.attrs).get("ome")
        multiscales = ome.get("multiscales") if isinstance(ome, dict) else None
        datasets = (
            multiscales[0].get("datasets")
            if isinstance(multiscales, list)
            and multiscales
            and isinstance(multiscales[0], dict)
            else None
        )
        if not isinstance(datasets, list) or not datasets:
            raise ValueError(
                f"Cannot add labels to '{self.path}': it has no multiscales "
                "metadata, so it is not an OME-Zarr image. Write the image first "
                "with .write()."
            )
        return len(datasets)

    def _upscale_label_to_level0(self, array: da.Array, source_level: int) -> da.Array:
        """Nearest-neighbour upscale of a label computed on ``source_level`` to level 0.

        Factors are the ratios of the image's per-level scale transformations.
        """
        if isinstance(source_level, bool) or not isinstance(source_level, int):
            raise TypeError(f"source_level must be an int, got {type(source_level)}")
        root_group = zarr.open_group(str(self.path), mode="r")
        self._image_level_count(root_group)  # validates multiscales exist
        datasets = dict(root_group.attrs)["ome"]["multiscales"][0]["datasets"]  # type: ignore
        if not 0 <= source_level < len(datasets):
            raise ValueError(
                f"source_level={source_level} does not exist: the image at "
                f"'{self.path}' has levels 0..{len(datasets) - 1}."
            )
        if array.ndim != len(self.dims):
            raise ValueError(
                f"Label array dimensions ({array.ndim}) do not match source image "
                f"dimensions ({len(self.dims)})"
            )

        def level_scale(level: int) -> List[float]:
            for t in datasets[level].get("coordinateTransformations", []):
                if t.get("type") == "scale" and "scale" in t:
                    return [float(v) for v in t["scale"]]
            raise ValueError(
                f"Level {level} of the image has no scale transformation, so the "
                "upscaling factor for source_level cannot be determined."
            )

        if source_level == 0:
            return array
        s0, sk = level_scale(0), level_scale(source_level)
        factors = []
        for axis, (a, b) in enumerate(zip(s0, sk)):
            ratio = b / a
            factor = round(ratio)
            if factor < 1 or abs(ratio - factor) > 1e-6:
                raise ValueError(
                    f"Level {source_level} is not an integer multiple of level 0 "
                    f"along axis '{self.dims[axis]}' (scale ratio {ratio}); cannot "
                    "upscale by repetition."
                )
            factors.append(factor)

        # The array must match the image's shape at source_level (size 1 is allowed
        # on non-spatial axes, as for full-resolution labels).
        level_shape = root_group[datasets[source_level]["path"]].shape
        n_nonspatial = array.ndim - 2
        for axis, (size, expected) in enumerate(zip(array.shape, level_shape)):
            if size == expected or (size == 1 and axis < n_nonspatial):
                continue
            raise ValueError(
                f"Label array shape {array.shape} does not match image level "
                f"{source_level} shape {tuple(level_shape)}: axis "
                f"'{self.dims[axis]}' must be {expected}"
                + (" or 1" if axis < n_nonspatial else "")
                + ". Pass the array computed at that level, or the correct source_level."
            )

        out = array
        for axis, factor in enumerate(factors):
            if factor == 1 or (array.shape[axis] == 1 and axis < n_nonspatial):
                continue
            out = da.repeat(out, factor, axis=axis)
        # Level sizes are rounded when downscaling, so the repeat can be a pixel or
        # two larger or smaller than level 0: crop the excess, edge-pad the shortfall
        # (nearest neighbour).
        target = self.image.shape[: out.ndim]
        out = out[tuple(slice(0, min(n, t)) for n, t in zip(out.shape, target))]
        pad = [
            (0, 0) if (n == t or n == 1 and axis < n_nonspatial) else (0, t - n)
            for axis, (n, t) in enumerate(zip(out.shape, target))
        ]
        if any(p[1] for p in pad):
            out = da.pad(out, pad, mode="edge")
        return out

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
        source_level: Optional[int] = None,
    ) -> zarr.Group:
        """Attach a label image to an existing OME-Zarr image group.

        This creates the NGFF 0.5 structure expected for labeled segmentation masks:

        - parent image group: ome.labels = ["<name>"]
        - labels/<name>/zarr.json: ome.multiscales + ome.image-label metadata
        - label image arrays written as a multiscale pyramid

        See `https://ngff.openmicroscopy.org/specifications/0.5/index.html#labels-metadata` for details.

        Args:
            name: Name of the label image as it will appear under the parent labels group.
            array: Label array to attach. Must have an integer data type (uint8,
                int8, uint16, int16, uint32, int32, uint64 or int64); float and
                boolean arrays are rejected. Must have the same dimensions as the source
                image; each non-spatial axis (e.g. c, z, t) may have size 1 instead of
                the image's size, per the NGFF spec. Y/X must match the image.
            colors: Optional list of {"label-value": ..., "rgba": [...]} entries.
            properties: Optional list of label metadata entries.
            source_image: Relative path from the label image group to the source image group.
            overwrite: If True, overwrite an existing label image with the same name.
            downscale_method: Label images should typically use nearest-neighbor downsampling.
            downscale_levels: Deprecated. The label always gets the same number of
                levels as the image on disk, as the spec requires. Passing the
                image's own value emits a DeprecationWarning; any other value raises.
            downscale_factor: Optional override for the label downscaling factor. It
                must still allow as many levels as the image has.
            chunks: Chunk specification for label arrays.
            shards: Optional sharding settings.
            compressors: Optional compressor configuration.
            source_level: Resolution level of the image that ``array`` was computed
                on. ``None`` (default) means ``array`` is at full resolution (level 0).
                If given, ``array`` is upscaled to level 0 with nearest-neighbour
                repetition (lazily, for dask arrays) before the label pyramid is
                built. The per-axis factors come from the image's scale
                transformations, so anisotropic pyramids are handled. Non-spatial
                axes of size 1 are left as they are.

        Returns:
            The newly created label group.

        Raises:
            ValueError: If ``array`` is boolean (convert it explicitly, e.g.
                ``mask.astype("uint8")``), if ``source_level`` does not exist, or if
                ``array`` does not match the shape of that level.
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
        # NGFF 0.5: label pixels MUST be one of uint8, int8, uint16, int16, uint32,
        # int32, uint64, int64. Checked before anything is written.
        if array.dtype.kind not in "iu":
            hint = (
                ' Boolean masks are not accepted: convert explicitly with '
                'array.astype("uint8") and try again.'
                if array.dtype.kind == "b"
                else ""
            )
            raise ValueError(
                f"Label arrays must have an integer data type (uint8, int8, uint16, "
                f"int16, uint32, int32, uint64 or int64), got {array.dtype}.{hint}"
            )

        label_array = (
            da.from_array(array, chunks="auto")
            if isinstance(array, np.ndarray)
            else array
        )
        if source_level is not None:
            label_array = self._upscale_label_to_level0(label_array, source_level)
        if label_array.ndim != len(self.dims):
            raise ValueError(
                f"Label array dimensions ({label_array.ndim}) do not match source image dimensions ({len(self.dims)})"
            )

        # NGFF 0.5: each label dimension must equal the image's, or be 1 if that
        # dimension is irrelevant to the label (e.g. a mask for a single channel).
        # Y/X must match: they define the pyramid and the coordinate system.
        for axis, (label_size, image_size) in enumerate(
            zip(label_array.shape, self.image.shape)
        ):
            if label_size == image_size:
                continue
            if label_size == 1 and axis < label_array.ndim - 2:
                continue
            raise ValueError(
                f"Label array shape {label_array.shape} is incompatible with source "
                f"image shape {self.image.shape} (dims {self.dims}): axis "
                f"'{self.dims[axis]}' must be {image_size}"
                + (" or 1" if axis < label_array.ndim - 2 else "")
            )

        root_group = zarr.open_group(str(self.path), mode="a")

        # NGFF 0.5: a label image MUST have the same number of scale levels as the
        # image it annotates, so the count is read from the image on disk rather
        # than from this Writer's settings (which may not match it).
        image_levels = self._image_level_count(root_group)
        label_downscale_levels = image_levels - 1
        if downscale_levels is not None:
            if max(0, downscale_levels) != label_downscale_levels:
                raise ValueError(
                    f"downscale_levels={downscale_levels} would give the label "
                    f"{max(0, downscale_levels) + 1} levels, but the image at "
                    f"'{self.path}' has {image_levels}. A label must have the same "
                    "number of levels as its image; omit downscale_levels."
                )
            warnings.warn(
                "downscale_levels is deprecated in add_labels; the label always gets "
                "the same number of levels as the image on disk.",
                DeprecationWarning,
                stacklevel=2,
            )

        scaler = Downscaler(
            downscale_factor=(
                downscale_factor
                if downscale_factor is not None
                else self.downscale_factor
            ),
            downscale_method=(
                downscale_method if downscale_method is not None else "nearest"
            ),
            downscale_levels=label_downscale_levels,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the cap is reported as an error below
            possible = scaler.validate_downscale_levels(label_array.shape)
        if possible != label_downscale_levels:
            raise ValueError(
                f"The image at '{self.path}' has {image_levels} levels, but a label "
                f"of shape {label_array.shape} downscaled by a factor of "
                f"{scaler.downscale_factor} can only have {possible + 1}. Use a "
                "smaller downscale_factor so the label has as many levels as its image."
            )

        labels_group = root_group.require_group("labels")
        labels_group_ome_attrs = dict(labels_group.attrs.get("ome", {}))  # type: ignore
        labels = labels_group_ome_attrs.get("labels", [])  # type: ignore
        if name in labels and not overwrite:
            raise ValueError(f"Label '{name}' already exists in this image group")

        if overwrite and name in labels_group.group_keys():
            del labels_group[name]

        label_group = labels_group.require_group(name)

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
                # NGFF 0.5: dimension_names MUST match the names in "axes"
                "dimension_names": list(self.dims),
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

    def add_channels(
        self,
        array: Union[da.Array, np.ndarray],
        channels: Any,
        downscale_method: Literal["mean", "nearest"] = "mean",
    ) -> None:
        """Append channels to the image on disk, e.g. a background-subtracted copy.

        ``array`` is the new channels at full resolution (level 0). Its pyramid is
        built with the image's own downscale factor and number of levels, then the
        ``c`` axis of every level is extended and ``omero.channels`` gets one entry
        per new channel. The existing channels are not rewritten.

        Args:
            array: NumPy or dask array with the image's dimensions. Every axis but
                ``c`` must match the image, and the dtype must equal the image's
                (all channels of a zarr array share one dtype). Its ``c`` size is
                the number of new channels.
            channels: One entry per new channel, in any form ``Writer(channels=...)``
                accepts except an ``Omero`` object: a list of labels, a list of dicts
                or a dict keyed by label. Per channel: ``label``, ``color``,
                ``window``, ``family`` and ``active``; ``active=False`` hides the
                channel by default in viewers. Labels must differ from those already
                in the image. A missing color gets an automatic one that avoids the
                image's existing colors.
            downscale_method: "mean" (default) for intensity data.

        Raises:
            RuntimeError: If the Writer was not made with :meth:`from_existing`.
            TypeError: If ``array`` is not a NumPy or dask array.
            ValueError: If the image has no ``c`` axis or no ``omero`` channels
                matching its ``c`` axis, if ``array`` does not match the image's
                shape or dtype, if ``channels`` does not have one entry per new
                channel, or if a label already exists.

        If writing fails part way, the ``c`` axes are shrunk back and the image is
        left as it was.
        """
        if not self._from_existing:
            raise RuntimeError(
                "add_channels() needs an existing image: use "
                "Writer.from_existing(path).add_channels(...)."
            )
        if not isinstance(array, (np.ndarray, da.Array)):
            raise TypeError(
                f"array must be a numpy array or dask array, got {type(array)}"
            )
        if "c" not in self.dims:
            raise ValueError(
                f"The image at '{self.path}' has no 'c' axis (dims {self.dims}), so "
                "channels cannot be added to it."
            )
        c_axis = self.dims.index("c")
        if array.ndim != len(self.dims):
            raise ValueError(
                f"array has {array.ndim} dimensions but the image has "
                f"{len(self.dims)} ({self.dims})"
            )
        for axis, (size, image_size) in enumerate(zip(array.shape, self.image.shape)):
            if axis != c_axis and size != image_size:
                raise ValueError(
                    f"array shape {array.shape} does not match the image shape "
                    f"{self.image.shape} (dims {self.dims}): axis '{self.dims[axis]}' "
                    f"must be {image_size}; only 'c' may differ"
                )
        if array.shape[c_axis] < 1:
            raise ValueError("array has no channels")
        if array.dtype != self.image.dtype:
            raise ValueError(
                f"array has dtype {array.dtype} but the image has {self.image.dtype}; "
                "all channels of an image share one dtype. Convert explicitly with "
                "array.astype(...) if that is what you want."
            )
        n_new = array.shape[c_axis]

        root_group = zarr.open_group(str(self.path), mode="a")
        ome = dict(root_group.attrs.get("ome", {}))  # type: ignore
        omero = ome.get("omero")
        existing = omero.get("channels") if isinstance(omero, dict) else None
        n_old = self.image.shape[c_axis]
        if not isinstance(existing, list) or len(existing) != n_old:
            raise ValueError(
                f"The image at '{self.path}' has {n_old} channels on its 'c' axis but "
                f"{'no' if existing is None else len(existing)} entries in "
                "omero.channels, so new channels cannot be indexed consistently. "
                "Give the existing channels omero metadata first."
            )

        parsed = parse_channels(channels)
        if isinstance(parsed, Omero):
            raise ValueError("pass channels as labels or dicts, not an Omero object")
        if len(parsed) != n_new:
            raise ValueError(
                f"{len(parsed)} channels given but array has {n_new} on its 'c' axis"
            )
        old_labels = {c.get("label") for c in existing if isinstance(c, dict)}
        clashes = sorted(
            str(s.label) for s in parsed if s.label is not None and s.label in old_labels
        )
        if clashes:
            raise ValueError(f"channel labels already in the image: {clashes}")
        old_colors = [
            normalize_color(c["color"])
            for c in existing
            if isinstance(c, dict) and c.get("color")
        ]
        pool = random_colors(
            n_new,
            self.color_seed,
            avoid=old_colors
            + [s.color for s in parsed if s.color and not s.auto_color],
        )
        for i, spec in enumerate(parsed):
            if spec.auto_color or not spec.color:
                spec.color, spec.auto_color = pool[i], False

        # Locate and check every level before touching anything.
        image_levels = self._image_level_count(root_group)
        datasets = ome["multiscales"][0]["datasets"]
        targets = []
        for level, dataset in enumerate(datasets):
            target = root_group[dataset["path"]]
            if not isinstance(target, zarr.Array) or target.shape[c_axis] != n_old:
                raise ValueError(
                    f"Level {level} of '{self.path}' does not have {n_old} entries "
                    "on its 'c' axis; the image is inconsistent."
                )
            if target.dtype != array.dtype:
                raise ValueError(
                    f"Level {level} has dtype {target.dtype}, not {array.dtype}."
                )
            targets.append(target)

        new = da.from_array(array, chunks="auto") if isinstance(array, np.ndarray) else array
        scaler = Downscaler(
            downscale_factor=self.downscale_factor,
            downscale_method=downscale_method,
            downscale_levels=image_levels - 1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # reported as an error below
            possible = scaler.validate_downscale_levels(new.shape)
        if possible != image_levels - 1:
            raise ValueError(
                f"The image has {image_levels} levels but the new channels can only "
                f"have {possible + 1}."
            )
        levels = scaler.create_downscaled_arrays(new)

        old_shapes = [t.shape for t in targets]
        window_values: Optional[np.ndarray] = None
        try:
            for level, (target, data) in enumerate(zip(targets, levels)):
                if tuple(data.shape) != tuple(
                    n_new if a == c_axis else d
                    for a, d in enumerate(target.shape)
                ):
                    raise ValueError(
                        f"New channels at level {level} have shape {data.shape}, "
                        f"which does not fit the image level of shape {target.shape}."
                    )
                values = np.asarray(data)
                if level == 0:
                    window_values = values
                new_shape = list(target.shape)
                new_shape[c_axis] = n_old + n_new
                target.resize(tuple(new_shape))
                index = [slice(None)] * target.ndim
                index[c_axis] = slice(n_old, n_old + n_new)
                target[tuple(index)] = values
            resolved = resolve_channels(parsed, window_values, c_axis, self.color_seed)
        except BaseException:
            for target, shape in zip(targets, old_shapes):
                target.resize(shape)
            raise

        # Metadata last, so an interruption never leaves omero describing missing data.
        ome["omero"] = {
            **omero,
            "channels": existing + [self._channel_dict(c) for c in resolved.channels],
        }
        root_group.attrs["ome"] = ome  # type: ignore
        # Keep this Writer in step with the image on disk.
        self.image = Reader(self.path)._base_level_array(0)

    @staticmethod
    def _channel_dict(channel: Any) -> Dict[str, Any]:
        """An omero channel as it is stored in zarr.json (None fields left out)."""
        out: Dict[str, Any] = {}
        if channel.window is not None:
            w = channel.window
            out["window"] = {"start": w.start, "min": w.min, "end": w.end, "max": w.max}
        for key in ("label", "family", "color", "active"):
            value = getattr(channel, key)
            if value is not None:
                out[key] = value
        return out

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

        if self._from_existing:
            raise RuntimeError(
                f"This Writer was attached to the existing image at '{self.path}' "
                "with Writer.from_existing, so write() is disabled to protect the "
                "image. Use add_labels() or add_channels() to add to it."
            )

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
                # NGFF 0.5: dimension_names MUST match the names in "axes"
                "dimension_names": list(self.dims),
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
            if level == 0 and (
                (self._channel_specs and any_needs_stats(self._channel_specs))
                or (
                    self._channel_specs is None
                    and self.omero_metadata is not None
                    and omero_needs_stats(self.omero_metadata)
                )
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
            name=self.name if self.name is not None else self.path.stem,
        )

        omero = self.omero_metadata
        if self._channel_specs is not None:
            omero = resolve_channels(
                self._channel_specs, level0, self.dims.index("c"), self.color_seed
            )
        elif omero is not None:
            # NGFF 0.5: every channel MUST have a color and a window.
            omero = complete_omero(
                omero,
                level0,
                self.dims.index("c") if "c" in self.dims else None,
                self.color_seed,
            )

        # Create OME metadata
        ome_metadata = OMEMetadata(multiscales=[multiscale], version="0.5", omero=omero)

        # Create final metadata container
        metadata = OMEZarrImageMetadata(ome=ome_metadata)

        # Write metadata to zarr attributes
        root_group.attrs.update(metadata.to_dict())

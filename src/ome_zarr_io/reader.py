"""Reader module for OME-Zarr 0.5 filesets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import dask.array as da
import numpy as np
import zarr

from .schema_models import Axis, Multiscale, OMEZarrImageMetadata
from .validator import OMEZarrValidator


@dataclass
class PhysicalSize:
    """Physical size of a single axis (e.g. pixel/voxel spacing)."""

    value: float
    unit: Optional[str]


class Reader:
    """Reads and validates OME-Zarr 0.5 image filesets."""

    def __init__(self, path: Union[str, Path]):
        """Open an existing OME-Zarr fileset for reading.

        Args:
            path: Path to the root OME-Zarr group (the directory containing zarr.json).

        Raises:
            FileNotFoundError: If no zarr group exists at `path`.
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"No OME-Zarr group found at '{self.path}'")

        self._root = zarr.open_group(str(self.path), mode="r")
        self._metadata = OMEZarrImageMetadata.from_dict(dict(self._root.attrs))
        self._multiscale: Multiscale = self._metadata.ome.multiscales[0]
        self._validator = OMEZarrValidator()

    # -- Validation -----------------------------------------------------

    def validate(self, strict: bool = False) -> bool:
        """Validate the image (and any attached labels) against the OME-Zarr 0.5 schema.

        Args:
            strict: If True, validate against the `strict_image`/`strict_label` schemas
                instead of the permissive `image`/`label` schemas.

        Returns:
            True if the fileset is valid.

        Raises:
            jsonschema.exceptions.ValidationError: If validation fails.
        """
        image_schema = "strict_image" if strict else "image"
        self._validator.validate_metadata(dict(self._root.attrs), image_schema)

        label_schema = "strict_label" if strict else "label"
        for name in self.label_names:
            label_group = self._root["labels"][name]  # type: ignore[index]
            self._validator.validate_metadata(dict(label_group.attrs), label_schema)

        return True

    def get_validation_errors(self, strict: bool = False) -> List[str]:
        """Return a list of validation error messages instead of raising."""
        image_schema = "strict_image" if strict else "image"
        errors = self._validator.get_validation_errors(
            dict(self._root.attrs), image_schema
        )

        label_schema = "strict_label" if strict else "label"
        for name in self.label_names:
            label_group = self._root["labels"][name]  # type: ignore[index]
            errors.extend(
                self._validator.get_validation_errors(
                    dict(label_group.attrs), label_schema
                )
            )

        return errors

    # -- Channels ---------------------------------------------------------

    @property
    def channel_names(self) -> List[str]:
        """Names of channels defined in the OMERO metadata, in axis order."""
        omero = self._metadata.ome.omero
        if omero is None:
            return []
        return [channel.label for channel in omero.channels if channel.label is not None]

    def _channel_index(self, name: str) -> int:
        """Resolve a channel name to its position on the 'c' axis.

        Indexes into the raw `omero.channels` list (not `channel_names`) so
        unlabeled channels don't shift the position of later channels.
        """
        omero = self._metadata.ome.omero
        if omero is None:
            raise KeyError(f"Channel '{name}' not found. Image has no OMERO metadata")

        for i, channel in enumerate(omero.channels):
            if channel.label == name:
                return i

        raise KeyError(
            f"Channel '{name}' not found. Available channels: {self.channel_names}"
        )

    def get_channel(
        self,
        name: str,
        level: int = 0,
        as_type: Literal["dask", "numpy"] = "dask",
    ) -> Union[da.Array, np.ndarray]:
        """Get image data for a single channel, selected by name.

        Args:
            name: Channel name, matched against `omero.channels[i].label`.
            level: Multiscale pyramid level to read (0 = full resolution).
            as_type: Return a "dask" (lazy) or "numpy" (in-memory) array.

        Returns:
            Array with the channel ('c') axis removed.

        Raises:
            KeyError: If no channel with that name exists.
            ValueError: If the image has no channel ('c') axis.
        """
        if "c" not in self.dims:
            raise ValueError("Image has no channel ('c') axis")

        channel_index = self._channel_index(name)

        c_axis = self.dims.index("c")
        array = self._open_level_array(self._root, level)
        index: List[Union[slice, int]] = [slice(None)] * array.ndim
        index[c_axis] = channel_index
        channel_array = array[tuple(index)]

        return channel_array.compute() if as_type == "numpy" else channel_array

    # -- Labels -----------------------------------------------------------

    @property
    def label_names(self) -> List[str]:
        """Names of label images attached under the `labels/` subgroup."""
        if "labels" not in self._root:
            return []
        labels_group = self._root["labels"]
        return list(labels_group.attrs.get("ome", {}).get("labels", []))  # type: ignore[union-attr]

    def get_label(
        self,
        name: str,
        level: int = 0,
        as_type: Literal["dask", "numpy"] = "dask",
    ) -> Union[da.Array, np.ndarray]:
        """Get a label image by name.

        Args:
            name: Label name as listed in `label_names`.
            level: Multiscale pyramid level to read.
            as_type: Return a "dask" (lazy) or "numpy" (in-memory) array.

        Raises:
            KeyError: If no label with that name exists.
        """
        if name not in self.label_names:
            raise KeyError(
                f"Label '{name}' not found. Available labels: {self.label_names}"
            )

        label_group = self._root["labels"][name]  # type: ignore[index]
        array = self._open_level_array(label_group, level)

        return array.compute() if as_type == "numpy" else array

    # -- Metadata accessors -------------------------------------------------

    @property
    def dims(self) -> List[str]:
        """Axis names (e.g. ['t', 'c', 'z', 'y', 'x']) for the base image."""
        return [axis.name for axis in self._multiscale.axes]

    @property
    def axes(self) -> List[Axis]:
        """Axis metadata objects (name/type/unit) for the base image."""
        return self._multiscale.axes

    @property
    def n_levels(self) -> int:
        """Number of multiscale pyramid levels available."""
        return len(self._multiscale.datasets)

    def get_physical_size(self, level: int = 0) -> Dict[str, PhysicalSize]:
        """Get the physical pixel/voxel size for each spatial and time axis.

        Args:
            level: Multiscale pyramid level to compute sizes for.

        Returns:
            Mapping of axis name (e.g. "z", "y", "x", "t") to a `PhysicalSize`,
            for every non-channel axis.

        Raises:
            IndexError: If `level` is out of range.
        """
        dataset = self._multiscale.datasets[level]
        scale = next(
            t.scale
            for t in dataset.coordinateTransformations
            if hasattr(t, "scale")
        )

        return {
            axis.name: PhysicalSize(value=scale[i], unit=axis.unit)
            for i, axis in enumerate(self._multiscale.axes)
            if axis.type != "channel"
        }

    def get_voxel_size(self, level: int = 0) -> Dict[str, float]:
        """Get the spatial ('z', 'y', 'x') pixel/voxel size at the given level.

        Raises:
            ValueError: If a spatial axis has no unit assigned in metadata.
        """
        sizes = self.get_physical_size(level)
        voxel_size = {}
        for name in ("z", "y", "x"):
            if name not in sizes:
                continue
            if sizes[name].unit is None:
                raise ValueError(f"Axis '{name}' has no unit assigned in metadata")
            voxel_size[name] = sizes[name].value

        return voxel_size

    # -- Internal helpers ---------------------------------------------------

    def _open_level_array(self, group: zarr.Group, level: int) -> da.Array:
        """Open a resolution level as a lazy dask array."""
        zarr_array = group[str(level)]
        return da.from_array(zarr_array, chunks=zarr_array.chunks)  # type: ignore[union-attr]

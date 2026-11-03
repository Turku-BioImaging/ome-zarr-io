"""Reader module for OME-Zarr 0.5 images."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import dask.array as da
import numpy as np
import zarr

from .schema_models import (
    Axis,
    Multiscale,
    OMEZarrImageMetadata,
    ScaleTransformation,
    TranslationTransformation,
)
from .report import FilesetReport, validate as _validate_fileset
from .validator import OMEZarrValidator


@dataclass
class PhysicalSize:
    """Physical size of a single axis (e.g. pixel/voxel spacing)."""

    value: float
    unit: Optional[str]


_SPATIAL_NAMES = ("z", "y", "x")


@dataclass(frozen=True)
class LevelTransform:
    """The physical coordinate transform of one multiscale pyramid level.

    `scale` and `translation` are the final values: the dataset's own
    transformations followed by the multiscale-level ones, composed in order.
    `scale`, `translation` and `downsample_factor` are tuples aligned with `axes`.
    """

    level: int
    path: str
    axes: Tuple[str, ...]
    scale: Tuple[float, ...]
    translation: Tuple[float, ...]
    downsample_factor: Tuple[float, ...]
    _units: Tuple[Optional[str], ...]
    _axis_types: Tuple[Optional[str], ...]

    def _spatial_indices(self) -> List[int]:
        return [
            i
            for i, (name, typ) in enumerate(zip(self.axes, self._axis_types))
            if typ == "space" or (typ is None and name in _SPATIAL_NAMES)
        ]

    @property
    def physical_voxel_size(self) -> Dict[str, PhysicalSize]:
        """Size and unit of one voxel along each spatial axis.

        E.g. ``{"y": PhysicalSize(0.5, "micrometer"), "x": PhysicalSize(0.5, "micrometer")}``.
        Only spatial axes are included (never channel or time), so the result
        has two entries for 2D data and three for 3D data.
        """
        return {
            self.axes[i]: PhysicalSize(value=self.scale[i], unit=self._units[i])
            for i in self._spatial_indices()
        }

    @property
    def physical_pixel_size(self) -> Dict[str, PhysicalSize]:
        """Alias of `physical_voxel_size`, convenient for 2D data."""
        return self.physical_voxel_size

    @property
    def common_spatial_unit(self) -> Optional[str]:
        """The unit if all spatial axes share one, else None (absent or mixed).

        When this is None, read the per-axis units from `physical_voxel_size`.
        """
        units = {size.unit for size in self.physical_voxel_size.values()}
        return units.pop() if len(units) == 1 else None


def _compose(n_axes: int, transforms: list) -> Tuple[List[float], List[float]]:
    """Fold a list of scale/translation transforms into one (scale, translation)."""
    scale, translation = [1.0] * n_axes, [0.0] * n_axes
    for t in transforms:
        if isinstance(t, ScaleTransformation):
            values = t.scale
        elif isinstance(t, TranslationTransformation):
            values = t.translation
        else:
            continue
        if len(values) != n_axes:
            raise ValueError(
                f"A {t.type} transformation has {len(values)} values but the "
                f"image has {n_axes} axes."
            )
        if isinstance(t, ScaleTransformation):
            scale = [a * b for a, b in zip(scale, values)]
            translation = [a * b for a, b in zip(translation, values)]
        else:
            translation = [a + b for a, b in zip(translation, values)]
    return scale, translation


class Reader:
    """Reads and validates an OME-Zarr 0.5 image (a Zarr group with ``multiscales``)."""

    def __init__(self, path: Union[str, Path]):
        """Open an existing OME-Zarr image for reading.

        Args:
            path: Path to the image group (the directory containing zarr.json). For a
                plate, pass one of its fields, e.g. ``plate.zarr/A/1/0``.

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
            True if the image is valid.

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

    def report(self, strict: bool = False) -> FilesetReport:
        """Return a structured `FilesetReport` (validity plus a summary of the image).

        Unlike `validate()`, this never raises for invalid metadata; see
        `ome_zarr_io.validate` for details.
        """
        return _validate_fileset(self.path, strict=strict)

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
            level: Multiscale pyramid level to read (0 = full resolution; negative
                values count from the coarsest level).
            as_type: Return a "dask" (lazy) or "numpy" (in-memory) array.

        Returns:
            Array with the channel ('c') axis removed.

        Raises:
            KeyError: If no channel with that name exists, or the array listed for
                `level` is missing.
            ValueError: If the image has no channel ('c') axis.
            IndexError: If `level` is out of range.
        """
        if "c" not in self.dims:
            raise ValueError("Image has no channel ('c') axis")

        channel_index = self._channel_index(name)

        c_axis = self.dims.index("c")
        array = self._base_level_array(level)
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
            level: Multiscale pyramid level of the label to read (0 = full
                resolution; negative values count from the coarsest level).
            as_type: Return a "dask" (lazy) or "numpy" (in-memory) array.

        Raises:
            KeyError: If no label with that name exists, or the array listed for
                `level` is missing.
            ValueError: If the label has no valid multiscales metadata.
            IndexError: If `level` is out of range.
        """
        if name not in self.label_names:
            raise KeyError(
                f"Label '{name}' not found. Available labels: {self.label_names}"
            )

        label_group = self._root["labels"][name]  # type: ignore[index]
        # Label groups carry their own multiscales metadata; read the dataset
        # paths from it rather than assuming they match the base image.
        dataset_paths = self._label_dataset_paths(name, label_group)  # type: ignore[arg-type]
        array = self._open_level_array(label_group, dataset_paths, level)  # type: ignore[arg-type]

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

    @property
    def level_transforms(self) -> List[LevelTransform]:
        """The coordinate transform of every pyramid level, finest first."""
        return [self.get_level_transform(i) for i in range(self.n_levels)]

    def get_level_transform(self, level: int) -> LevelTransform:
        """Get the physical coordinate transform (metadata only) of one pyramid level.

        This does not read any image data; use `get_channel(..., level=)` for that.

        Args:
            level: Integer index of the pyramid level (0 is the full resolution).

        Returns:
            A `LevelTransform` with the composed scale and translation, the axis
            units, the voxel size and the downsampling factor relative to level 0.

        Raises:
            TypeError: If `level` is not an int.
            IndexError: If `level` is outside ``0..n_levels-1``.
        """
        if isinstance(level, bool) or not isinstance(level, int):
            raise TypeError(f"level must be an int, got {type(level).__name__}")
        if not 0 <= level < self.n_levels:
            raise IndexError(
                f"level={level} does not exist: the image has levels "
                f"0..{self.n_levels - 1}."
            )
        ms = self._multiscale
        n = len(ms.axes)
        extra = ms.coordinateTransformations or []

        def composed(index: int) -> Tuple[List[float], List[float]]:
            return _compose(n, [*ms.datasets[index].coordinateTransformations, *extra])

        scale, translation = composed(level)
        base_scale, _ = composed(0)
        return LevelTransform(
            level=level,
            path=ms.datasets[level].path,
            axes=tuple(a.name for a in ms.axes),
            scale=tuple(scale),
            translation=tuple(translation),
            downsample_factor=tuple(a / b for a, b in zip(scale, base_scale)),
            _units=tuple(a.unit for a in ms.axes),
            _axis_types=tuple(a.type for a in ms.axes),
        )

    def get_physical_size(self, level: int = 0) -> Dict[str, PhysicalSize]:
        """Get the physical pixel/voxel size for each spatial and time axis.

        See `get_level_transform` for a richer result that also has translation and the
        downsampling factor.

        Args:
            level: Multiscale pyramid level to compute sizes for.

        Returns:
            Mapping of axis name (e.g. "z", "y", "x", "t") to a `PhysicalSize`,
            for every non-channel axis.

        Raises:
            IndexError: If `level` is out of range.
        """
        scale = self.get_level_transform(level).scale

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

    def _base_level_array(self, level: int) -> da.Array:
        """Open a resolution level of the base image as a lazy dask array.

        The array is found through the `path` of the level's entry in the image's
        `multiscales[0].datasets`, so level arrays need not be named "0", "1", ...
        """
        dataset_paths = [dataset.path for dataset in self._multiscale.datasets]
        return self._open_level_array(self._root, dataset_paths, level)

    @staticmethod
    def _label_dataset_paths(name: str, label_group: zarr.Group) -> List[str]:
        """The level array paths listed in a label group's own multiscales metadata."""
        try:
            datasets = label_group.attrs["ome"]["multiscales"][0]["datasets"]  # type: ignore[index]
            return [dataset["path"] for dataset in datasets]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(
                f"Label '{name}' has no valid multiscales metadata, so its level "
                "arrays cannot be located"
            ) from e

    def _open_level_array(
        self, group: zarr.Group, dataset_paths: List[str], level: int
    ) -> da.Array:
        """Open a resolution level as a lazy dask array.

        Args:
            group: Multiscale group containing the level arrays.
            dataset_paths: `path` of each entry in the group's
                `multiscales[0].datasets`, in level order.
            level: Index into `dataset_paths` (negative values count from the end).

        Raises:
            IndexError: If `level` is out of range.
            KeyError: If the array listed for `level` does not exist.
        """
        n_levels = len(dataset_paths)
        if not -n_levels <= level < n_levels:
            raise IndexError(
                f"Level {level} is out of range: the image has {n_levels} level(s)"
            )
        path = dataset_paths[level]
        try:
            zarr_array = group[path]
        except KeyError:
            raise KeyError(
                f"Array '{path}' listed in the multiscales metadata for level "
                f"{level} does not exist"
            ) from None
        return da.from_array(zarr_array, chunks=zarr_array.chunks)  # type: ignore[union-attr]

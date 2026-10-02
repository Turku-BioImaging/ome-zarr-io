"""High-content-screening (plate) writer for OME-Zarr 0.5."""

import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import dask.array as da
import numpy as np
import zarr

from .validator import OMEZarrValidator
from .writer import Writer

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9]+$")


class PlateWriter:
    """Write an OME-Zarr plate: a tree of ``plate/<row>/<column>/<field>`` images.

    Rows and columns are declared up front because their order defines each
    well's ``rowIndex`` and ``columnIndex``. Every field is written with
    :class:`Writer`, and the well and plate metadata are rewritten after each
    ``add_field`` call, so an interrupted acquisition leaves a valid partial plate.

    The plate metadata is only written once the first field exists, since the
    schema requires at least one well.

    Example:
        >>> with PlateWriter("screen.ome.zarr", rows=["A", "B"], columns=["1", "2"]) as plate:
        ...     plate.add_field("A", "1", image, dims=["c", "y", "x"], axis_units=units)
    """

    def __init__(
        self,
        path: Union[str, Path],
        rows: Sequence[str],
        columns: Sequence[str],
        name: Optional[str] = None,
        acquisitions: Optional[List[Dict[str, Any]]] = None,
        overwrite: bool = False,
    ):
        """Create the plate's root group.

        Args:
            path: Where the plate will be written.
            rows: Row names in order, e.g. ``["A", "B"]``. Alphanumeric and unique.
            columns: Column names in order, e.g. ``["1", "2", "3"]``. Alphanumeric
                and unique.
            name: Optional plate name.
            acquisitions: Optional list of acquisition dicts as defined by the spec
                (``id`` required; ``name``, ``description``, ``maximumfieldcount``,
                ``starttime``, ``endtime`` optional). ``add_field(acquisition=...)``
                must refer to one of these ids.
            overwrite: Whether to replace an existing plate at ``path``.
        """
        self.path = Path(path)
        self.rows = self._check_names(rows, "row")
        self.columns = self._check_names(columns, "column")
        self.name = name
        self.acquisitions = self._check_acquisitions(acquisitions)
        self.overwrite = overwrite

        if self.overwrite and self.path.exists():
            if self.path.is_dir():
                shutil.rmtree(self.path)
            else:
                self.path.unlink()

        self._root = zarr.create_group(
            str(self.path), overwrite=self.overwrite, zarr_format=3
        )
        # (row, column) -> list of {"path": ..., "acquisition": ...} in insertion order
        self._wells: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    @staticmethod
    def _check_names(names: Sequence[str], kind: str) -> List[str]:
        names = list(names)
        if not names:
            raise ValueError(f"at least one {kind} is required")
        for n in names:
            if not isinstance(n, str) or not _NAME_PATTERN.match(n):
                raise ValueError(f"invalid {kind} name {n!r}: must match [A-Za-z0-9]+")
        if len(set(names)) != len(names):
            raise ValueError(f"{kind} names must be unique: {names}")
        return names

    @staticmethod
    def _check_acquisitions(
        acquisitions: Optional[List[Dict[str, Any]]],
    ) -> Optional[List[Dict[str, Any]]]:
        if acquisitions is None:
            return None
        ids = []
        for acq in acquisitions:
            if "id" not in acq:
                raise ValueError(f"acquisition {acq!r} is missing the required 'id'")
            ids.append(acq["id"])
        if len(set(ids)) != len(ids):
            raise ValueError(f"acquisition ids must be unique: {ids}")
        return [dict(a) for a in acquisitions]

    def add_field(
        self,
        row: str,
        column: str,
        image: Union[da.Array, np.ndarray],
        dims: List[str],
        axis_units: Any,
        *,
        field: Optional[int] = None,
        acquisition: Optional[int] = None,
        **writer_kwargs: Any,
    ) -> Writer:
        """Write one field of view into a well and update the plate metadata.

        Args:
            row: Row name; must have been declared in ``rows``.
            column: Column name; must have been declared in ``columns``.
            image: Dask or NumPy array for this field.
            dims: Dimension names, as for :class:`Writer`.
            axis_units: Axis units, as for :class:`Writer`.
            field: Field index within the well. Defaults to the next free index.
            acquisition: Acquisition id, which must be one of the plate's
                ``acquisitions`` when any were declared.
            **writer_kwargs: Any other :class:`Writer` argument (``downscale_levels``,
                ``channels``, ...). ``overwrite`` is not accepted; it is controlled
                by the plate.

        Returns:
            The :class:`Writer` that wrote the field, e.g. to call ``add_labels``.
        """
        if row not in self.rows:
            raise ValueError(f"unknown row {row!r}; declared rows: {self.rows}")
        if column not in self.columns:
            raise ValueError(
                f"unknown column {column!r}; declared columns: {self.columns}"
            )
        if "overwrite" in writer_kwargs:
            raise TypeError("overwrite is controlled by the PlateWriter")
        if acquisition is not None and self.acquisitions is not None:
            if acquisition not in {a["id"] for a in self.acquisitions}:
                raise ValueError(
                    f"unknown acquisition {acquisition}; declared ids: "
                    f"{[a['id'] for a in self.acquisitions]}"
                )

        images = self._wells.setdefault((row, column), [])
        used = {img["path"] for img in images}
        if field is None:
            field = len(images)
            while str(field) in used:
                field += 1
        if field < 0:
            raise ValueError("field must be a non-negative integer")
        if str(field) in used:
            raise ValueError(f"field {field} already exists in well {row}/{column}")

        well_group = self._root.require_group(row).require_group(column)
        field_path = self.path / row / column / str(field)
        writer = Writer(field_path, image, dims, axis_units, **writer_kwargs)
        writer.write()

        entry: Dict[str, Any] = {"path": str(field)}
        if acquisition is not None:
            entry["acquisition"] = acquisition
        images.append(entry)

        well_group.attrs.update(
            {"ome": {"version": "0.5", "well": {"images": list(images)}}}
        )
        self._write_plate_metadata()
        return writer

    def _write_plate_metadata(self) -> None:
        wells = [
            {
                "path": f"{row}/{column}",
                "rowIndex": self.rows.index(row),
                "columnIndex": self.columns.index(column),
            }
            for (row, column) in sorted(
                self._wells,
                key=lambda rc: (self.rows.index(rc[0]), self.columns.index(rc[1])),
            )
        ]
        plate: Dict[str, Any] = {
            "rows": [{"name": r} for r in self.rows],
            "columns": [{"name": c} for c in self.columns],
            "wells": wells,
            "field_count": max(len(v) for v in self._wells.values()),
        }
        if self.name is not None:
            plate["name"] = self.name
        if self.acquisitions is not None:
            plate["acquisitions"] = self.acquisitions
        self._root.attrs.update({"ome": {"version": "0.5", "plate": plate}})

    def close(self) -> None:
        """Validate the written plate and well metadata.

        Raises:
            ValueError: If no field was added (the plate would have no wells).
            jsonschema.exceptions.ValidationError: If any metadata is invalid.
        """
        if not self._wells:
            raise ValueError("plate has no fields; call add_field at least once")
        validator = OMEZarrValidator()
        validator.validate_plate_metadata(dict(self._root.attrs))
        for row, column in self._wells:
            well_group = self._root[row][column]
            validator.validate_well_metadata(dict(well_group.attrs))

    def __enter__(self) -> "PlateWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Don't mask an exception from the with-body with a validation error.
        if exc_type is None:
            self.close()

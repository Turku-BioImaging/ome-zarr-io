"""Structured validation reports for OME-Zarr 0.5 filesets."""

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import zarr
from jsonschema.exceptions import ValidationError

from .validator import OMEZarrValidator


@dataclass
class ValidationIssue:
    """A single problem found while validating a fileset."""

    path: str  # dotted location of the offending value, e.g. "ome.multiscales.0.axes"
    message: str
    # Which node the problem is in: "image", "plate", "well", "labels/<name>", or
    # the node's path inside a plate or well, e.g. "A/1", "A/1/0", "A/1/0/labels/cells".
    location: str = "image"

    def __str__(self) -> str:
        where = f"{self.location}: {self.path}" if self.path else self.location
        return f"{where}: {self.message}"


@dataclass
class LevelInfo:
    """One resolution level of a multiscale pyramid."""

    path: str
    shape: Optional[Tuple[int, ...]] = None
    chunks: Optional[Tuple[int, ...]] = None
    dtype: Optional[str] = None
    scale: Optional[List[float]] = None
    translation: Optional[List[float]] = None


@dataclass
class ChannelInfo:
    """A channel described by the OMERO metadata."""

    index: int
    label: Optional[str] = None
    color: Optional[str] = None
    window: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        extras = []
        if self.color:
            extras.append(f"#{self.color.lstrip('#')}")
        if self.window and {"start", "end"} <= self.window.keys():
            extras.append(f"window {self.window['start']}-{self.window['end']}")
        name = self.label if self.label is not None else f"<unnamed {self.index}>"
        return f"{self.index}: {name}" + (f" ({', '.join(extras)})" if extras else "")


@dataclass
class LabelInfo:
    """A label image attached under the `labels/` group."""

    name: str
    n_levels: int = 0
    dtype: Optional[str] = None
    has_colors: bool = False
    has_properties: bool = False
    source: Optional[str] = None

    def __str__(self) -> str:
        extras = [self.dtype or "unknown dtype", f"{self.n_levels} levels"]
        if self.has_colors:
            extras.append("colors")
        if self.has_properties:
            extras.append("properties")
        return f"{self.name} ({', '.join(extras)})"


@dataclass
class FilesetReport:
    """Validity of an OME-Zarr fileset plus a summary of what it contains."""

    path: str
    strict: bool = False
    kind: str = "image"  # "image", "plate", "well" or "bf2raw"
    spec_version: Optional[str] = None
    zarr_format: Optional[int] = None
    creator: Optional[str] = None
    axes: List[Dict[str, Any]] = field(default_factory=list)
    levels: List[LevelInfo] = field(default_factory=list)
    channels: List[ChannelInfo] = field(default_factory=list)
    labels: List[LabelInfo] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)  # plates, wells, bf2raw
    errors: List[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if no validation issues were found."""
        return not self.errors

    def __bool__(self) -> bool:
        return self.is_valid

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable representation of the report."""
        result = asdict(self)
        result["is_valid"] = self.is_valid
        return result

    def raise_if_invalid(self) -> None:
        """Raise `jsonschema.exceptions.ValidationError` if the fileset is invalid."""
        if not self.is_valid:
            details = "\n".join(f"  - {issue}" for issue in self.errors)
            raise ValidationError(
                f"'{self.path}' is not a valid OME-Zarr fileset:\n{details}"
            )

    def __str__(self) -> str:
        zarr_format = self.zarr_format if self.zarr_format else "unknown"
        lines = [
            f"path:      {self.path}",
            f"kind:      {self.kind}",
            f"valid:     {'yes' if self.is_valid else 'NO'}"
            f"{' (strict)' if self.strict else ''}",
            f"OME-Zarr:  {self.spec_version or 'unknown'} (zarr format {zarr_format})",
        ]
        if self.creator:
            lines.append(f"creator:   {self.creator}")
        for key, value in self.details.items():
            lines.append(f"{key + ':':10} {value}")
        if self.axes:
            axes = ", ".join(
                a["name"] + (f" [{a['unit']}]" if a.get("unit") else "")
                for a in self.axes
            )
            lines.append(f"axes:      {axes}")
        if self.levels:
            lines.append(f"levels:    {len(self.levels)}")
            for i, level in enumerate(self.levels):
                lines.append(
                    f"  {i}: path={level.path} shape={level.shape}"
                    f" chunks={level.chunks} dtype={level.dtype}"
                )
        if self.channels:
            lines.append(f"channels:  {len(self.channels)}")
            lines.extend(f"  {c}" for c in self.channels)
        if self.labels:
            lines.append(f"labels:    {len(self.labels)}")
            lines.extend(f"  {label}" for label in self.labels)
        if self.errors:
            lines.append(f"problems:  {len(self.errors)}")
            lines.extend(f"  - {issue}" for issue in self.errors)
        return "\n".join(lines)


_PATH_SEGMENT = re.compile(r"[A-Za-z0-9]+")
_WELL_PATH = re.compile(r"[A-Za-z0-9]+/[A-Za-z0-9]+")


def _dotted(path: Sequence[Any], prefix: str = "") -> str:
    parts = [prefix] if prefix else []
    parts.extend(str(p) for p in path)
    return ".".join(parts)


def _kind(ome: Any) -> str:
    """Which kind of OME-Zarr node the root group describes."""
    if isinstance(ome, dict):
        for key, kind in (
            ("plate", "plate"),
            ("well", "well"),
            ("bioformats2raw.layout", "bf2raw"),
        ):
            if key in ome:
                return kind
    return "image"


def _describe_node(kind: str, ome: Any) -> Dict[str, Any]:
    """Short facts about a plate, well or bioformats2raw root."""
    if not isinstance(ome, dict):
        return {}
    if kind == "plate" and isinstance(ome.get("plate"), dict):
        plate = ome["plate"]
        rows, columns = plate.get("rows"), plate.get("columns")
        details: Dict[str, Any] = {}
        if plate.get("name"):
            details["name"] = plate["name"]
        if isinstance(rows, list) and isinstance(columns, list):
            details["layout"] = f"{len(rows)} rows x {len(columns)} columns"
        if isinstance(plate.get("wells"), list):
            details["wells"] = len(plate["wells"])
        if plate.get("field_count") is not None:
            details["fields"] = plate["field_count"]
        return details
    if kind == "well" and isinstance(ome.get("well"), dict):
        images = ome["well"].get("images")
        return {"images": len(images)} if isinstance(images, list) else {}
    if kind == "bf2raw":
        return {"layout": f"bioformats2raw v{ome.get('bioformats2raw.layout')}"}
    return {}


def _first_multiscale(ome: Any) -> Dict[str, Any]:
    if isinstance(ome, dict):
        multiscales = ome.get("multiscales")
        if isinstance(multiscales, list) and multiscales:
            if isinstance(multiscales[0], dict):
                return multiscales[0]
    return {}


def _transformation(transformations: Any, kind: str) -> Optional[List[float]]:
    if isinstance(transformations, list):
        for t in transformations:
            if isinstance(t, dict) and t.get("type") == kind:
                value = t.get(kind)
                if isinstance(value, list):
                    return value
    return None


def _describe_levels(
    group: zarr.Group, multiscale: Dict[str, Any], location: str
) -> Tuple[List[LevelInfo], List[ValidationIssue]]:
    """Summarize pyramid levels, checking that listed arrays exist on disk."""
    levels: List[LevelInfo] = []
    issues: List[ValidationIssue] = []
    n_axes = (
        len(multiscale["axes"]) if isinstance(multiscale.get("axes"), list) else None
    )

    datasets = multiscale.get("datasets")
    if not isinstance(datasets, list):
        return levels, issues

    for i, dataset in enumerate(datasets):
        if not isinstance(dataset, dict) or not isinstance(dataset.get("path"), str):
            continue  # reported by schema validation
        transformations = dataset.get("coordinateTransformations")
        level = LevelInfo(
            path=dataset["path"],
            scale=_transformation(transformations, "scale"),
            translation=_transformation(transformations, "translation"),
        )
        item_path = f"ome.multiscales.0.datasets.{i}.path"
        ct_path = f"ome.multiscales.0.datasets.{i}.coordinateTransformations"
        for kind, values in (
            ("scale", level.scale),
            ("translation", level.translation),
        ):
            if values is None:
                continue
            if n_axes is not None and len(values) != n_axes:
                issues.append(
                    ValidationIssue(
                        ct_path,
                        f"{kind} has {len(values)} values but {n_axes} axes are declared",
                        location,
                    )
                )
            if kind == "scale" and any(
                isinstance(v, (int, float)) and v <= 0 for v in values
            ):
                issues.append(
                    ValidationIssue(ct_path, "scale values must be positive", location)
                )
        try:
            array = group[dataset["path"]]
        except KeyError:
            issues.append(
                ValidationIssue(
                    item_path,
                    f"array '{dataset['path']}' listed in datasets does not exist",
                    location,
                )
            )
        else:
            if isinstance(array, zarr.Array):
                level.shape = tuple(array.shape)
                level.chunks = tuple(array.chunks)
                level.dtype = str(array.dtype)
                if n_axes is not None and n_axes != array.ndim:
                    issues.append(
                        ValidationIssue(
                            item_path,
                            f"array has {array.ndim} dimensions but {n_axes} axes "
                            "are declared",
                            location,
                        )
                    )
            else:
                issues.append(
                    ValidationIssue(
                        item_path,
                        f"'{dataset['path']}' is a group, not an array",
                        location,
                    )
                )
        levels.append(level)

    return levels, issues


def _describe_channels(ome: Any) -> List[ChannelInfo]:
    omero = ome.get("omero") if isinstance(ome, dict) else None
    channels = omero.get("channels") if isinstance(omero, dict) else None
    if not isinstance(channels, list):
        return []
    return [
        ChannelInfo(
            index=i,
            label=ch.get("label"),
            color=ch.get("color"),
            window=ch.get("window"),
        )
        for i, ch in enumerate(channels)
        if isinstance(ch, dict)
    ]


def _check_channels(
    ome: Any, axes: List[Dict[str, Any]], levels: Any, location: str = "image"
) -> List[ValidationIssue]:
    """Semantic OMERO checks the JSON schema does not express."""
    omero = ome.get("omero") if isinstance(ome, dict) else None
    channels = omero.get("channels") if isinstance(omero, dict) else None
    if not isinstance(channels, list):
        return []
    issues: List[ValidationIssue] = []

    def add(path: str, message: str) -> None:
        issues.append(ValidationIssue(f"ome.omero.{path}", message, location))

    names = [a.get("name") for a in axes]
    if "c" not in names:
        add("channels", "omero channels are present but there is no 'c' axis")
    else:
        shape = getattr(levels[0], "shape", None) if levels else None
        c_index = names.index("c")
        if (
            shape is not None
            and c_index < len(shape)
            and shape[c_index] != len(channels)
        ):
            add(
                "channels",
                f"{len(channels)} channels described but the 'c' axis has "
                f"{shape[c_index]} entries",
            )

    labels = [ch.get("label") for ch in channels if isinstance(ch, dict)]
    labels = [x for x in labels if x is not None]
    for dup in sorted({x for x in labels if labels.count(x) > 1}, key=str):
        add("channels", f"duplicate channel label '{dup}'")

    for i, ch in enumerate(channels):
        if not isinstance(ch, dict):
            continue
        color = ch.get("color")
        if isinstance(color, str) and not re.fullmatch(r"[0-9A-Fa-f]{6}", color):
            add(
                f"channels.{i}.color",
                f"color '{color}' is not 6 hex digits without '#'",
            )
        w = ch.get("window")
        if isinstance(w, dict):
            try:
                ok = w["min"] <= w["start"] <= w["end"] <= w["max"]
            except (KeyError, TypeError):
                continue
            if not ok:
                add(
                    f"channels.{i}.window",
                    "window must satisfy min <= start <= end <= max",
                )
    return issues


def _describe_label(
    name: str,
    label_group: zarr.Group,
    validator: OMEZarrValidator,
    strict: bool,
    prefix: str = "",
) -> Tuple[LabelInfo, List[ValidationIssue]]:
    location = _join(prefix, f"labels/{name}")
    attrs = dict(label_group.attrs)
    ome = attrs.get("ome")
    issues = [
        ValidationIssue(_dotted(path), message, location)
        for path, message in validator.iter_issues(
            attrs, "strict_label" if strict else "label"
        )
    ]

    multiscale = _first_multiscale(ome)
    if not multiscale and not any(i.path.startswith("ome.multiscales") for i in issues):
        issues.append(
            ValidationIssue(
                "ome.multiscales", "label image has no multiscales", location
            )
        )
    levels, level_issues = _describe_levels(label_group, multiscale, location)
    issues.extend(level_issues)

    image_label = ome.get("image-label") if isinstance(ome, dict) else None
    image_label = image_label if isinstance(image_label, dict) else {}
    source = image_label.get("source")
    label = LabelInfo(
        name=name,
        n_levels=len(levels),
        dtype=levels[0].dtype if levels else None,
        has_colors=bool(image_label.get("colors")),
        has_properties=bool(image_label.get("properties")),
        source=source.get("image") if isinstance(source, dict) else None,
    )

    if levels and levels[0].dtype is not None:
        if levels[0].dtype.startswith("float") or levels[0].dtype in (
            "bool",
            "complex128",
        ):
            issues.append(
                ValidationIssue(
                    "ome.multiscales.0.datasets.0.path",
                    f"label arrays must have an integer data type, got {levels[0].dtype}",
                    location,
                )
            )

    return label, issues


def _join(prefix: str, path: str) -> str:
    return f"{prefix}/{path}" if prefix else path


def _child_group(group: zarr.Group, path: str) -> Optional[zarr.Group]:
    try:
        node = group[path]
    except KeyError:
        return None
    return node if isinstance(node, zarr.Group) else None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _version_issues(ome: Any, expected: Optional[str], location: str) -> List[ValidationIssue]:
    """NGFF 0.5: the OME-Zarr version MUST be consistent within a hierarchy."""
    if expected is None or not isinstance(ome, dict):
        return []
    version = ome.get("version")
    if isinstance(version, str) and version != expected:
        return [
            ValidationIssue(
                "ome.version",
                f"version '{version}' differs from the root's '{expected}'; the "
                "OME-Zarr version must be consistent within a hierarchy",
                location,
            )
        ]
    return []


def _inspect_image(
    group: zarr.Group,
    ome: Any,
    validator: OMEZarrValidator,
    strict: bool,
    prefix: str = "",
) -> Tuple[
    List[Dict[str, Any]],
    List[LevelInfo],
    List[ChannelInfo],
    List[LabelInfo],
    List[ValidationIssue],
]:
    """Describe an image group and run the checks the image schema cannot express.

    `prefix` is the image's path inside a plate or well ("" for a root image); it
    is used to say where each issue was found.
    """
    location = prefix or "image"
    issues: List[ValidationIssue] = []

    multiscale = _first_multiscale(ome)
    raw_axes = multiscale.get("axes")
    axes = [a for a in raw_axes if isinstance(a, dict)] if isinstance(raw_axes, list) else []
    levels, level_issues = _describe_levels(group, multiscale, location)
    issues.extend(level_issues)
    channels = _describe_channels(ome)
    issues.extend(_check_channels(ome, axes, levels, location))

    labels: List[LabelInfo] = []
    if "labels" in group:
        labels_group = group["labels"]
        labels_ome = dict(labels_group.attrs).get("ome")
        listed = labels_ome.get("labels") if isinstance(labels_ome, dict) else None
        label_names = [n for n in listed or [] if isinstance(n, str)]
        for name in label_names:
            label_group = (
                labels_group.get(name) if isinstance(labels_group, zarr.Group) else None
            )
            if not isinstance(label_group, zarr.Group):
                issues.append(
                    ValidationIssue(
                        "ome.labels",
                        f"label '{name}' is listed but no such group exists",
                        _join(prefix, "labels"),
                    )
                )
                continue
            label, label_issues = _describe_label(
                name, label_group, validator, strict, prefix
            )
            labels.append(label)
            issues.extend(label_issues)

    return axes, levels, channels, labels, issues


def _check_well(
    well_group: zarr.Group,
    ome: Any,
    validator: OMEZarrValidator,
    strict: bool,
    version: Optional[str],
    prefix: str = "",
    acquisition_ids: Optional[List[Any]] = None,
) -> List[ValidationIssue]:
    """Check a well's fields of view: each listed image must exist and be a valid image.

    `acquisition_ids` are the ids declared by the enclosing plate; pass None when the
    well is validated on its own, in which case acquisitions cannot be checked.
    """
    location = prefix or "well"
    issues: List[ValidationIssue] = []
    well = ome.get("well") if isinstance(ome, dict) else None
    images = well.get("images") if isinstance(well, dict) else None
    if not isinstance(images, list):
        return issues  # reported by schema validation

    seen: set = set()
    for i, image in enumerate(images):
        if not isinstance(image, dict) or not isinstance(image.get("path"), str):
            continue  # reported by schema validation
        path = image["path"]
        item = f"ome.well.images.{i}"
        if path in seen:
            issues.append(
                ValidationIssue(f"{item}.path", f"duplicate image path '{path}'", location)
            )
            continue
        seen.add(path)

        if acquisition_ids is not None:
            acquisition = image.get("acquisition")
            if acquisition is None:
                if len(acquisition_ids) > 1:
                    issues.append(
                        ValidationIssue(
                            item,
                            "'acquisition' is required because the plate defines "
                            "several acquisitions",
                            location,
                        )
                    )
            elif acquisition not in acquisition_ids:
                issues.append(
                    ValidationIssue(
                        f"{item}.acquisition",
                        f"acquisition {acquisition!r} is not defined in the plate metadata",
                        location,
                    )
                )

        if not _PATH_SEGMENT.fullmatch(path):
            continue  # reported by schema validation; do not follow odd paths
        field_group = _child_group(well_group, path)
        field_location = _join(prefix, path)
        if field_group is None:
            issues.append(
                ValidationIssue(
                    f"{item}.path",
                    f"image '{path}' is listed but no such group exists",
                    location,
                )
            )
            continue

        field_attrs = dict(field_group.attrs)
        field_ome = field_attrs.get("ome")
        issues.extend(
            ValidationIssue(_dotted(p), message, field_location)
            for p, message in validator.iter_issues(
                field_attrs, "strict_image" if strict else "image"
            )
        )
        issues.extend(_version_issues(field_ome, version, field_location))
        issues.extend(
            _inspect_image(field_group, field_ome, validator, strict, field_location)[4]
        )
    return issues


def _names(items: Any) -> Optional[List[Any]]:
    """The `name` of each row/column object, or None if the list is malformed."""
    if not isinstance(items, list):
        return None
    return [item.get("name") if isinstance(item, dict) else None for item in items]


def _check_plate(
    root: zarr.Group,
    ome: Any,
    validator: OMEZarrValidator,
    strict: bool,
    version: Optional[str],
) -> List[ValidationIssue]:
    """Plate rules the schema cannot express, then every listed well and its fields."""
    issues: List[ValidationIssue] = []
    plate = ome.get("plate") if isinstance(ome, dict) else None
    if not isinstance(plate, dict):
        return issues  # reported by schema validation

    def add(path: str, message: str) -> None:
        issues.append(ValidationIssue(f"ome.plate.{path}", message, "plate"))

    rows, columns = _names(plate.get("rows")), _names(plate.get("columns"))
    for key, names in (("rows", rows), ("columns", columns)):
        strings = [n for n in names or [] if isinstance(n, str)]
        for dup in sorted({n for n in strings if strings.count(n) > 1}):
            add(key, f"duplicate {key[:-1]} name '{dup}'")

    acquisitions = plate.get("acquisitions")
    acquisition_ids = (
        [a.get("id") for a in acquisitions if isinstance(a, dict)]
        if isinstance(acquisitions, list)
        else []
    )
    for dup in sorted({i for i in acquisition_ids if _is_int(i) and acquisition_ids.count(i) > 1}):
        add("acquisitions", f"duplicate acquisition id {dup}")

    wells = plate.get("wells")
    seen: set = set()
    for i, well in enumerate(wells if isinstance(wells, list) else []):
        if not isinstance(well, dict) or not isinstance(well.get("path"), str):
            continue  # reported by schema validation
        path = well["path"]
        if path in seen:
            add(f"wells.{i}.path", f"duplicate well path '{path}'")
            continue
        seen.add(path)
        if not _WELL_PATH.fullmatch(path):
            continue  # reported by schema validation; do not follow odd paths
        row_name, column_name = path.split("/")

        # rowIndex, columnIndex and path MUST all refer to the same row/column pair.
        for key, names, name, kind in (
            ("rowIndex", rows, row_name, "row"),
            ("columnIndex", columns, column_name, "column"),
        ):
            index = well.get(key)
            if names is None or not _is_int(index):
                continue  # reported by schema validation
            if not 0 <= index < len(names):
                add(
                    f"wells.{i}.{key}",
                    f"{key} {index} is out of range: the plate has {len(names)} {kind}s",
                )
            elif names[index] != name:
                add(
                    f"wells.{i}.{key}",
                    f"{key} {index} refers to {kind} '{names[index]}' but the path "
                    f"'{path}' names {kind} '{name}'",
                )

        well_group = _child_group(root, path)
        if well_group is None:
            add(f"wells.{i}.path", f"well '{path}' is listed but no such group exists")
            continue
        well_attrs = dict(well_group.attrs)
        well_ome = well_attrs.get("ome")
        issues.extend(
            ValidationIssue(_dotted(p), message, path)
            for p, message in validator.iter_issues(
                well_attrs, "strict_well" if strict else "well"
            )
        )
        issues.extend(_version_issues(well_ome, version, path))
        issues.extend(
            _check_well(
                well_group, well_ome, validator, strict, version, path, acquisition_ids
            )
        )
    return issues


def validate(path: Union[str, Path], strict: bool = False) -> FilesetReport:
    """Validate an OME-Zarr 0.5 fileset and summarize what it contains.

    Unlike `Reader`, this never fails on malformed metadata: every problem found is
    recorded in the returned report's `errors` instead of being raised.

    A plate is validated as a whole: its own metadata, every well it lists and every
    field of view (and label image) in those wells. A well is validated together with
    its fields. Each issue's `location` says which node it was found in.

    Args:
        path: Path (or URL) to the root OME-Zarr group. Images, plates, wells and
            bioformats2raw layouts are recognized; only a root image is described in
            detail (axes, levels, channels, labels).
        strict: If True, validate against the `strict_*` schemas.

    Returns:
        A `FilesetReport`. `bool(report)` is True when the fileset is valid.

    Raises:
        FileNotFoundError: If `path` does not exist (for URLs: if no Zarr group is
            found there).
    """
    location_str = str(path)
    if "://" not in location_str and not Path(location_str).exists():
        raise FileNotFoundError(f"No OME-Zarr group found at '{location_str}'")

    report = FilesetReport(path=location_str, strict=strict)

    try:
        root = zarr.open_group(location_str, mode="r")
    except zarr.errors.GroupNotFoundError:
        if "://" in location_str:  # e.g. a mistyped URL, not a malformed fileset
            raise FileNotFoundError(
                f"No OME-Zarr group found at '{location_str}'"
            ) from None
        report.errors.append(
            ValidationIssue("", "not a Zarr group (no zarr.json found)")
        )
        return report

    report.zarr_format = root.metadata.zarr_format
    if report.zarr_format != 3:
        report.errors.append(
            ValidationIssue(
                "",
                f"Zarr format {report.zarr_format} is not supported; "
                "OME-Zarr 0.5 requires Zarr format 3",
            )
        )

    validator = OMEZarrValidator()
    attrs = dict(root.attrs)
    ome = attrs.get("ome")

    if isinstance(ome, dict):
        version = ome.get("version")
        report.spec_version = version if isinstance(version, str) else None
        creator = ome.get("_creator")
        if isinstance(creator, dict) and creator.get("name"):
            report.creator = str(creator["name"])
    if report.spec_version and report.spec_version != validator.schema_version:
        report.errors.append(
            ValidationIssue(
                "ome.version",
                f"fileset declares OME-Zarr {report.spec_version}, "
                f"but this validator checks {validator.schema_version}",
            )
        )

    report.kind = _kind(ome)
    schema = (
        f"strict_{report.kind}" if strict and report.kind != "bf2raw" else report.kind
    )
    root_location = report.kind if report.kind in ("plate", "well") else "image"
    report.errors.extend(
        ValidationIssue(_dotted(p), message, root_location)
        for p, message in validator.iter_issues(attrs, schema)
    )
    if report.kind != "image":
        report.details = _describe_node(report.kind, ome)
        if report.kind == "plate":
            report.errors.extend(
                _check_plate(root, ome, validator, strict, report.spec_version)
            )
        elif report.kind == "well":
            report.errors.extend(
                _check_well(root, ome, validator, strict, report.spec_version)
            )
        return report  # axes, levels, channels and labels describe a root image only

    report.axes, report.levels, report.channels, report.labels, issues = _inspect_image(
        root, ome, validator, strict
    )
    report.errors.extend(issues)
    return report

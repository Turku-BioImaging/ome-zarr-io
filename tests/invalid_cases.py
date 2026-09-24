"""Deliberately broken OME-Zarr metadata, shared by the validation tests.

Each case is `(id, mutate, expected)`: `mutate` edits a valid `zarr.json` attributes
dict in place, and `expected` must appear in "<path> <message>" of at least one
reported issue, so a test fails if a case is rejected for an unintended reason.

Cases are grouped by which layer catches them:
  * SCHEMA_*   - rejected by the bundled JSON schema (`OMEZarrValidator`).
  * SEMANTIC_* - pass the JSON schema but are still invalid; only `validate()` /
                 `Reader.report()` catch them.
"""

import json
import shutil
from pathlib import Path


def _ms(attrs):
    return attrs["ome"]["multiscales"][0]


def _ct(attrs):
    return _ms(attrs)["datasets"][0]["coordinateTransformations"][0]


SCHEMA_IMAGE_CASES = [
    ("no_ome", lambda a: a.pop("ome"), "'ome' is a required property"),
    (
        "no_version",
        lambda a: a["ome"].pop("version"),
        "'version' is a required property",
    ),
    ("old_version", lambda a: a["ome"].update(version="0.4"), "ome.version"),
    ("empty_multiscales", lambda a: a["ome"].update(multiscales=[]), "ome.multiscales"),
    (
        "multiscales_not_list",
        lambda a: a["ome"].update(multiscales={}),
        "ome.multiscales",
    ),
    ("no_axes", lambda a: _ms(a).pop("axes"), "'axes' is a required property"),
    (
        "bad_axis_type",
        lambda a: _ms(a)["axes"][1].update(type="bogus"),
        "ome.multiscales.0.axes",
    ),
    (
        "axis_no_name",
        lambda a: _ms(a)["axes"][1].pop("name"),
        "ome.multiscales.0.axes.1",
    ),
    (
        "single_axis",
        lambda a: _ms(a).update(axes=_ms(a)["axes"][:1]),
        "ome.multiscales.0.axes",
    ),
    (
        "six_axes",
        lambda a: _ms(a).update(axes=_ms(a)["axes"] * 2),
        "ome.multiscales.0.axes",
    ),
    (
        "duplicate_axis_names",
        lambda a: _ms(a)["axes"][2].update(name="y"),
        "ome.multiscales.0.axes",
    ),
    ("no_datasets", lambda a: _ms(a).update(datasets=[]), "ome.multiscales.0.datasets"),
    (
        "dataset_no_path",
        lambda a: _ms(a)["datasets"][0].pop("path"),
        "'path' is a required property",
    ),
    (
        "dataset_path_not_string",
        lambda a: _ms(a)["datasets"][0].update(path=0),
        "datasets.0.path",
    ),
    (
        "dataset_no_transformations",
        lambda a: _ms(a)["datasets"][0].pop("coordinateTransformations"),
        "'coordinateTransformations' is a required property",
    ),
    (
        "unknown_transformation_type",
        lambda a: _ct(a).update(type="rotate"),
        "coordinateTransformations",
    ),
    (
        "scale_contains_string",
        lambda a: _ct(a).update(scale=[1.0, "a", 0.5]),
        "coordinateTransformations",
    ),
    (
        "omero_window_not_object",
        lambda a: a["ome"]["omero"]["channels"][0].update(window="x"),
        "ome.omero.channels.0.window",
    ),
    (
        "omero_channels_not_list",
        lambda a: a["ome"]["omero"].update(channels="x"),
        "ome.omero.channels",
    ),
]

SEMANTIC_IMAGE_CASES = [
    (
        "scale_too_short",
        lambda a: _ct(a).update(scale=[1.0, 0.5]),
        "scale has 2 values but 3 axes",
    ),
    (
        "scale_zero",
        lambda a: _ct(a).update(scale=[1.0, 0.0, 0.5]),
        "scale values must be positive",
    ),
    (
        "scale_negative",
        lambda a: _ct(a).update(scale=[1.0, -0.5, 0.5]),
        "scale values must be positive",
    ),
]

SCHEMA_LABEL_CASES = [
    (
        "label_no_image_label",
        lambda a: a["ome"].pop("image-label"),
        "'image-label' is a required property",
    ),
    ("label_old_version", lambda a: a["ome"].update(version="0.4"), "ome.version"),
    (
        "label_color_no_value",
        lambda a: a["ome"]["image-label"]["colors"][0].pop("label-value"),
        "'label-value' is a required property",
    ),
    (
        "label_color_short_rgba",
        lambda a: a["ome"]["image-label"]["colors"][0].update(rgba=[1, 2]),
        "rgba",
    ),
    (
        "label_color_rgba_out_of_range",
        lambda a: a["ome"]["image-label"]["colors"][0].update(rgba=[300, 0, 0, 255]),
        "rgba",
    ),
    (
        "label_source_not_object",
        lambda a: a["ome"]["image-label"].update(source="x"),
        "ome.image-label.source",
    ),
    (
        "label_properties_not_list",
        lambda a: a["ome"]["image-label"].update(properties="x"),
        "ome.image-label.properties",
    ),
]

SEMANTIC_LABEL_CASES = [
    (
        "label_no_multiscales",
        lambda a: a["ome"].pop("multiscales"),
        "label image has no multiscales",
    ),
]


def read_attrs(node_dir: Path) -> dict:
    """Read the `attributes` of a zarr v3 node's `zarr.json`."""
    return json.loads((Path(node_dir) / "zarr.json").read_text())["attributes"]


def rewrite_attrs(node_dir: Path, mutate) -> None:
    """Apply `mutate` to a zarr v3 node's attributes and write them back."""
    zarr_json = Path(node_dir) / "zarr.json"
    document = json.loads(zarr_json.read_text())
    mutate(document["attributes"])
    zarr_json.write_text(json.dumps(document))


def make_broken_copy(
    valid_fileset: Path, dest: Path, mutate, label: str = None
) -> Path:
    """Copy `valid_fileset` to `dest` and apply `mutate` to the image (or a label) attrs."""
    shutil.copytree(valid_fileset, dest)
    target = dest if label is None else dest / "labels" / label
    rewrite_attrs(target, mutate)
    return dest

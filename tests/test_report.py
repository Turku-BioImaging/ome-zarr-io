"""Tests for `validate()` and `FilesetReport`."""

import json
import shutil

import numpy as np
import pytest
import zarr
from jsonschema.exceptions import ValidationError

from invalid_cases import (
    SCHEMA_IMAGE_CASES,
    SCHEMA_LABEL_CASES,
    SEMANTIC_IMAGE_CASES,
    SEMANTIC_LABEL_CASES,
    make_broken_copy,
    rewrite_attrs,
)
from ome_zarr_io import Reader, Writer, validate

IMAGE_CASES = SCHEMA_IMAGE_CASES + SEMANTIC_IMAGE_CASES
LABEL_CASES = SCHEMA_LABEL_CASES + SEMANTIC_LABEL_CASES


def issue_texts(report):
    return [f"{i.path} {i.message}" for i in report.errors]


class TestValidFileset:
    def test_is_valid(self, valid_fileset):
        report = validate(valid_fileset)
        assert report.is_valid
        assert bool(report)
        assert report.errors == []
        report.raise_if_invalid()  # does not raise

    def test_summary(self, valid_fileset):
        report = validate(valid_fileset)
        assert report.kind == "image"
        assert report.spec_version == "0.5"
        assert report.zarr_format == 3
        assert [a["name"] for a in report.axes] == ["c", "y", "x"]
        assert [(lv.path, lv.shape) for lv in report.levels] == [
            ("0", (2, 20, 20)),
            ("1", (2, 10, 10)),
        ]
        assert report.levels[0].dtype == "uint8"
        assert report.levels[0].scale == [1.0, 0.5, 0.5]
        assert [(c.index, c.label, c.color) for c in report.channels] == [
            (0, "DAPI", "0000FF"),
            (1, "GFP", "00FF00"),
        ]
        assert report.channels[0].window == {
            "start": 0,
            "min": 0,
            "end": 255,
            "max": 255,
        }
        (label,) = report.labels
        assert (label.name, label.n_levels, label.dtype) == ("nuclei", 2, "uint8")
        assert label.has_colors and not label.has_properties
        assert label.source == "../../"

    def test_no_labels_or_omero(self, tmp_path):
        path = tmp_path / "plain.zarr"
        Writer(
            path=path,
            image=np.zeros((20, 20), dtype=np.uint8),
            dims=["y", "x"],
            axis_units={"y": "micrometer", "x": "micrometer"},
        ).write()
        report = validate(path)
        assert report.is_valid
        assert report.labels == [] and report.channels == []

    def test_accepts_str_and_path(self, valid_fileset):
        assert validate(str(valid_fileset)).is_valid

    def test_reader_report_matches_validate(self, valid_fileset):
        assert (
            Reader(valid_fileset).report().to_dict()
            == validate(valid_fileset).to_dict()
        )

    def test_to_dict_is_json_serializable(self, valid_fileset):
        data = json.loads(json.dumps(validate(valid_fileset).to_dict()))
        assert data["is_valid"] is True
        assert data["labels"][0]["name"] == "nuclei"

    def test_str_summary(self, valid_fileset):
        text = str(validate(valid_fileset))
        for expected in (
            "kind:      image",
            "valid:     yes",
            "OME-Zarr:  0.5 (zarr format 3)",
            "axes:      c, y [micrometer], x [micrometer]",
            "0: path=0 shape=(2, 20, 20)",
            "channels:  2",
            "0: DAPI (#0000FF, window 0-255)",
            "1: GFP (#00FF00)",
            "labels:    1",
            "nuclei (uint8, 2 levels, colors)",
        ):
            assert expected in text, text
        assert "problems" not in text

    def test_str_lists_problems(self, valid_fileset, tmp_path):
        broken = make_broken_copy(
            valid_fileset,
            tmp_path / "broken.zarr",
            lambda a: a["ome"].pop("image-label"),
            label="nuclei",
        )
        text = str(validate(broken))
        assert "valid:     NO" in text
        assert "problems:  1" in text
        assert "- labels/nuclei: ome: 'image-label' is a required property" in text


class TestInvalidImageMetadata:
    @pytest.mark.parametrize(
        "case_id,mutate,expected", IMAGE_CASES, ids=[c[0] for c in IMAGE_CASES]
    )
    def test_reported_not_raised(
        self, valid_fileset, tmp_path, case_id, mutate, expected
    ):
        broken = make_broken_copy(valid_fileset, tmp_path / "broken.zarr", mutate)
        report = validate(broken)

        assert report.is_valid is False
        assert not report
        assert any(expected in text for text in issue_texts(report)), issue_texts(
            report
        )
        assert all(i.location == "image" for i in report.errors)
        with pytest.raises(ValidationError):
            report.raise_if_invalid()

    def test_reader_cannot_open_but_validate_reports(self, valid_fileset, tmp_path):
        """`Reader.__init__` parses metadata eagerly, so it fails where `validate` reports."""
        broken = make_broken_copy(
            valid_fileset,
            tmp_path / "broken.zarr",
            lambda a: a["ome"]["multiscales"][0]["datasets"][0].pop(
                "coordinateTransformations"
            ),
        )
        with pytest.raises(KeyError):
            Reader(broken)
        assert validate(broken).is_valid is False

    def test_wrong_spec_version_is_its_own_issue(self, valid_fileset, tmp_path):
        broken = make_broken_copy(
            valid_fileset,
            tmp_path / "broken.zarr",
            lambda a: a["ome"].update(version="0.4"),
        )
        report = validate(broken)
        assert report.spec_version == "0.4"
        assert any("declares OME-Zarr 0.4" in i.message for i in report.errors)


class TestInvalidLabelMetadata:
    @pytest.mark.parametrize(
        "case_id,mutate,expected", LABEL_CASES, ids=[c[0] for c in LABEL_CASES]
    )
    def test_reported_with_label_location(
        self, valid_fileset, tmp_path, case_id, mutate, expected
    ):
        broken = make_broken_copy(
            valid_fileset, tmp_path / "broken.zarr", mutate, label="nuclei"
        )
        report = validate(broken)

        assert report.is_valid is False
        assert any(expected in text for text in issue_texts(report)), issue_texts(
            report
        )
        assert {i.location for i in report.errors} == {"labels/nuclei"}

    def test_listed_label_missing_on_disk(self, valid_fileset, tmp_path):
        broken = tmp_path / "broken.zarr"
        shutil.copytree(valid_fileset, broken)
        rewrite_attrs(broken / "labels", lambda a: a["ome"]["labels"].append("ghost"))

        report = validate(broken)
        assert not report
        (issue,) = report.errors
        assert issue.location == "labels" and "'ghost'" in issue.message
        assert [lb.name for lb in report.labels] == [
            "nuclei"
        ]  # the real one is still described

    def test_non_integer_label_dtype(self, tmp_path):
        path = tmp_path / "floatlabel.zarr"
        writer = Writer(
            path=path,
            image=np.zeros((20, 20), dtype=np.uint8),
            dims=["y", "x"],
            axis_units={"y": "micrometer", "x": "micrometer"},
        )
        writer.write()
        writer.add_labels(name="floaty", array=np.zeros((20, 20), dtype=np.float32))

        report = validate(path)
        assert not report
        assert any("integer data type, got float32" in i.message for i in report.errors)
        assert report.errors[0].location == "labels/floaty"


class TestFilesetStructure:
    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate(tmp_path / "does-not-exist.zarr")

    def test_directory_is_not_a_zarr_group(self, tmp_path):
        report = validate(tmp_path)
        assert not report
        assert "not a Zarr group" in report.errors[0].message

    def test_zarr_v2_store_is_unsupported(self, tmp_path):
        path = tmp_path / "v2.zarr"
        group = zarr.open_group(str(path), mode="w", zarr_format=2)
        group.attrs["multiscales"] = [{"version": "0.4"}]

        report = validate(path)
        assert report.zarr_format == 2
        assert not report
        assert any("Zarr format 2 is not supported" in i.message for i in report.errors)

    def test_level_array_missing_on_disk(self, valid_fileset, tmp_path):
        broken = tmp_path / "broken.zarr"
        shutil.copytree(valid_fileset, broken)
        shutil.rmtree(broken / "1")

        report = validate(broken)
        assert not report
        (issue,) = report.errors
        assert issue.path == "ome.multiscales.0.datasets.1.path"
        assert "does not exist" in issue.message
        assert report.levels[1].shape is None

    def test_axes_do_not_match_array_dimensions(self, valid_fileset, tmp_path):
        def drop_channel_axis(attrs):
            multiscale = attrs["ome"]["multiscales"][0]
            multiscale["axes"] = multiscale["axes"][1:]
            for dataset in multiscale["datasets"]:
                dataset["coordinateTransformations"][0]["scale"].pop(0)
            attrs["ome"].pop("omero")

        broken = make_broken_copy(
            valid_fileset, tmp_path / "broken.zarr", drop_channel_axis
        )
        report = validate(broken)
        assert not report
        assert any("3 dimensions but 2 axes" in i.message for i in report.errors)


class TestStrictMode:
    def test_strict_is_stricter_than_default(self, valid_fileset):
        """Files written by `Writer` pass the permissive schema but not the strict one."""
        assert validate(valid_fileset).is_valid
        strict = validate(valid_fileset, strict=True)
        assert strict.strict is True
        assert not strict
        assert any("is a required property" in i.message for i in strict.errors)

    def test_strict_reports_everything_default_does(self, valid_fileset, tmp_path):
        broken = make_broken_copy(
            valid_fileset,
            tmp_path / "broken.zarr",
            lambda a: a["ome"].update(version="0.4"),
        )
        default = {(i.location, i.path) for i in validate(broken).errors}
        strict = {(i.location, i.path) for i in validate(broken, strict=True).errors}
        assert default <= strict


PLATE = {
    "columns": [{"name": "1"}],
    "rows": [{"name": "A"}],
    "wells": [{"path": "A/1", "rowIndex": 0, "columnIndex": 0}],
}


def write_root(path, ome):
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    group.attrs["ome"] = ome
    return path


class TestNodeKinds:
    """Roots that are not images (plates, bioformats2raw) are validated by their own schema."""

    def test_valid_plate(self, tmp_path):
        report = validate(
            write_root(tmp_path / "p.zarr", {"version": "0.5", "plate": PLATE})
        )
        assert report.kind == "plate"
        assert report.is_valid, report.errors
        assert report.levels == [] and report.labels == []
        assert report.details == {"layout": "1 rows x 1 columns", "wells": 1}
        assert "layout:    1 rows x 1 columns" in str(report)

    def test_invalid_plate_is_reported_against_plate_schema(self, tmp_path):
        plate = {k: v for k, v in PLATE.items() if k != "wells"}
        report = validate(
            write_root(tmp_path / "p.zarr", {"version": "0.5", "plate": plate})
        )
        assert report.kind == "plate"
        assert not report
        assert any("'wells' is a required property" in i.message for i in report.errors)

    def test_valid_bioformats2raw_layout(self, tmp_path):
        ome = {"version": "0.5", "bioformats2raw.layout": 3}
        report = validate(write_root(tmp_path / "b.zarr", ome))
        assert report.kind == "bf2raw"
        assert report.is_valid, report.errors
        assert report.details == {"layout": "bioformats2raw v3"}

    def test_group_without_ome_metadata(self, tmp_path):
        group = zarr.open_group(str(tmp_path / "empty.zarr"), mode="w", zarr_format=3)
        group.attrs["something"] = 1
        report = validate(tmp_path / "empty.zarr")
        assert report.kind == "image"
        assert not report
        assert any("'ome' is a required property" in i.message for i in report.errors)

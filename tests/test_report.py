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
            "1: GFP (#00FF00, window 0-0)",
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
        # add_labels rejects float arrays, so write a valid label and replace its data.
        writer.add_labels(name="floaty", array=np.zeros((20, 20), dtype=np.uint8))
        label = zarr.open_group(str(path / "labels" / "floaty"), mode="a")
        label.create_array("0", shape=(20, 20), dtype=np.float32, overwrite=True)

        report = validate(path)
        assert not report
        assert any("integer data type, got float32" in i.message for i in report.errors)
        assert report.errors[0].location == "labels/floaty"


class TestFilesetStructure:
    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate(tmp_path / "does-not-exist.zarr")

    def test_url_without_a_group_raises_like_a_missing_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate(f"file://{tmp_path}")

    def test_file_url_to_valid_fileset(self, valid_fileset):
        assert validate(f"file://{valid_fileset}").is_valid

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
        path = write_root(tmp_path / "p.zarr", {"version": "0.5", "plate": PLATE})
        well = zarr.open_group(str(path), mode="a").require_group("A/1")
        well.attrs["ome"] = {"version": "0.5", "well": {"images": [{"path": "0"}]}}
        Writer(
            path / "A" / "1" / "0",
            np.zeros((8, 8), dtype=np.uint8),
            ["y", "x"],
            {"y": "micrometer", "x": "micrometer"},
        ).write()

        report = validate(path)
        assert report.kind == "plate"
        assert report.is_valid, report.errors
        assert report.levels == [] and report.labels == []
        assert report.details == {"layout": "1 rows x 1 columns", "wells": 1}
        assert "layout:    1 rows x 1 columns" in str(report)

    def test_plate_listing_a_well_that_does_not_exist(self, tmp_path):
        report = validate(
            write_root(tmp_path / "p.zarr", {"version": "0.5", "plate": PLATE})
        )
        assert report.kind == "plate"
        assert not report
        (issue,) = report.errors
        assert issue.location == "plate" and issue.path == "ome.plate.wells.0.path"
        assert "well 'A/1' is listed but no such group exists" in issue.message

    def test_invalid_plate_is_reported_against_plate_schema(self, tmp_path):
        plate = {k: v for k, v in PLATE.items() if k != "wells"}
        report = validate(
            write_root(tmp_path / "p.zarr", {"version": "0.5", "plate": plate})
        )
        assert report.kind == "plate"
        assert not report
        assert any("'wells' is a required property" in i.message for i in report.errors)
        assert all(i.location == "plate" for i in report.errors)

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


# --- plates and wells are validated all the way down ----------------------
from ome_zarr_io import PlateWriter  # noqa: E402

UNITS = {"y": "micrometer", "x": "micrometer"}


def edit_ome(group_path, change):
    """Rewrite the `ome` attributes of the group at `group_path` in place."""
    file = group_path / "zarr.json"
    meta = json.loads(file.read_text())
    change(meta["attributes"]["ome"])
    file.write_text(json.dumps(meta))


def found(report, location, text):
    return any(i.location == location and text in i.message for i in report.errors)


@pytest.fixture
def plate(tmp_path):
    """2x2 plate, two acquisitions; A/1 has two fields (one with a label), B/2 one."""
    path = tmp_path / "plate.ome.zarr"
    image = np.zeros((2, 16, 16), dtype=np.uint8)
    with PlateWriter(
        path,
        rows=["A", "B"],
        columns=["1", "2"],
        acquisitions=[{"id": 0}, {"id": 1}],
    ) as writer:
        first = writer.add_field(
            "A", "1", image, ["c", "y", "x"], UNITS, acquisition=0, downscale_levels=1
        )
        first.add_labels("cells", np.ones((1, 16, 16), dtype=np.uint8))
        writer.add_field(
            "A", "1", image, ["c", "y", "x"], UNITS, acquisition=1, downscale_levels=1
        )
        writer.add_field(
            "B", "2", image, ["c", "y", "x"], UNITS, acquisition=0, downscale_levels=1
        )
    return path


class TestPlateIsValidatedAllTheWayDown:
    def test_valid_plate(self, plate):
        report = validate(plate)
        assert report.kind == "plate"
        assert report.is_valid, report.errors

    # -- plate metadata rules the schema cannot express ---------------------
    def test_row_index_disagrees_with_path(self, plate):
        edit_ome(plate, lambda o: o["plate"]["wells"][0].update(rowIndex=1))
        report = validate(plate)
        assert found(report, "plate", "rowIndex 1 refers to row 'B' but the path 'A/1'")
        assert report.errors[0].path == "ome.plate.wells.0.rowIndex"

    def test_column_index_disagrees_with_path(self, plate):
        edit_ome(plate, lambda o: o["plate"]["wells"][0].update(columnIndex=1))
        assert found(validate(plate), "plate", "columnIndex 1 refers to column '2'")

    def test_index_out_of_range(self, plate):
        edit_ome(plate, lambda o: o["plate"]["wells"][0].update(rowIndex=5))
        assert found(validate(plate), "plate", "rowIndex 5 is out of range")

    def test_well_path_not_in_rows(self, plate):
        """A path naming an undeclared row cannot agree with any rowIndex."""
        edit_ome(plate, lambda o: o["plate"]["wells"][0].update(path="Z/1"))
        report = validate(plate)
        assert found(report, "plate", "names row 'Z'")
        assert found(report, "plate", "well 'Z/1' is listed but no such group exists")

    def test_duplicate_well_path(self, plate):
        edit_ome(
            plate,
            lambda o: o["plate"]["wells"].append(
                {"path": "A/1", "rowIndex": 0, "columnIndex": 1}
            ),
        )
        assert found(validate(plate), "plate", "duplicate well path 'A/1'")

    def test_duplicate_acquisition_id(self, plate):
        edit_ome(plate, lambda o: o["plate"]["acquisitions"].append({"id": 0, "name": "x"}))
        assert found(validate(plate), "plate", "duplicate acquisition id 0")

    def test_duplicate_row_name(self, plate):
        edit_ome(plate, lambda o: o["plate"]["rows"].append({"name": "A", "x": 1}))
        assert found(validate(plate), "plate", "duplicate row name 'A'")

    # -- wells ---------------------------------------------------------------
    def test_listed_well_missing_on_disk(self, plate):
        edit_ome(
            plate,
            lambda o: o["plate"]["wells"].append(
                {"path": "A/2", "rowIndex": 0, "columnIndex": 1}
            ),
        )
        report = validate(plate)
        assert found(report, "plate", "well 'A/2' is listed but no such group exists")

    def test_invalid_well_metadata(self, plate):
        edit_ome(plate / "A" / "1", lambda o: o["well"].update(images=[]))
        report = validate(plate)
        assert not report
        assert {i.location for i in report.errors} == {"A/1"}
        assert report.errors[0].path == "ome.well.images"

    def test_well_group_without_well_metadata(self, plate):
        edit_ome(plate / "B" / "2", lambda o: o.pop("well"))
        assert found(validate(plate), "B/2", "'well' is a required property")

    def test_field_listed_but_missing(self, plate):
        edit_ome(
            plate / "A" / "1",
            lambda o: o["well"]["images"].append({"path": "9", "acquisition": 0}),
        )
        report = validate(plate)
        assert found(report, "A/1", "image '9' is listed but no such group exists")

    def test_duplicate_field_path(self, plate):
        edit_ome(
            plate / "A" / "1",
            lambda o: o["well"]["images"].append({"path": "0", "acquisition": 1}),
        )
        assert found(validate(plate), "A/1", "duplicate image path '0'")

    def test_field_acquisition_not_defined_in_plate(self, plate):
        edit_ome(
            plate / "A" / "1", lambda o: o["well"]["images"][0].update(acquisition=9)
        )
        report = validate(plate)
        assert found(report, "A/1", "acquisition 9 is not defined in the plate metadata")

    def test_field_acquisition_missing_with_several_acquisitions(self, plate):
        edit_ome(plate / "B" / "2", lambda o: o["well"]["images"][0].pop("acquisition"))
        report = validate(plate)
        assert found(report, "B/2", "'acquisition' is required because the plate defines")

    def test_field_acquisition_optional_with_one_acquisition(self, plate):
        edit_ome(plate, lambda o: o["plate"].update(acquisitions=[{"id": 0}]))
        edit_ome(plate / "B" / "2", lambda o: o["well"]["images"][0].pop("acquisition"))
        edit_ome(
            plate / "A" / "1", lambda o: o["well"]["images"][1].update(acquisition=0)
        )
        assert validate(plate).is_valid

    def test_acquisition_given_without_plate_acquisitions(self, plate):
        edit_ome(plate, lambda o: o["plate"].pop("acquisitions"))
        report = validate(plate)
        assert found(report, "A/1", "acquisition 0 is not defined in the plate metadata")

    # -- fields --------------------------------------------------------------
    def test_invalid_field_metadata(self, plate):
        edit_ome(plate / "A" / "1" / "1", lambda o: o["multiscales"][0].pop("axes"))
        report = validate(plate)
        assert found(report, "A/1/1", "'axes' is a required property")
        assert {i.location for i in report.errors} == {"A/1/1"}

    def test_field_that_is_not_an_image(self, plate):
        edit_ome(plate / "B" / "2" / "0", lambda o: o.pop("multiscales"))
        assert found(validate(plate), "B/2/0", "'multiscales' is a required property")

    def test_field_array_missing_on_disk(self, plate):
        shutil.rmtree(plate / "A" / "1" / "0" / "1")
        report = validate(plate)
        assert found(report, "A/1/0", "array '1' listed in datasets does not exist")

    def test_field_channel_problem(self, plate):
        edit_ome(
            plate / "A" / "1" / "0",
            lambda o: o.update(omero={"channels": [{"color": "nope"}] * 2}),
        )
        assert found(validate(plate), "A/1/0", "is not 6 hex digits")

    def test_field_label_problem(self, plate):
        edit_ome(
            plate / "A" / "1" / "0" / "labels" / "cells",
            lambda o: o.pop("image-label"),
        )
        report = validate(plate)
        assert found(report, "A/1/0/labels/cells", "'image-label' is a required property")

    def test_field_label_listed_but_missing(self, plate):
        edit_ome(plate / "A" / "1" / "0" / "labels", lambda o: o["labels"].append("ghost"))
        assert found(validate(plate), "A/1/0/labels", "label 'ghost' is listed")

    # -- version consistency -------------------------------------------------
    @pytest.mark.parametrize("node", ["A/1", "A/1/0"])
    def test_version_must_be_consistent_within_the_hierarchy(self, plate, node):
        edit_ome(plate / node, lambda o: o.update(version="0.4"))
        report = validate(plate)
        assert found(report, node, "version '0.4' differs from the root's '0.5'")

    # -- reporting -----------------------------------------------------------
    def test_every_problem_is_reported(self, plate):
        edit_ome(plate, lambda o: o["plate"]["wells"][0].update(rowIndex=1))
        edit_ome(plate / "B" / "2", lambda o: o["well"]["images"][0].update(acquisition=9))
        edit_ome(plate / "A" / "1" / "0", lambda o: o["multiscales"][0].pop("axes"))
        locations = {i.location for i in validate(plate).errors}
        assert {"plate", "B/2", "A/1/0"} <= locations

    def test_locations_appear_in_the_text_report(self, plate):
        edit_ome(plate / "A" / "1" / "1", lambda o: o["multiscales"][0].pop("axes"))
        assert "A/1/1: ome.multiscales.0: 'axes' is a required property" in str(
            validate(plate)
        )

    def test_strict_checks_fields_with_the_strict_schema(self, plate):
        assert validate(plate).is_valid
        report = validate(plate, strict=True)
        assert found(report, "A/1/0", "'type' is a required property")
        assert found(report, "plate", "'name' is a required property")


class TestWellIsValidatedWithItsFields:
    def test_valid_well(self, plate):
        report = validate(plate / "A" / "1")
        assert report.kind == "well"
        assert report.is_valid, report.errors
        assert report.details == {"images": 2}

    def test_broken_field(self, plate):
        edit_ome(plate / "A" / "1" / "1", lambda o: o["multiscales"][0].pop("axes"))
        report = validate(plate / "A" / "1")
        assert found(report, "1", "'axes' is a required property")

    def test_missing_field(self, plate):
        shutil.rmtree(plate / "A" / "1" / "1")
        report = validate(plate / "A" / "1")
        assert found(report, "well", "image '1' is listed but no such group exists")

    def test_acquisitions_are_not_checked_without_the_plate(self, plate):
        edit_ome(
            plate / "A" / "1", lambda o: o["well"]["images"][0].update(acquisition=99)
        )
        assert validate(plate / "A" / "1").is_valid


# --- labels: integer dtype and level count (NGFF 0.5 MUSTs) ---------------
INTEGER_DTYPES = ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64"]


@pytest.fixture
def labelled(tmp_path):
    """A 3-level image with one valid uint8 label called "mask"."""
    path = tmp_path / "labelled.ome.zarr"
    writer = Writer(
        path, np.zeros((32, 32), dtype=np.uint8), ["y", "x"], UNITS, downscale_levels=2
    )
    writer.write()
    writer.add_labels("mask", np.ones((32, 32), dtype=np.uint8))
    return path


def retype_label(path, dtype, name="mask"):
    """Replace level 0 of a label with an array of another dtype."""
    label = zarr.open_group(str(path / "labels" / name), mode="a")
    label.create_array("0", shape=(32, 32), dtype=dtype, overwrite=True)


class TestLabelDtype:
    @pytest.mark.parametrize("dtype", INTEGER_DTYPES)
    def test_integer_dtypes_are_valid(self, labelled, dtype):
        retype_label(labelled, dtype)
        report = validate(labelled)
        assert report.is_valid, report.errors
        assert report.labels[0].dtype == dtype

    @pytest.mark.parametrize("dtype", ["float32", "float64", "bool", "complex64"])
    def test_other_dtypes_are_invalid(self, labelled, dtype):
        retype_label(labelled, dtype)
        report = validate(labelled)
        assert not report
        assert found(report, "labels/mask", f"integer data type, got {dtype}")


class TestLabelLevelCount:
    def test_matching_count_is_valid(self, labelled):
        report = validate(labelled)
        assert report.is_valid, report.errors
        assert report.labels[0].n_levels == len(report.levels) == 3

    def test_label_with_fewer_levels(self, labelled):
        edit_ome(
            labelled / "labels" / "mask", lambda o: o["multiscales"][0]["datasets"].pop()
        )
        report = validate(labelled)
        assert not report
        (issue,) = report.errors
        assert issue.location == "labels/mask"
        assert issue.path == "ome.multiscales.0.datasets"
        assert "label has 2 scale levels but its image has 3" in issue.message

    def test_label_with_more_levels(self, labelled):
        edit_ome(labelled, lambda o: o["multiscales"][0]["datasets"].pop())
        report = validate(labelled)
        assert found(report, "labels/mask", "label has 3 scale levels but its image has 2")

    def test_only_the_mismatched_label_is_reported(self, labelled):
        writer = Writer(
            labelled, np.zeros((32, 32), dtype=np.uint8), ["y", "x"], UNITS
        )
        writer.add_labels("good", np.ones((32, 32), dtype=np.uint8))
        edit_ome(
            labelled / "labels" / "mask", lambda o: o["multiscales"][0]["datasets"].pop()
        )
        assert {i.location for i in validate(labelled).errors} == {"labels/mask"}

    def test_reported_in_text_and_raise(self, labelled):
        edit_ome(
            labelled / "labels" / "mask", lambda o: o["multiscales"][0]["datasets"].pop()
        )
        report = validate(labelled)
        assert "labels/mask: ome.multiscales.0.datasets: label has 2 scale levels" in str(
            report
        )
        with pytest.raises(ValidationError, match="label has 2 scale levels"):
            report.raise_if_invalid()

    def test_mismatch_inside_a_plate_field(self, plate):
        edit_ome(
            plate / "A" / "1" / "0" / "labels" / "cells",
            lambda o: o["multiscales"][0]["datasets"].pop(),
        )
        report = validate(plate)
        assert found(
            report, "A/1/0/labels/cells", "label has 1 scale levels but its image has 2"
        )

    def test_malformed_image_datasets_is_not_blamed_on_the_label(self, labelled):
        """A broken image entry is an image problem, not a label level mismatch."""
        edit_ome(labelled, lambda o: o["multiscales"][0]["datasets"][1].pop("path"))
        report = validate(labelled)
        assert not report
        assert all(i.location == "image" for i in report.errors)

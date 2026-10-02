"""Tests for PlateWriter (HCS plate/well/field writing)."""

import json

import dask.array as da
import numpy as np
import pytest
import zarr
from jsonschema.exceptions import ValidationError

from ome_zarr_io import PlateWriter, Reader, Writer, validate

UNITS = {"y": "micrometer", "x": "micrometer"}
UNITS_Z = {"z": "micrometer", **UNITS}
DIMS = ["c", "y", "x"]


def image(c=2, size=32, value=None):
    if value is not None:
        return np.full((c, size, size), value, dtype=np.uint8)
    return np.random.randint(0, 255, size=(c, size, size), dtype=np.uint8)


def attrs(path):
    return json.loads((path / "zarr.json").read_text())["attributes"]["ome"]


@pytest.fixture
def plate_path(tmp_path):
    return tmp_path / "plate.ome.zarr"


class TestWriting:
    def test_round_trip(self, plate_path):
        with PlateWriter(
            plate_path, rows=["A", "B"], columns=["1", "2", "3"], name="Screen"
        ) as plate:
            plate.add_field("A", "1", image(), DIMS, UNITS)
            plate.add_field("A", "1", image(), DIMS, UNITS)
            plate.add_field("B", "3", image(), DIMS, UNITS)

        report = validate(plate_path)
        assert report.is_valid
        assert report.kind == "plate"
        assert report.details == {
            "name": "Screen",
            "layout": "2 rows x 3 columns",
            "wells": 2,
            "fields": 2,
        }
        for field in ("A/1/0", "A/1/1", "B/3/0"):
            reader = Reader(plate_path / field)
            assert reader.validate()
            assert reader.dims == DIMS

    def test_plate_and_well_metadata(self, plate_path):
        acquisitions = [{"id": 0, "name": "t0"}, {"id": 1, "name": "t1"}]
        with PlateWriter(
            plate_path,
            rows=["A", "B"],
            columns=["1", "2"],
            acquisitions=acquisitions,
        ) as plate:
            # Written out of order: wells must still come out in row/column order.
            plate.add_field("B", "2", image(), DIMS, UNITS, acquisition=1)
            plate.add_field("A", "1", image(), DIMS, UNITS, acquisition=0)
            plate.add_field("A", "1", image(), DIMS, UNITS)

        ome = attrs(plate_path)
        assert ome["version"] == "0.5"
        plate_meta = ome["plate"]
        assert plate_meta["rows"] == [{"name": "A"}, {"name": "B"}]
        assert plate_meta["columns"] == [{"name": "1"}, {"name": "2"}]
        assert plate_meta["wells"] == [
            {"path": "A/1", "rowIndex": 0, "columnIndex": 0},
            {"path": "B/2", "rowIndex": 1, "columnIndex": 1},
        ]
        assert plate_meta["field_count"] == 2
        assert plate_meta["acquisitions"] == acquisitions
        assert "name" not in plate_meta

        assert attrs(plate_path / "A" / "1")["well"]["images"] == [
            {"path": "0", "acquisition": 0},
            {"path": "1"},
        ]
        assert attrs(plate_path / "B" / "2")["well"]["images"] == [
            {"path": "0", "acquisition": 1}
        ]

    def test_explicit_field_index(self, plate_path):
        with PlateWriter(plate_path, rows=["A"], columns=["1"]) as plate:
            plate.add_field("A", "1", image(), DIMS, UNITS, field=5)
            plate.add_field("A", "1", image(), DIMS, UNITS)
        paths = [i["path"] for i in attrs(plate_path / "A" / "1")["well"]["images"]]
        assert paths == ["5", "1"]

    def test_returns_writer_and_names_field(self, plate_path):
        with PlateWriter(plate_path, rows=["A"], columns=["1"]) as plate:
            writer = plate.add_field("A", "1", image(), DIMS, UNITS)
            named = plate.add_field("A", "1", image(), DIMS, UNITS, name="custom")
        assert isinstance(writer, Writer)
        multiscale = lambda f: attrs(plate_path / "A" / "1" / f)["multiscales"][0]
        assert multiscale("0")["name"] == "A/1/0"
        assert multiscale("1")["name"] == "custom"
        assert named.name == "custom"

    def test_writer_kwargs_are_forwarded(self, plate_path):
        with PlateWriter(plate_path, rows=["A"], columns=["1"]) as plate:
            plate.add_field(
                "A",
                "1",
                image(),
                DIMS,
                UNITS,
                downscale_levels=2,
                channels=["DAPI", "GFP"],
                scale_transformations={"y": 0.5, "x": 0.5},
            )
        field = plate_path / "A" / "1" / "0"
        assert Reader(field).n_levels == 3
        assert Reader(field).channel_names == ["DAPI", "GFP"]

    def test_dask_input(self, plate_path):
        data = da.from_array(image(value=7), chunks=(1, 16, 16))
        with PlateWriter(plate_path, rows=["A"], columns=["1"]) as plate:
            plate.add_field("A", "1", data, DIMS, UNITS)
        level0 = zarr.open_array(str(plate_path / "A/1/0/0"))
        assert level0.shape == (2, 32, 32)
        assert (level0[:] == 7).all()

    def test_labels_on_a_field(self, plate_path):
        with PlateWriter(plate_path, rows=["A"], columns=["1"]) as plate:
            writer = plate.add_field("A", "1", image(), DIMS, UNITS)
            writer.add_labels("cells", np.ones((1, 32, 32), dtype=np.uint16))
        assert validate(plate_path).is_valid
        assert Reader(plate_path / "A/1/0").label_names == ["cells"]
        label = attrs(plate_path / "A/1/0/labels/cells")
        assert label["image-label"]["source"] == {"image": "../../"}


class TestPartialPlate:
    def test_valid_after_every_field(self, plate_path):
        """No close() and no with-block: the plate is valid after each add_field."""
        plate = PlateWriter(plate_path, rows=["A", "B"], columns=["1", "2"])
        for row, column in [("A", "1"), ("B", "2"), ("A", "2")]:
            plate.add_field(row, column, image(), DIMS, UNITS)
            assert validate(plate_path).is_valid

    def test_no_plate_metadata_before_first_field(self, plate_path):
        PlateWriter(plate_path, rows=["A"], columns=["1"])
        assert "ome" not in zarr.open_group(str(plate_path)).attrs

    def test_exception_in_with_body_is_not_masked(self, plate_path):
        with pytest.raises(RuntimeError, match="boom"):
            with PlateWriter(plate_path, rows=["A"], columns=["1"]):
                raise RuntimeError("boom")  # close() would raise ValueError: no fields

    def test_close_without_fields_raises(self, plate_path):
        with pytest.raises(ValueError, match="no fields"):
            with PlateWriter(plate_path, rows=["A"], columns=["1"]):
                pass

    def test_close_validates_the_metadata(self, plate_path):
        plate = PlateWriter(plate_path, rows=["A"], columns=["1"])
        plate.add_field("A", "1", image(), DIMS, UNITS)
        well = zarr.open_group(str(plate_path / "A" / "1"))
        well.attrs["ome"] = {"version": "0.5", "well": {"images": []}}
        with pytest.raises(ValidationError):
            plate.close()


class TestArgumentErrors:
    @pytest.mark.parametrize(
        "kwargs, message",
        [
            (dict(rows=[], columns=["1"]), "at least one row"),
            (dict(rows=["A"], columns=[]), "at least one column"),
            (dict(rows=["A", "A"], columns=["1"]), "row names must be unique"),
            (dict(rows=["A"], columns=["1", "1"]), "column names must be unique"),
            (dict(rows=["A_1"], columns=["1"]), "invalid row name"),
            (dict(rows=["A"], columns=["1/2"]), "invalid column name"),
            (dict(rows=["A"], columns=[""]), "invalid column name"),
            (dict(rows=["A"], columns=[1]), "invalid column name"),
            (
                dict(rows=["A"], columns=["1"], acquisitions=[{"name": "x"}]),
                "missing the required 'id'",
            ),
            (
                dict(rows=["A"], columns=["1"], acquisitions=[{"id": 0}, {"id": 0}]),
                "acquisition ids must be unique",
            ),
        ],
    )
    def test_constructor(self, plate_path, kwargs, message):
        with pytest.raises(ValueError, match=message):
            PlateWriter(plate_path, **kwargs)

    @pytest.mark.parametrize(
        "kwargs, error, message",
        [
            (dict(row="C", column="1"), ValueError, "unknown row"),
            (dict(row="A", column="9"), ValueError, "unknown column"),
            (
                dict(row="A", column="1", acquisition=5),
                ValueError,
                "unknown acquisition",
            ),
            (dict(row="A", column="1", field=-1), ValueError, "non-negative"),
            (
                dict(row="A", column="1", overwrite=True),
                TypeError,
                "controlled by the PlateWriter",
            ),
            (
                dict(row="A", column="1", downscale_lvls=1),
                TypeError,
                "unexpected keyword",
            ),
        ],
    )
    def test_add_field(self, plate_path, kwargs, error, message):
        plate = PlateWriter(
            plate_path, rows=["A"], columns=["1"], acquisitions=[{"id": 0}]
        )
        with pytest.raises(error, match=message):
            plate.add_field(image=image(), dims=DIMS, axis_units=UNITS, **kwargs)

    def test_duplicate_field(self, plate_path):
        plate = PlateWriter(plate_path, rows=["A"], columns=["1"])
        plate.add_field("A", "1", image(), DIMS, UNITS, field=3)
        with pytest.raises(ValueError, match="field 3 already exists"):
            plate.add_field("A", "1", image(), DIMS, UNITS, field=3)

    def test_unknown_acquisition_is_not_checked_without_declared_ones(
        self, plate_path
    ):
        with PlateWriter(plate_path, rows=["A"], columns=["1"]) as plate:
            plate.add_field("A", "1", image(), DIMS, UNITS, acquisition=7)
        assert validate(plate_path).is_valid


class TestFailedCallsLeaveNoTrace:
    def test_invalid_call_creates_no_groups(self, plate_path):
        plate = PlateWriter(plate_path, rows=["A"], columns=["1", "2"])
        plate.add_field("A", "1", image(), DIMS, UNITS)
        with pytest.raises(TypeError):
            plate.add_field("A", "2", image(), DIMS, UNITS, downscale_lvls=1)
        assert not (plate_path / "A" / "2").exists()

    def test_failed_first_call_to_a_well_does_not_register_it(self, plate_path):
        """Regression: a failed call must not leave a phantom well in the plate."""
        plate = PlateWriter(plate_path, rows=["A"], columns=["1", "2"])
        with pytest.raises(TypeError):
            plate.add_field("A", "2", image(), DIMS, UNITS, downscale_lvls=1)
        plate.add_field("A", "1", image(), DIMS, UNITS)
        plate.close()
        wells = attrs(plate_path)["plate"]["wells"]
        assert [w["path"] for w in wells] == ["A/1"]
        assert validate(plate_path).is_valid


class TestOverwrite:
    def test_existing_plate_is_an_error_by_default(self, plate_path):
        PlateWriter(plate_path, rows=["A"], columns=["1"])
        with pytest.raises(Exception):
            PlateWriter(plate_path, rows=["A"], columns=["1"])

    def test_overwrite_replaces_the_plate(self, plate_path):
        with PlateWriter(plate_path, rows=["A"], columns=["1"]) as plate:
            plate.add_field("A", "1", image(), DIMS, UNITS)
        with PlateWriter(
            plate_path, rows=["B"], columns=["2"], overwrite=True
        ) as plate:
            plate.add_field("B", "2", image(), DIMS, UNITS)
        assert not (plate_path / "A").exists()
        assert [w["path"] for w in attrs(plate_path)["plate"]["wells"]] == ["B/2"]
        assert validate(plate_path).is_valid


class TestConsistency:
    @pytest.fixture
    def plate(self, plate_path):
        plate = PlateWriter(plate_path, rows=["A"], columns=["1", "2"])
        plate.add_field("A", "1", image(), DIMS, UNITS, channels=["DAPI", "GFP"])
        return plate

    def test_matching_field_is_accepted(self, plate):
        plate.add_field("A", "2", image(), DIMS, UNITS, channels=["DAPI", "GFP"])

    def test_different_dims(self, plate):
        with pytest.raises(ValueError, match=r"dims \['z', 'y', 'x'\]"):
            plate.add_field("A", "2", image(), ["z", "y", "x"], UNITS_Z)

    def test_different_axis_units(self, plate):
        units = {"y": "nanometer", "x": "nanometer"}
        with pytest.raises(ValueError, match="axes"):
            plate.add_field("A", "2", image(), DIMS, units, channels=["DAPI", "GFP"])

    def test_different_channel_labels(self, plate):
        with pytest.raises(ValueError, match="channels"):
            plate.add_field("A", "2", image(), DIMS, UNITS, channels=["DAPI", "RFP"])

    def test_different_channel_count(self, plate):
        with pytest.raises(ValueError, match="channels"):
            plate.add_field("A", "2", image(c=3), DIMS, UNITS)

    def test_missing_channel_labels(self, plate):
        with pytest.raises(ValueError, match="channels"):
            plate.add_field("A", "2", image(), DIMS, UNITS)

    def test_message_names_the_field(self, plate):
        with pytest.raises(ValueError, match=r"field A/2/0 is inconsistent"):
            plate.add_field("A", "2", image(), ["z", "y", "x"], UNITS_Z)

    def test_rejected_field_is_not_written(self, plate, plate_path):
        with pytest.raises(ValueError):
            plate.add_field("A", "2", image(), ["z", "y", "x"], UNITS_Z)
        assert not (plate_path / "A" / "2").exists()
        plate.close()
        assert validate(plate_path).is_valid

    def test_reference_is_the_first_successful_field(self, plate_path):
        plate = PlateWriter(plate_path, rows=["A"], columns=["1"])
        with pytest.raises(TypeError):
            plate.add_field("A", "1", image(), ["z", "y", "x"], UNITS_Z, nope=1)
        # The failed call must not have set the reference.
        plate.add_field("A", "1", image(), DIMS, UNITS)

    def test_opt_out(self, plate_path):
        with PlateWriter(
            plate_path, rows=["A"], columns=["1"], check_consistency=False
        ) as plate:
            plate.add_field("A", "1", image(), DIMS, UNITS)
            plate.add_field("A", "1", image(), ["z", "y", "x"], UNITS_Z)
        assert validate(plate_path).is_valid

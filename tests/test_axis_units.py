"""This module tests the functionality of the axis_units parameter."""

from typing import List, Dict, Any
import numpy as np
import pytest

from ome_zarr_writer.image import OmeZarrImage
from ome_zarr_writer.schema_models import Axis


class TestAxisUnits:
    """This is the documentation for TestAxisUnits class, for testing the axis_units parameter in OmeZarrImage."""

    @pytest.fixture
    def sample_2D_image(self):
        return np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)

    def test_axis_units_with_list_of_axes(self, tmp_path):
       
        image = np.random.randint(0, 255, size=(2, 5, 64, 64), dtype=np.uint8)
        dims = ["c", "z", "y", "x"]

        axis_units: List[Axis] = [
            Axis(name="c", type="channel"),
            Axis(name="z", type="space", unit="micrometer"),
            Axis(name="y", type="space", unit="micrometer"),
            Axis(name="x", type="space", unit="micrometer"),
        ]

        ome_zarr_image = OmeZarrImage(
            path=tmp_path / "test_list.zarr",
            image=image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )

        assert len(ome_zarr_image.axes) == 4

        assert ome_zarr_image.axes[0].name == "c"
        assert ome_zarr_image.axes[0].type == "channel"
        assert ome_zarr_image.axes[0].unit is None

        assert ome_zarr_image.axes[1].name == "z"
        assert ome_zarr_image.axes[1].type == "space"
        assert ome_zarr_image.axes[1].unit == "micrometer"

        assert ome_zarr_image.axes[2].name == "y"
        assert ome_zarr_image.axes[2].type == "space"
        assert ome_zarr_image.axes[2].unit == "micrometer"

        assert ome_zarr_image.axes[3].name == "x"
        assert ome_zarr_image.axes[3].type == "space"
        assert ome_zarr_image.axes[3].unit == "micrometer"

    def test_axis_units_with_dictionary(self, tmp_path):
        image = np.random.randint(
            0, 255, size=(10, 2, 5, 64, 64), dtype=np.uint8
        )
        dims = ["t", "c", "z", "y", "x"]

        axis_units: Dict[str, Any] = {
            "t": "second",
            "z": "nanometer",
            "y": "nanometer",
            "x": "nanometer",
        }

        ome_zarr_image = OmeZarrImage(
            path=tmp_path / "test_dict.zarr",
            image=image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )

        assert len(ome_zarr_image.axes) == 5

        assert ome_zarr_image.axes[0].name == "t"
        assert ome_zarr_image.axes[0].type == "time"
        assert ome_zarr_image.axes[0].unit == "second"

        assert ome_zarr_image.axes[1].name == "c"
        assert ome_zarr_image.axes[1].type == "channel"
        assert ome_zarr_image.axes[1].unit is None

        assert ome_zarr_image.axes[2].name == "z"
        assert ome_zarr_image.axes[2].type == "space"
        assert ome_zarr_image.axes[2].unit == "nanometer"

        assert ome_zarr_image.axes[3].name == "y"
        assert ome_zarr_image.axes[3].type == "space"
        assert ome_zarr_image.axes[3].unit == "nanometer"

        assert ome_zarr_image.axes[4].name == "x"
        assert ome_zarr_image.axes[4].type == "space"
        assert ome_zarr_image.axes[4].unit == "nanometer"

    def test_axis_units_with_dictionary_2d_image(self, tmp_path, sample_2D_image):
        dims = ["y", "x"]
        axis_units: Dict[str, Any] = {"y": "micrometer", "x": "micrometer"}

        ome_zarr_image = OmeZarrImage(
            path=tmp_path / "test_minimal.zarr",
            image=sample_2D_image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )

        assert len(ome_zarr_image.axes) == 2

        assert ome_zarr_image.axes[0].name == "y"
        assert ome_zarr_image.axes[0].type == "space"
        assert ome_zarr_image.axes[0].unit == "micrometer"

        assert ome_zarr_image.axes[1].name == "x"
        assert ome_zarr_image.axes[1].type == "space"
        assert ome_zarr_image.axes[1].unit == "micrometer"

    def test_axis_units_list_length_match_dims_length(self, tmp_path):
        image = np.random.randint(0, 255, size=(2, 64, 64), dtype=np.uint8)
        dims = ["c", "y", "x"]

        axis_units: List[Axis] = [
            Axis(name="c", type="channel"),
            Axis(name="y", type="space", unit="micrometer"),
        ]

        with pytest.raises(
            ValueError,
            match="Length of axis_units list.*must match number of dimensions",
        ):
            OmeZarrImage(
                path=tmp_path / "test_error.zarr",
                image=image,
                dims=dims,
                axis_units=axis_units,
                overwrite=True,
            )

    def test_dims_length_match_image_dimensions(self, tmp_path):
        image = np.random.randint(0, 255, size=(2, 64, 64), dtype=np.uint8)
        dims = ["y", "x"]
        axis_units: Dict[str, Any] = {"y": "micrometer", "x": "micrometer"}

        with pytest.raises(
            ValueError, match="Length of dims.*must match number of image dimensions"
        ):
            OmeZarrImage(
                path=tmp_path / "test_error2.zarr",
                image=image,
                dims=dims,
                axis_units=axis_units,
                overwrite=True,
            )

    def test_validity_of_dimension_names(self, tmp_path, sample_2D_image):
        dims = ["y", "invalid"] 
        axis_units: Dict[str, Any] = {"y": "micrometer", "invalid": "micrometer"}

        with pytest.raises(ValueError, match="Unknown dimension 'invalid'"):
            OmeZarrImage(
                path=tmp_path / "test_error3.zarr",
                image=sample_2D_image,
                dims=dims,
                axis_units=axis_units,
                overwrite=True,
            )

    def test_validity_of_axis_units_type(self, tmp_path, sample_2D_image):
        dims = ["y", "x"]
        axis_units = "invalid" 

        with pytest.raises(
            ValueError,
            match="axis_units must be either a list of Axis objects or a dictionary",
        ):
            OmeZarrImage(
                path=tmp_path / "test_error4.zarr",
                image=sample_2D_image,
                dims=dims,
                axis_units=axis_units,
                overwrite=True,
            )

    def test_axis_units_list_contain_axis_objects(self, tmp_path, sample_2D_image):
        dims = ["y", "x"]
        axis_units = [
            Axis(name="y", type="space", unit="micrometer"),
            "not_an_axis",
        ]

        with pytest.raises(ValueError, match="axis_units.*must be an Axis object"):
            OmeZarrImage(
                path=tmp_path / "test_error5.zarr",
                image=sample_2D_image,
                dims=dims,
                axis_units=axis_units,
                overwrite=True,
            )

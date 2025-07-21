"""Test axis_units parameter functionality."""

from typing import List, Dict, Any
import numpy as np
import pytest

from ome_zarr_writer.image import OmeZarrImage
from ome_zarr_writer.schema_models import Axis


class TestAxisUnits:
    """Test the axis_units parameter in OmeZarrImage."""

    def test_axis_units_with_list_of_axes(self, tmp_path):
        """Test axis_units parameter with a list of Axis objects."""
        # Create test image
        image = np.random.randint(0, 255, size=(2, 5, 64, 64), dtype=np.uint8)  # CZYX
        dims = ["c", "z", "y", "x"]

        # Create axis units as a list of Axis objects
        axis_units: List[Axis] = [
            Axis(name="c", type="channel"),
            Axis(name="z", type="space", unit="micrometer"),
            Axis(name="y", type="space", unit="micrometer"),
            Axis(name="x", type="space", unit="micrometer"),
        ]

        # Create OME-Zarr image
        ome_zarr_image = OmeZarrImage(
            path=tmp_path / "test_list.zarr",
            image=image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )

        # Check that axes were properly stored
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
        """Test axis_units parameter with a dictionary."""
        # Create test image
        image = np.random.randint(
            0, 255, size=(10, 2, 5, 64, 64), dtype=np.uint8
        )  # TCZYX
        dims = ["t", "c", "z", "y", "x"]

        # Create axis units as a dictionary
        axis_units: Dict[str, Any] = {"t": "second", "z": "nanometer", "y": "nanometer", "x": "nanometer"}

        # Create OME-Zarr image
        ome_zarr_image = OmeZarrImage(
            path=tmp_path / "test_dict.zarr",
            image=image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )

        # Check that axes were properly created from dictionary
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

    def test_axis_units_with_minimal_dictionary(self, tmp_path):
        """Test axis_units parameter with per-dimension specification."""
        # Create test image
        image = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)  # YX
        dims = ["y", "x"]

        # Create axis units with per-dimension specification
        axis_units: Dict[str, Any] = {"y": "micrometer", "x": "micrometer"}

        # Create OME-Zarr image
        ome_zarr_image = OmeZarrImage(
            path=tmp_path / "test_minimal.zarr",
            image=image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )

        # Check that axes were created with specified units
        assert len(ome_zarr_image.axes) == 2

        assert ome_zarr_image.axes[0].name == "y"
        assert ome_zarr_image.axes[0].type == "space"
        assert ome_zarr_image.axes[0].unit == "micrometer"

        assert ome_zarr_image.axes[1].name == "x"
        assert ome_zarr_image.axes[1].type == "space"
        assert ome_zarr_image.axes[1].unit == "micrometer"

    def test_axis_units_list_length_mismatch(self, tmp_path):
        """Test that axis_units list length must match dims length."""
        image = np.random.randint(0, 255, size=(2, 64, 64), dtype=np.uint8)  # CYX
        dims = ["c", "y", "x"]

        # Create axis units with wrong length
        axis_units: List[Axis] = [
            Axis(name="c", type="channel"),
            Axis(name="y", type="space", unit="micrometer"),
            # Missing X axis
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

    def test_dims_length_mismatch_with_image(self, tmp_path):
        """Test that dims length must match image dimensions."""
        image = np.random.randint(0, 255, size=(2, 64, 64), dtype=np.uint8)  # 3D image
        dims = ["y", "x"]  # Only 2D dims

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

    def test_invalid_dimension_name(self, tmp_path):
        """Test that invalid dimension names are rejected."""
        image = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
        dims = ["y", "invalid"]  # Invalid dimension name

        axis_units: Dict[str, Any] = {"y": "micrometer", "invalid": "micrometer"}

        with pytest.raises(ValueError, match="Unknown dimension 'invalid'"):
            OmeZarrImage(
                path=tmp_path / "test_error3.zarr",
                image=image,
                dims=dims,
                axis_units=axis_units,
                overwrite=True,
            )

    def test_invalid_axis_units_type(self, tmp_path):
        """Test that invalid axis_units type is rejected."""
        image = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
        dims = ["y", "x"]

        # Invalid type for axis_units
        axis_units = "invalid"  # type: ignore  # String instead of list or dict

        with pytest.raises(
            ValueError,
            match="axis_units must be either a list of Axis objects or a dictionary",
        ):
            OmeZarrImage(
                path=tmp_path / "test_error4.zarr",
                image=image,
                dims=dims,
                axis_units=axis_units,  # type: ignore
                overwrite=True,
            )

    def test_axis_units_list_with_non_axis_objects(self, tmp_path):
        """Test that axis_units list must contain only Axis objects."""
        image = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
        dims = ["y", "x"]

        # List with non-Axis object
        axis_units = [
            Axis(name="y", type="space", unit="micrometer"),
            "not_an_axis",  # Invalid item
        ]

        with pytest.raises(ValueError, match="axis_units.*must be an Axis object"):
            OmeZarrImage(
                path=tmp_path / "test_error5.zarr",
                image=image,
                dims=dims,
                axis_units=axis_units,
                overwrite=True,
            )

"""Tests for per-dimension axis units functionality."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from ome_zarr_writer import OmeZarrImage


class TestPerDimensionAxisUnits:
    """Test per-dimension axis units specification."""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_path = self.temp_dir / "test_output.zarr"
        self.test_image = np.random.randint(
            0, 255, size=(2, 10, 64, 64), dtype=np.uint8
        )
        self.dims = ["c", "z", "y", "x"]

    def test_per_dimension_axis_units_specification(self):
        axis_units = {
            "c": None,  
            "z": "micrometer",  
            "y": "micrometer",  
            "x": "micrometer",  
        }

        ome_zarr_image = OmeZarrImage(
            path=self.output_path,
            image=self.test_image,
            dims=self.dims,
            axis_units=axis_units,
            overwrite=True,
        )

        assert len(ome_zarr_image.axes) == 4

        c_axis = ome_zarr_image.axes[0]
        assert c_axis.name == "c"
        assert c_axis.type == "channel"
        assert c_axis.unit is None

        z_axis = ome_zarr_image.axes[1]
        assert z_axis.name == "z"
        assert z_axis.type == "space"
        assert z_axis.unit == "micrometer"

        y_axis = ome_zarr_image.axes[2]
        assert y_axis.name == "y"
        assert y_axis.type == "space"
        assert y_axis.unit == "micrometer"

        x_axis = ome_zarr_image.axes[3]
        assert x_axis.name == "x"
        assert x_axis.type == "space"
        assert x_axis.unit == "micrometer"

    def test_per_dimension_with_time_axis(self):
        test_image = np.random.randint(0, 255, size=(5, 10, 64, 64), dtype=np.uint8)
        dims = ["t", "z", "y", "x"]

        axis_units = {
            "t": "second",
            "z": "micrometer",
            "y": "micrometer",
            "x": "micrometer",
        }

        ome_zarr_image = OmeZarrImage(
            path=self.output_path,
            image=test_image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )

        assert len(ome_zarr_image.axes) == 4

        t_axis = ome_zarr_image.axes[0]
        assert t_axis.name == "t"
        assert t_axis.type == "time"
        assert t_axis.unit == "second"

    def test_per_dimension_mixed_units(self):
        axis_units = {
            "c": None,
            "z": "nanometer",  
            "y": "micrometer",  
            "x": "millimeter",  
        }

        ome_zarr_image = OmeZarrImage(
            path=self.output_path,
            image=self.test_image,
            dims=self.dims,
            axis_units=axis_units,
            overwrite=True,
        )

        z_axis = ome_zarr_image.axes[1]
        assert z_axis.unit == "nanometer"

        y_axis = ome_zarr_image.axes[2]
        assert y_axis.unit == "micrometer"

        x_axis = ome_zarr_image.axes[3]
        assert x_axis.unit == "millimeter"

    def test_per_dimension_case_insensitive(self):
        axis_units = {
            "C": None,  
            "Z": "micrometer",  
            "y": "micrometer",  
            "X": "micrometer",  
        }

        ome_zarr_image = OmeZarrImage(
            path=self.output_path,
            image=self.test_image,
            dims=self.dims,
            axis_units=axis_units,
            overwrite=True,
        )

        assert len(ome_zarr_image.axes) == 4
        
        c_axis = ome_zarr_image.axes[0]
        assert c_axis.name == "c"
        assert c_axis.type == "channel"
        assert c_axis.unit is None

        z_axis = ome_zarr_image.axes[1]
        assert z_axis.name == "z"
        assert z_axis.type == "space"
        assert z_axis.unit == "micrometer"

        y_axis = ome_zarr_image.axes[2]
        assert y_axis.name == "y"
        assert y_axis.type == "space"
        assert y_axis.unit == "micrometer"

        x_axis = ome_zarr_image.axes[3]
        assert x_axis.name == "x"
        assert x_axis.type == "space"
        assert x_axis.unit == "micrometer"

    def test_per_dimension_missing_spatial_unit_not_supported(self):
        axis_units = {
            "c": None,
            "z": "micrometer",
            "y": "micrometer",
        }

        with pytest.raises(
            ValueError, match="Spatial dimension 'x' requires a unit specification"
        ):
            OmeZarrImage(
                path=self.output_path,
                image=self.test_image,
                dims=self.dims,
                axis_units=axis_units,
                overwrite=True,
            )

    def test_per_dimension_missing_time_unit_not_supported(self):
        test_image = np.random.randint(0, 255, size=(5, 10, 64, 64), dtype=np.uint8)
        dims = ["t", "z", "y", "x"]

        axis_units = {
            "z": "micrometer",
            "y": "micrometer",
            "x": "micrometer",
        }

        with pytest.raises(
            ValueError, match="Time dimension 't' requires a unit specification"
        ):
            OmeZarrImage(
                path=self.output_path,
                image=test_image,
                dims=dims,
                axis_units=axis_units,
                overwrite=True,
            )

    def test_channel_dimension_automatic_handling(self):
        axis_units = {
            "z": "micrometer",
            "y": "micrometer",
            "x": "micrometer",
        }

        ome_zarr_image = OmeZarrImage(
            path=self.output_path,
            image=self.test_image,
            dims=self.dims,
            axis_units=axis_units,
            overwrite=True,
        )

        c_axis = ome_zarr_image.axes[0]
        assert c_axis.name == "c"
        assert c_axis.type == "channel"
        assert c_axis.unit is None

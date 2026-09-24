"""Tests for per-dimension axis units functionality."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from ome_zarr_io import Writer


class TestPerDimensionAxisUnits:
    """Test per-dimension axis units specification."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_path = self.temp_dir / "test_output.zarr"
        self.test_image = np.random.randint(
            0, 255, size=(2, 10, 64, 64), dtype=np.uint8
        )
        self.dims = ["c", "z", "y", "x"]

    def test_per_dimension_axis_units(self):
        """Test per-dimension axis units specification."""
        axis_units = {
            "c": None,  # Channel has no unit (automatically handled)
            "z": "micrometer",  # Z dimension in micrometers
            "y": "micrometer",  # Y dimension in micrometers
            "x": "micrometer",  # X dimension in micrometers
        }

        ome_zarr_image = Writer(
            path=self.output_path,
            image=self.test_image,
            dims=self.dims,
            axis_units=axis_units,
            overwrite=True,
        )

        # Check that axes were created correctly
        assert len(ome_zarr_image.axes) == 4

        # Check each axis
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
        """Test per-dimension axis units with time dimension."""
        test_image = np.random.randint(0, 255, size=(5, 10, 64, 64), dtype=np.uint8)
        dims = ["t", "z", "y", "x"]

        axis_units = {
            "t": "second",
            "z": "micrometer",
            "y": "micrometer",
            "x": "micrometer",
        }

        ome_zarr_image = Writer(
            path=self.output_path,
            image=test_image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )

        # Check that axes were created correctly
        assert len(ome_zarr_image.axes) == 4

        t_axis = ome_zarr_image.axes[0]
        assert t_axis.name == "t"
        assert t_axis.type == "time"
        assert t_axis.unit == "second"

    def test_per_dimension_mixed_units(self):
        """Test per-dimension axis units with different spatial units."""
        axis_units = {
            "c": None,
            "z": "nanometer",  # Different unit for Z
            "y": "micrometer",  # Different unit for Y
            "x": "millimeter",  # Different unit for X
        }

        ome_zarr_image = Writer(
            path=self.output_path,
            image=self.test_image,
            dims=self.dims,
            axis_units=axis_units,
            overwrite=True,
        )

        # Check that each axis has the correct unit
        z_axis = ome_zarr_image.axes[1]
        assert z_axis.unit == "nanometer"

        y_axis = ome_zarr_image.axes[2]
        assert y_axis.unit == "micrometer"

        x_axis = ome_zarr_image.axes[3]
        assert x_axis.unit == "millimeter"

    def test_per_dimension_case_insensitive(self):
        """Test that dimension matching is case insensitive."""
        axis_units = {
            "C": None,  # Uppercase C
            "Z": "micrometer",  # Uppercase Z
            "y": "micrometer",  # Lowercase y
            "X": "micrometer",  # Uppercase X
        }

        ome_zarr_image = Writer(
            path=self.output_path,
            image=self.test_image,
            dims=self.dims,
            axis_units=axis_units,
            overwrite=True,
        )

        # Should work without errors
        assert len(ome_zarr_image.axes) == 4

    def test_per_dimension_missing_spatial_unit(self):
        """Test error when spatial dimension is missing a unit."""
        axis_units = {
            "c": None,
            "z": "micrometer",
            "y": "micrometer",
            # Missing x unit
        }

        with pytest.raises(
            ValueError, match="Spatial dimension 'x' requires a unit specification"
        ):
            Writer(
                path=self.output_path,
                image=self.test_image,
                dims=self.dims,
                axis_units=axis_units,
                overwrite=True,
            )

    def test_per_dimension_missing_time_unit(self):
        """Test error when time dimension is missing a unit."""
        test_image = np.random.randint(0, 255, size=(5, 10, 64, 64), dtype=np.uint8)
        dims = ["t", "z", "y", "x"]

        axis_units = {
            # Missing t unit
            "z": "micrometer",
            "y": "micrometer",
            "x": "micrometer",
        }

        with pytest.raises(
            ValueError, match="Time dimension 't' requires a unit specification"
        ):
            Writer(
                path=self.output_path,
                image=test_image,
                dims=dims,
                axis_units=axis_units,
                overwrite=True,
            )

    def test_channel_dimension_automatic_handling(self):
        """Test that channel dimension is automatically handled without requiring unit."""
        axis_units = {
            "z": "micrometer",
            "y": "micrometer",
            "x": "micrometer",
            # Note: no "c" entry - should be handled automatically
        }

        ome_zarr_image = Writer(
            path=self.output_path,
            image=self.test_image,
            dims=self.dims,
            axis_units=axis_units,
            overwrite=True,
        )

        # Channel axis should be created automatically
        c_axis = ome_zarr_image.axes[0]
        assert c_axis.name == "c"
        assert c_axis.type == "channel"
        assert c_axis.unit is None

"""
Tests for multichannel axis creation functions.
"""

import pytest
from ome_zarr_writer.schema_models import (
    create_cyx_axes,
    create_czyx_axes,
    create_tyx_axes,
    create_tcyx_axes,
    Axis,
)


class TestMultichannelAxes:
    """Test the multichannel axis creation convenience functions."""

    def test_create_cyx_axes(self):
        """Test CYX (Channel, Y, X) axis creation."""
        axes = create_cyx_axes(0.1, 0.2, unit="micrometer")

        assert len(axes) == 3

        # Check channel axis
        assert axes[0].name == "c"
        assert axes[0].type == "channel"
        assert axes[0].unit is None

        # Check Y axis
        assert axes[1].name == "y"
        assert axes[1].type == "space"
        assert axes[1].unit == "micrometer"

        # Check X axis
        assert axes[2].name == "x"
        assert axes[2].type == "space"
        assert axes[2].unit == "micrometer"

    def test_create_czyx_axes(self):
        """Test CZYX (Channel, Z, Y, X) axis creation."""
        axes = create_czyx_axes(0.1, 0.1, 0.3, unit="nanometer")

        assert len(axes) == 4

        # Check channel axis
        assert axes[0].name == "c"
        assert axes[0].type == "channel"
        assert axes[0].unit is None

        # Check Z axis
        assert axes[1].name == "z"
        assert axes[1].type == "space"
        assert axes[1].unit == "nanometer"

        # Check Y axis
        assert axes[2].name == "y"
        assert axes[2].type == "space"
        assert axes[2].unit == "nanometer"

        # Check X axis
        assert axes[3].name == "x"
        assert axes[3].type == "space"
        assert axes[3].unit == "nanometer"

    def test_create_tyx_axes(self):
        """Test TYX (Time, Y, X) axis creation."""
        axes = create_tyx_axes(0.5, 0.5, unit="millimeter")

        assert len(axes) == 3

        # Check time axis
        assert axes[0].name == "t"
        assert axes[0].type == "time"
        assert axes[0].unit is None

        # Check Y axis
        assert axes[1].name == "y"
        assert axes[1].type == "space"
        assert axes[1].unit == "millimeter"

        # Check X axis
        assert axes[2].name == "x"
        assert axes[2].type == "space"
        assert axes[2].unit == "millimeter"

    def test_create_tcyx_axes(self):
        """Test TCYX (Time, Channel, Y, X) axis creation."""
        axes = create_tcyx_axes(0.25, 0.25, unit="meter")

        assert len(axes) == 4

        # Check time axis
        assert axes[0].name == "t"
        assert axes[0].type == "time"
        assert axes[0].unit is None

        # Check channel axis
        assert axes[1].name == "c"
        assert axes[1].type == "channel"
        assert axes[1].unit is None

        # Check Y axis
        assert axes[2].name == "y"
        assert axes[2].type == "space"
        assert axes[2].unit == "meter"

        # Check X axis
        assert axes[3].name == "x"
        assert axes[3].type == "space"
        assert axes[3].unit == "meter"

    def test_axes_default_unit(self):
        """Test that default unit is applied correctly."""
        axes = create_cyx_axes(1.0, 1.0)  # No unit specified

        # Channel axis should not have unit
        assert axes[0].unit is None

        # Spatial axes should have default unit (micrometers)
        assert axes[1].unit == "micrometer"
        assert axes[2].unit == "micrometer"

    def test_axes_are_axis_objects(self):
        """Test that all returned objects are Axis instances."""
        axes = create_czyx_axes(0.1, 0.1, 0.2)

        for axis in axes:
            assert isinstance(axis, Axis)

    def test_axes_different_pixel_sizes(self):
        """Test with different pixel sizes for X and Y."""
        pixel_size_x = 0.065
        pixel_size_y = 0.065
        pixel_size_z = 0.2

        axes = create_czyx_axes(pixel_size_x, pixel_size_y, pixel_size_z)

        # The axes themselves don't store pixel size (that's in transformations)
        # but they should have the correct structure
        assert len(axes) == 4
        assert [ax.name for ax in axes] == ["c", "z", "y", "x"]
        assert [ax.type for ax in axes] == ["channel", "space", "space", "space"]

    def test_multichannel_axis_ordering(self):
        """Test that axes are returned in the correct order."""
        # CYX ordering
        cyx_axes = create_cyx_axes(0.1, 0.1)
        assert [ax.name for ax in cyx_axes] == ["c", "y", "x"]

        # CZYX ordering
        czyx_axes = create_czyx_axes(0.1, 0.1, 0.2)
        assert [ax.name for ax in czyx_axes] == ["c", "z", "y", "x"]

        # TYX ordering
        tyx_axes = create_tyx_axes(0.1, 0.1)
        assert [ax.name for ax in tyx_axes] == ["t", "y", "x"]

        # TCYX ordering
        tcyx_axes = create_tcyx_axes(0.1, 0.1)
        assert [ax.name for ax in tcyx_axes] == ["t", "c", "y", "x"]

    def test_custom_units(self):
        """Test with custom units."""
        custom_unit = "angstrom"
        axes = create_cyx_axes(0.001, 0.001, unit=custom_unit)

        # Spatial axes should have custom unit
        assert axes[1].unit == custom_unit
        assert axes[2].unit == custom_unit

        # Channel axis should not have unit
        assert axes[0].unit is None

    def test_valid_space_units(self):
        """Test that valid space units are accepted."""
        # Test a few representative units from the valid set
        for unit in ["micrometer", "nanometer", "millimeter", "meter", "angstrom"]:
            axes = create_cyx_axes(0.1, 0.1, unit=unit)
            assert axes[1].unit == unit
            assert axes[2].unit == unit

    def test_invalid_space_unit(self):
        """Test that invalid space units raise ValueError."""
        # Try to create an axis with an invalid unit
        with pytest.raises(
            ValueError, match="Invalid unit 'invalid_unit' for space axis"
        ):
            Axis(name="y", type="space", unit="invalid_unit")

    def test_invalid_axis_type(self):
        """Test that invalid axis types raise ValueError."""
        # Try to create an axis with an invalid type
        with pytest.raises(ValueError, match="Invalid axis type 'invalid_type'"):
            Axis(name="z", type="invalid_type")

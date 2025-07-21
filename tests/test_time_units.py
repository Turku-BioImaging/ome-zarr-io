"""
Tests for time axis unit validation.
"""

import pytest
from ome_zarr_writer.schema_models import (
    Axis,
    create_axes,
    validate_tczyx_axis_ordering,
    VALID_TIME_UNITS,
)


class TestTimeUnits:
    """Test time axis unit validation."""

    def test_valid_time_units(self):
        """Test that all valid time units are accepted."""
        valid_time_units = [
            "attosecond",
            "centisecond",
            "day",
            "decisecond",
            "exasecond",
            "femtosecond",
            "gigasecond",
            "hectosecond",
            "hour",
            "kilosecond",
            "megasecond",
            "microsecond",
            "millisecond",
            "minute",
            "nanosecond",
            "petasecond",
            "picosecond",
            "second",
            "terasecond",
            "yoctosecond",
            "yottasecond",
            "zeptosecond",
            "zettasecond",
        ]

        for unit in valid_time_units:
            # Should not raise any exception
            axis = Axis(name="t", type="time", unit=unit)
            assert axis.unit == unit
            assert axis.type == "time"

    def test_invalid_time_unit(self):
        """Test that invalid time units are rejected."""
        invalid_units = ["meter", "inch", "invalid_unit", "frames", "ticks"]

        for unit in invalid_units:
            with pytest.raises(ValueError, match="Invalid unit .* for time axis"):
                Axis(name="t", type="time", unit=unit)

    def test_time_axis_without_unit(self):
        """Test that time axes can be created without units."""
        axis = Axis(name="t", type="time")
        assert axis.unit is None
        assert axis.type == "time"

    def test_create_axes_with_time_unit(self):
        """Test create_axes function with time units."""
        # Test TYX with time unit
        axes = create_axes("tyx", 0.1, 0.1, time_unit="second")

        assert len(axes) == 3
        assert [ax.name for ax in axes] == ["t", "y", "x"]
        assert [ax.type for ax in axes] == ["time", "space", "space"]
        assert axes[0].unit == "second"
        assert axes[1].unit == "micrometer"
        assert axes[2].unit == "micrometer"

        # Validate TCZYX ordering
        validate_tczyx_axis_ordering(axes)

    def test_create_axes_tczyx_with_time_unit(self):
        """Test create_axes for TCZYX with time unit."""
        axes = create_axes(
            "tczyx", 0.1, 0.1, 0.3, unit="nanometer", time_unit="millisecond"
        )

        assert len(axes) == 5
        assert [ax.name for ax in axes] == ["t", "c", "z", "y", "x"]
        assert [ax.type for ax in axes] == [
            "time",
            "channel",
            "space",
            "space",
            "space",
        ]
        assert axes[0].unit == "millisecond"  # Time unit
        assert axes[1].unit is None  # Channel has no unit
        assert axes[2].unit == "nanometer"  # Space unit
        assert axes[3].unit == "nanometer"  # Space unit
        assert axes[4].unit == "nanometer"  # Space unit

        validate_tczyx_axis_ordering(axes)

    def test_create_axes_time_without_unit(self):
        """Test create_axes with time axes but no time_unit specified."""
        axes = create_axes("tyx", 0.1, 0.1)

        assert len(axes) == 3
        assert axes[0].name == "t"
        assert axes[0].type == "time"
        assert axes[0].unit is None  # No time unit specified

        validate_tczyx_axis_ordering(axes)

    def test_create_axes_invalid_time_unit(self):
        """Test that invalid time units are rejected in create_axes."""
        with pytest.raises(ValueError, match="Invalid unit .* for time axis"):
            create_axes("tyx", 0.1, 0.1, time_unit="invalid_unit")

    def test_common_time_units(self):
        """Test commonly used time units."""
        common_units = ["second", "millisecond", "microsecond", "minute", "hour"]

        for unit in common_units:
            axes = create_axes("tcyx", 0.1, 0.1, time_unit=unit)
            assert axes[0].unit == unit
            validate_tczyx_axis_ordering(axes)

    def test_axis_validation_preserves_space_units(self):
        """Test that time unit validation doesn't interfere with space unit validation."""
        # Valid combination
        axes = create_axes("tzyx", 0.1, 0.1, 0.3, unit="micrometer", time_unit="second")
        validate_tczyx_axis_ordering(axes)

        # Invalid space unit should still be caught
        with pytest.raises(ValueError, match="Invalid unit .* for space axis"):
            create_axes(
                "tzyx", 0.1, 0.1, 0.3, unit="invalid_space_unit", time_unit="second"
            )

        # Invalid time unit should be caught
        with pytest.raises(ValueError, match="Invalid unit .* for time axis"):
            create_axes(
                "tzyx", 0.1, 0.1, 0.3, unit="micrometer", time_unit="invalid_time_unit"
            )

    def test_time_unit_constants(self):
        """Test that the VALID_TIME_UNITS constant contains all expected units."""
        expected_units = {
            "attosecond",
            "centisecond",
            "day",
            "decisecond",
            "exasecond",
            "femtosecond",
            "gigasecond",
            "hectosecond",
            "hour",
            "kilosecond",
            "megasecond",
            "microsecond",
            "millisecond",
            "minute",
            "nanosecond",
            "petasecond",
            "picosecond",
            "second",
            "terasecond",
            "yoctosecond",
            "yottasecond",
            "zeptosecond",
            "zettasecond",
        }

        assert VALID_TIME_UNITS == expected_units
        assert len(VALID_TIME_UNITS) == 23  # Verify count

    def test_integration_with_existing_validation(self):
        """Test that time unit validation integrates properly with existing TCZYX validation."""
        # Create axes with time units and verify full validation chain works
        axes = create_axes(
            "tczyx", 0.1, 0.1, 0.3, unit="micrometer", time_unit="millisecond"
        )

        # This should pass all validations
        validate_tczyx_axis_ordering(axes)

        # Verify each axis has correct properties
        t_axis, c_axis, z_axis, y_axis, x_axis = axes

        assert (
            t_axis.name == "t"
            and t_axis.type == "time"
            and t_axis.unit == "millisecond"
        )
        assert c_axis.name == "c" and c_axis.type == "channel" and c_axis.unit is None
        assert (
            z_axis.name == "z"
            and z_axis.type == "space"
            and z_axis.unit == "micrometer"
        )
        assert (
            y_axis.name == "y"
            and y_axis.type == "space"
            and y_axis.unit == "micrometer"
        )
        assert (
            x_axis.name == "x"
            and x_axis.type == "space"
            and x_axis.unit == "micrometer"
        )

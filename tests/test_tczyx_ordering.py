"""
Tests for TCZYX dimension ordering validation.
"""

import pytest
from ome_zarr_io.schema_models import (
    Axis,
    validate_tczyx_axis_ordering,
    create_axes,  # New unified function
    create_yx_axes,
    create_zyx_axes,
    create_cyx_axes,
    create_czyx_axes,
    create_tyx_axes,
    create_tzyx_axes,
    create_tcyx_axes,
    create_tczyx_axes,
)


class TestTCZYXOrdering:
    """Test TCZYX dimension ordering validation and convenience functions."""

    def test_valid_yx_ordering(self):
        """Test minimum valid ordering: YX."""
        axes = [
            Axis(name="y", type="space", unit="micrometer"),
            Axis(name="x", type="space", unit="micrometer"),
        ]
        # Should not raise
        validate_tczyx_axis_ordering(axes)

    def test_valid_zyx_ordering(self):
        """Test valid ZYX ordering."""
        axes = [
            Axis(name="z", type="space", unit="micrometer"),
            Axis(name="y", type="space", unit="micrometer"),
            Axis(name="x", type="space", unit="micrometer"),
        ]
        validate_tczyx_axis_ordering(axes)

    def test_valid_cyx_ordering(self):
        """Test valid CYX ordering."""
        axes = [
            Axis(name="c", type="channel"),
            Axis(name="y", type="space", unit="micrometer"),
            Axis(name="x", type="space", unit="micrometer"),
        ]
        validate_tczyx_axis_ordering(axes)

    def test_valid_czyx_ordering(self):
        """Test valid CZYX ordering."""
        axes = [
            Axis(name="c", type="channel"),
            Axis(name="z", type="space", unit="micrometer"),
            Axis(name="y", type="space", unit="micrometer"),
            Axis(name="x", type="space", unit="micrometer"),
        ]
        validate_tczyx_axis_ordering(axes)

    def test_valid_tyx_ordering(self):
        """Test valid TYX ordering."""
        axes = [
            Axis(name="t", type="time"),
            Axis(name="y", type="space", unit="micrometer"),
            Axis(name="x", type="space", unit="micrometer"),
        ]
        validate_tczyx_axis_ordering(axes)

    def test_valid_tzyx_ordering(self):
        """Test valid TZYX ordering."""
        axes = [
            Axis(name="t", type="time"),
            Axis(name="z", type="space", unit="micrometer"),
            Axis(name="y", type="space", unit="micrometer"),
            Axis(name="x", type="space", unit="micrometer"),
        ]
        validate_tczyx_axis_ordering(axes)

    def test_valid_tcyx_ordering(self):
        """Test valid TCYX ordering."""
        axes = [
            Axis(name="t", type="time"),
            Axis(name="c", type="channel"),
            Axis(name="y", type="space", unit="micrometer"),
            Axis(name="x", type="space", unit="micrometer"),
        ]
        validate_tczyx_axis_ordering(axes)

    def test_valid_tczyx_ordering(self):
        """Test valid TCZYX ordering (maximum dimensions)."""
        axes = [
            Axis(name="t", type="time"),
            Axis(name="c", type="channel"),
            Axis(name="z", type="space", unit="micrometer"),
            Axis(name="y", type="space", unit="micrometer"),
            Axis(name="x", type="space", unit="micrometer"),
        ]
        validate_tczyx_axis_ordering(axes)

    def test_invalid_ordering_wrong_last_axes(self):
        """Test that validation fails if last axes are not YX."""
        axes = [
            Axis(name="x", type="space", unit="micrometer"),
            Axis(name="y", type="space", unit="micrometer"),
        ]
        with pytest.raises(
            ValueError, match="Last two axes must be Y, X in that order"
        ):
            validate_tczyx_axis_ordering(axes)

    def test_invalid_ordering_wrong_sequence(self):
        """Test that validation fails for wrong dimension sequence."""
        # CZT instead of TCZ - wrong order
        axes = [
            Axis(name="c", type="channel"),
            Axis(name="z", type="space", unit="micrometer"),
            Axis(name="t", type="time"),
            Axis(name="y", type="space", unit="micrometer"),
            Axis(name="x", type="space", unit="micrometer"),
        ]
        with pytest.raises(ValueError, match="Axes must follow TCZYX ordering"):
            validate_tczyx_axis_ordering(axes)

    def test_invalid_axis_name(self):
        """Test that validation fails for invalid axis names."""
        axes = [
            Axis(name="w", type="space", unit="micrometer"),
            Axis(name="y", type="space", unit="micrometer"),
            Axis(name="x", type="space", unit="micrometer"),
        ]
        with pytest.raises(ValueError, match="Invalid axis name 'w'"):
            validate_tczyx_axis_ordering(axes)

    def test_invalid_axis_type(self):
        """Test that validation fails for wrong axis types."""
        axes = [
            Axis(name="t", type="space"),  # Should be "time"
            Axis(name="y", type="space", unit="micrometer"),
            Axis(name="x", type="space", unit="micrometer"),
        ]
        with pytest.raises(
            ValueError, match="Axis 't' should have type 'time', got 'space'"
        ):
            validate_tczyx_axis_ordering(axes)

    def test_too_few_axes(self):
        """Test that validation fails with too few axes."""
        axes = [Axis(name="x", type="space", unit="micrometer")]
        with pytest.raises(ValueError, match="Must have at least 2 axes"):
            validate_tczyx_axis_ordering(axes)

    def test_convenience_functions_follow_tczyx(self):
        """Test that all convenience functions produce valid TCZYX ordering."""
        test_cases = [
            (create_yx_axes(0.1, 0.1), ["y", "x"]),
            (create_zyx_axes(0.1, 0.1, 0.3), ["z", "y", "x"]),
            (create_cyx_axes(0.1, 0.1), ["c", "y", "x"]),
            (create_czyx_axes(0.1, 0.1, 0.3), ["c", "z", "y", "x"]),
            (create_tyx_axes(0.1, 0.1), ["t", "y", "x"]),
            (create_tzyx_axes(0.1, 0.1, 0.3), ["t", "z", "y", "x"]),
            (create_tcyx_axes(0.1, 0.1), ["t", "c", "y", "x"]),
            (create_tczyx_axes(0.1, 0.1, 0.3), ["t", "c", "z", "y", "x"]),
        ]

        for axes, expected_names in test_cases:
            # Check names are correct
            actual_names = [ax.name for ax in axes]
            assert (
                actual_names == expected_names
            ), f"Expected {expected_names}, got {actual_names}"

            # Check that validation passes
            validate_tczyx_axis_ordering(axes)

    def test_convenience_functions_axis_types(self):
        """Test that convenience functions assign correct axis types."""
        # Test TCZYX (maximum case)
        axes = create_tczyx_axes(0.1, 0.1, 0.3)

        expected_types = ["time", "channel", "space", "space", "space"]
        actual_types = [ax.type for ax in axes]

        assert actual_types == expected_types

        # Test units are only assigned to space axes
        expected_units = [None, None, "micrometer", "micrometer", "micrometer"]
        actual_units = [ax.unit for ax in axes]

        assert actual_units == expected_units

    def test_backward_compatibility_aliases(self):
        """Test that backward compatibility aliases work."""
        from ome_zarr_io.schema_models import create_2d_axes, create_3d_axes

        # Test that old function names still work
        yx_axes = create_2d_axes(0.1, 0.1)
        zyx_axes = create_3d_axes(0.1, 0.1, 0.3)

        # Should be equivalent to new functions
        assert [ax.name for ax in yx_axes] == ["y", "x"]
        assert [ax.name for ax in zyx_axes] == ["z", "y", "x"]

        # Should pass validation
        validate_tczyx_axis_ordering(yx_axes)
        validate_tczyx_axis_ordering(zyx_axes)


class TestTCZYXIntegration:
    """Test integration of TCZYX ordering with other components."""

    def test_multiscale_validates_tczyx(self):
        """Test that Multiscale class validates TCZYX ordering."""
        from ome_zarr_io.schema_models import (
            Multiscale,
            Dataset,
            ScaleTransformation,
        )

        # Valid TCZYX axes should work
        valid_axes = create_tczyx_axes(0.1, 0.1, 0.3)
        dataset = Dataset("0", [ScaleTransformation([1.0, 1.0, 0.3, 0.1, 0.1])])

        multiscale = Multiscale(datasets=[dataset], axes=valid_axes)
        assert len(multiscale.axes) == 5

        # Invalid ordering should fail
        invalid_axes = [
            Axis(name="c", type="channel"),
            Axis(name="t", type="time"),  # Wrong order: CT instead of TC
            Axis(name="y", type="space", unit="micrometer"),
            Axis(name="x", type="space", unit="micrometer"),
        ]

        with pytest.raises(ValueError, match="Axes must follow TCZYX ordering"):
            Multiscale(datasets=[dataset], axes=invalid_axes)

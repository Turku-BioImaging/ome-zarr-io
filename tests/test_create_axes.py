"""
Tests for the unified create_axes function.
"""

import pytest
from ome_zarr_io.schema_models import create_axes, validate_tczyx_axis_ordering


class TestCreateAxes:
    """Test the unified create_axes function."""

    def test_create_axes_yx(self):
        """Test creating YX axes."""
        axes = create_axes("yx", 0.1, 0.1)

        assert len(axes) == 2
        assert [ax.name for ax in axes] == ["y", "x"]
        assert [ax.type for ax in axes] == ["space", "space"]
        assert all(ax.unit == "micrometer" for ax in axes)
        validate_tczyx_axis_ordering(axes)

    def test_create_axes_zyx(self):
        """Test creating ZYX axes."""
        axes = create_axes("zyx", 0.1, 0.1, 0.3)

        assert len(axes) == 3
        assert [ax.name for ax in axes] == ["z", "y", "x"]
        assert [ax.type for ax in axes] == ["space", "space", "space"]
        assert all(ax.unit == "micrometer" for ax in axes)
        validate_tczyx_axis_ordering(axes)

    def test_create_axes_cyx(self):
        """Test creating CYX axes."""
        axes = create_axes("cyx", 0.1, 0.1)

        assert len(axes) == 3
        assert [ax.name for ax in axes] == ["c", "y", "x"]
        assert [ax.type for ax in axes] == ["channel", "space", "space"]
        assert axes[0].unit is None  # Channel has no unit
        assert axes[1].unit == "micrometer"
        assert axes[2].unit == "micrometer"
        validate_tczyx_axis_ordering(axes)

    def test_create_axes_czyx(self):
        """Test creating CZYX axes."""
        axes = create_axes("czyx", 0.1, 0.1, 0.3)

        assert len(axes) == 4
        assert [ax.name for ax in axes] == ["c", "z", "y", "x"]
        assert [ax.type for ax in axes] == ["channel", "space", "space", "space"]
        assert axes[0].unit is None  # Channel has no unit
        assert all(ax.unit == "micrometer" for ax in axes[1:])
        validate_tczyx_axis_ordering(axes)

    def test_create_axes_tyx(self):
        """Test creating TYX axes."""
        axes = create_axes("tyx", 0.1, 0.1)

        assert len(axes) == 3
        assert [ax.name for ax in axes] == ["t", "y", "x"]
        assert [ax.type for ax in axes] == ["time", "space", "space"]
        assert axes[0].unit is None  # Time has no unit
        assert axes[1].unit == "micrometer"
        assert axes[2].unit == "micrometer"
        validate_tczyx_axis_ordering(axes)

    def test_create_axes_tzyx(self):
        """Test creating TZYX axes."""
        axes = create_axes("tzyx", 0.1, 0.1, 0.3)

        assert len(axes) == 4
        assert [ax.name for ax in axes] == ["t", "z", "y", "x"]
        assert [ax.type for ax in axes] == ["time", "space", "space", "space"]
        assert axes[0].unit is None  # Time has no unit
        assert all(ax.unit == "micrometer" for ax in axes[1:])
        validate_tczyx_axis_ordering(axes)

    def test_create_axes_tcyx(self):
        """Test creating TCYX axes."""
        axes = create_axes("tcyx", 0.1, 0.1)

        assert len(axes) == 4
        assert [ax.name for ax in axes] == ["t", "c", "y", "x"]
        assert [ax.type for ax in axes] == ["time", "channel", "space", "space"]
        assert axes[0].unit is None  # Time has no unit
        assert axes[1].unit is None  # Channel has no unit
        assert axes[2].unit == "micrometer"
        assert axes[3].unit == "micrometer"
        validate_tczyx_axis_ordering(axes)

    def test_create_axes_tczyx(self):
        """Test creating TCZYX axes."""
        axes = create_axes("tczyx", 0.1, 0.1, 0.3)

        assert len(axes) == 5
        assert [ax.name for ax in axes] == ["t", "c", "z", "y", "x"]
        assert [ax.type for ax in axes] == [
            "time",
            "channel",
            "space",
            "space",
            "space",
        ]
        assert axes[0].unit is None  # Time has no unit
        assert axes[1].unit is None  # Channel has no unit
        assert all(ax.unit == "micrometer" for ax in axes[2:])
        validate_tczyx_axis_ordering(axes)

    def test_create_axes_custom_unit(self):
        """Test creating axes with custom unit."""
        axes = create_axes("yx", 0.1, 0.1, unit="nanometer")

        assert all(ax.unit == "nanometer" for ax in axes)
        validate_tczyx_axis_ordering(axes)

    def test_create_axes_case_insensitive(self):
        """Test that axes string is case insensitive."""
        axes1 = create_axes("YX", 0.1, 0.1)
        axes2 = create_axes("yx", 0.1, 0.1)
        axes3 = create_axes("  Yx  ", 0.1, 0.1)

        for axes in [axes1, axes2, axes3]:
            assert [ax.name for ax in axes] == ["y", "x"]
            validate_tczyx_axis_ordering(axes)

    def test_create_axes_invalid_combination(self):
        """Test that invalid axis combinations are rejected."""
        invalid_combinations = [
            "xy",  # Wrong order
            "xyz",  # Wrong order
            "ct",  # Wrong order
            "abc",  # Invalid names
            "tcz",  # Missing Y, X
            "tx",  # Missing Y
        ]

        for invalid in invalid_combinations:
            with pytest.raises(ValueError, match="Invalid axes"):
                create_axes(invalid, 0.1, 0.1)

    def test_create_axes_missing_z_size(self):
        """Test that z_size is required when Z is in axes."""
        with pytest.raises(ValueError, match="z_size is required"):
            create_axes("zyx", 0.1, 0.1)  # Missing z_size

        with pytest.raises(ValueError, match="z_size is required"):
            create_axes("czyx", 0.1, 0.1)  # Missing z_size

        with pytest.raises(ValueError, match="z_size is required"):
            create_axes("tczyx", 0.1, 0.1)  # Missing z_size

    def test_create_axes_equivalent_to_specific_functions(self):
        """Test that create_axes produces same results as specific functions."""
        from ome_zarr_io.schema_models import (
            create_yx_axes,
            create_zyx_axes,
            create_cyx_axes,
            create_czyx_axes,
            create_tyx_axes,
            create_tzyx_axes,
            create_tcyx_axes,
            create_tczyx_axes,
        )

        test_cases = [
            ("yx", create_yx_axes(0.1, 0.1)),
            ("zyx", create_zyx_axes(0.1, 0.1, 0.3)),
            ("cyx", create_cyx_axes(0.1, 0.1)),
            ("czyx", create_czyx_axes(0.1, 0.1, 0.3)),
            ("tyx", create_tyx_axes(0.1, 0.1)),
            ("tzyx", create_tzyx_axes(0.1, 0.1, 0.3)),
            ("tcyx", create_tcyx_axes(0.1, 0.1)),
            ("tczyx", create_tczyx_axes(0.1, 0.1, 0.3)),
        ]

        for axes_str, expected_axes in test_cases:
            if "z" in axes_str:
                axes = create_axes(axes_str, 0.1, 0.1, 0.3)
            else:
                axes = create_axes(axes_str, 0.1, 0.1)

            # Compare names and types
            assert [ax.name for ax in axes] == [ax.name for ax in expected_axes]
            assert [ax.type for ax in axes] == [ax.type for ax in expected_axes]
            assert [ax.unit for ax in axes] == [ax.unit for ax in expected_axes]

    def test_create_axes_documentation_example(self):
        """Test examples from the function docstring."""
        # 2D spatial image
        axes = create_axes("yx", 0.1, 0.1)
        assert [ax.name for ax in axes] == ["y", "x"]

        # Multichannel 3D image
        axes = create_axes("czyx", 0.1, 0.1, 0.3)
        assert [ax.name for ax in axes] == ["c", "z", "y", "x"]

        # Time-series multichannel 2D
        axes = create_axes("tcyx", 0.2, 0.2)
        assert [ax.name for ax in axes] == ["t", "c", "y", "x"]

        # Full 5D image
        axes = create_axes("tczyx", 0.25, 0.25, 0.5)
        assert [ax.name for ax in axes] == ["t", "c", "z", "y", "x"]

        # All should pass validation
        for axes in [axes]:
            validate_tczyx_axis_ordering(axes)

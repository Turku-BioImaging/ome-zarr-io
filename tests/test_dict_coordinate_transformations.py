"""This module tests dictionary-based scale transformations."""

import pytest
import numpy as np
from pathlib import Path
import shutil
from ome_zarr_writer import OmeZarrImage
from ome_zarr_writer.schema_models import ScaleTransformation


class TestDictScaleTransformations:
    """
       This is the documentation for TestDictScaleTransformations class, used for testing dictionary-based scale transformations in the new write() method API.
    """

    def setup_method(self):
        """Set up test data."""
        self.test_image = np.random.randint(0, 255, size=(2, 5, 64, 64), dtype=np.uint8)
        self.dims = ["c", "z", "y", "x"]
        self.axis_units = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}
        self.output_path = Path("test_dict_coords.zarr")

    def clean_up_test_data(self):
        if self.output_path.exists():
            shutil.rmtree(self.output_path)

    def test_transformations_with_floats(self):
        scale_transformations = {"z": 0.25, "y": 0.1, "x": 0.1}

        ome_zarr_image = OmeZarrImage(
            path=self.output_path,
            image=self.test_image,
            dims=self.dims,
            axis_units=self.axis_units,
            scale_transformations=scale_transformations,
            overwrite=True,
        )

        # Check that transformations were processed correctly
        assert ome_zarr_image.coordinate_transformations is not None
        assert len(ome_zarr_image.coordinate_transformations) == 1

        transform = ome_zarr_image.coordinate_transformations[0]
        assert isinstance(transform, ScaleTransformation)
        # Expected scale: [1.0, 0.25, 0.1, 0.1] for [c, z, y, x]
        expected_scale = [1.0, 0.25, 0.1, 0.1]
        assert transform.scale == expected_scale

        ome_zarr_image.write()
        assert self.output_path.exists()

    def test_tuple_format_raises_error(self):
        scale_transformations = {
            "z": (0.25, "micrometer"),
            "y": (0.1, "micrometer"),
            "x": (0.1, "micrometer"),
        }
        

        with pytest.raises(ValueError, match="Expected a number"):
            OmeZarrImage(
                path=self.output_path,
                image=self.test_image,
                dims=self.dims,
                axis_units=self.axis_units,
                scale_transformations=scale_transformations,
                overwrite=True,
            )

    def test_partial_specification(self):
        """Test partial specification where only some dimensions are provided."""
        scale_transformations = {
            "y": 0.065,
            "x": 0.065,
            # z and c will get default scale of 1.0
        }

        ome_zarr_image = OmeZarrImage(
            path=self.output_path,
            image=self.test_image,
            dims=self.dims,
            axis_units=self.axis_units,
            scale_transformations=scale_transformations,
            overwrite=True,
        )

        assert ome_zarr_image.coordinate_transformations is not None
        transform = ome_zarr_image.coordinate_transformations[0]
        assert isinstance(transform, ScaleTransformation)
        # Expected scale: [1.0, 1.0, 0.065, 0.065] for [c, z, y, x]
        expected_scale = [1.0, 1.0, 0.065, 0.065]
        assert transform.scale == expected_scale

    def test_different_dimension_order(self):
        test_image = np.random.randint(0, 255, size=(10, 64, 64), dtype=np.uint8)
        dims = ["z", "y", "x"]

        scale_transformations = {"z": 0.5, "y": 0.1, "x": 0.1}

        ome_zarr_image = OmeZarrImage(
            path=self.output_path,
            image=test_image,
            dims=dims,
            axis_units=self.axis_units,
            scale_transformations=scale_transformations,
            overwrite=True,
        )

        assert ome_zarr_image.coordinate_transformations is not None
        assert len(ome_zarr_image.coordinate_transformations) == 1

        transform = ome_zarr_image.coordinate_transformations[0]
        assert isinstance(transform, ScaleTransformation)
        expected_scale = [0.5, 0.1, 0.1]  # [z, y, x]
        assert transform.scale == expected_scale

    def test_raises_error_for_invalid_axis_name(self):
        scale_transformations = {"invalid_dim": 0.1, "y": 0.1, "x": 0.1}

        with pytest.raises(
            ValueError, match="Dimension 'invalid_dim' not found in dims"
        ):
            OmeZarrImage(
                path=self.output_path,
                image=self.test_image,
                dims=self.dims,
                axis_units=self.axis_units,
                scale_transformations=scale_transformations,
                overwrite=True,
            )

    def test_invalid_scale_value(self):
        """Test error handling for invalid scale values."""
        scale_transformations = {
            "z": -0.25,  # Negative scale value should be invalid
            "y": 0.1,
            "x": 0.1,
        }

        with pytest.raises(
            ValueError, match="Scale value for dimension 'z' must be positive"
        ):
            OmeZarrImage(
                path=self.output_path,
                image=self.test_image,
                dims=self.dims,
                axis_units=self.axis_units,
                scale_transformations=scale_transformations,
                overwrite=True,
            )

    def test_invalid_tuple_format(self):
        """Test error handling for invalid tuple format."""
        scale_transformations = {
            "z": (0.25,),  # Tuple with only one element
            "y": 0.1,
            "x": 0.1,
        }

        with pytest.raises(ValueError, match="Invalid value for dimension 'z'"):
            OmeZarrImage(
                path=self.output_path,
                image=self.test_image,
                dims=self.dims,
                axis_units=self.axis_units,
                scale_transformations=scale_transformations,
                overwrite=True,
            )

    def test_none_scale_transformations(self):
        """Test that None scale_transformations works correctly."""
        ome_zarr_image = OmeZarrImage(
            path=self.output_path,
            image=self.test_image,
            dims=self.dims,
            axis_units=self.axis_units,
            scale_transformations=None,
            overwrite=True,
        )

        assert ome_zarr_image.coordinate_transformations is None

    def test_empty_dict_scale_transformations(self):
        """Test that empty dictionary creates default scale transformation."""
        scale_transformations = {}

        ome_zarr_image = OmeZarrImage(
            path=self.output_path,
            image=self.test_image,
            dims=self.dims,
            axis_units=self.axis_units,
            scale_transformations=scale_transformations,
            overwrite=True,
        )

        # Should create a transformation with all 1.0 scales
        assert ome_zarr_image.coordinate_transformations is not None
        assert len(ome_zarr_image.coordinate_transformations) == 1

        transform = ome_zarr_image.coordinate_transformations[0]
        assert isinstance(transform, ScaleTransformation)
        expected_scale = [1.0, 1.0, 1.0, 1.0]  # All default scales
        assert transform.scale == expected_scale

    def test_mixed_types_in_dict(self):
        """Test that mixing tuples with numbers is no longer supported."""
        scale_transformations = {
            "z": (0.25, "micrometer"),  # Tuple format (no longer supported)
            "y": 0.1,  # Float format
            "x": 0.1,  # Float format
        }

        # Should raise an error since tuple format is not supported
        with pytest.raises(ValueError, match="Expected a number"):
            OmeZarrImage(
                path=self.output_path,
                image=self.test_image,
                dims=self.dims,
                axis_units=self.axis_units,
                scale_transformations=scale_transformations,
                overwrite=True,
            )

"""This module tests dictionary-based scale transformations."""

import pytest
import numpy as np
from pathlib import Path
import shutil
from ome_zarr_writer import OmeZarrImage
from ome_zarr_writer.schema_models import ScaleTransformation


class TestDictScaleTransformations:
    """
       This is the documentation for TestDictScaleTransformations class, used for testing dictionary-based scale transformations.
    """

    def setup_method(self):
        self.test_image = np.random.randint(0, 255, size=(2, 5, 64, 64), dtype=np.uint8)
        self.dims = ["c", "z", "y", "x"]
        self.axis_units = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}
        self.output_path = Path("test_dict_coords.zarr")

    def teardown_module(self):
        if self.output_path.exists():
            shutil.rmtree(self.output_path)

    def test_transformations_with_floats_supported(self):
        scale_transformations = {"z": 3.2, "y": 0.75, "x": 0.75}

        ome_zarr_image = OmeZarrImage(
            path=self.output_path,
            image=self.test_image,
            dims=self.dims,
            axis_units=self.axis_units,
            scale_transformations=scale_transformations,
            overwrite=True,
        )

        assert ome_zarr_image.coordinate_transformations is not None
        assert len(ome_zarr_image.coordinate_transformations) == 1

        transform = ome_zarr_image.coordinate_transformations[0]
        assert isinstance(transform, ScaleTransformation)

        expected_scale = [1.0, 3.2, 0.75, 0.75]
        assert transform.scale == expected_scale

        ome_zarr_image.write()
        assert self.output_path.exists()

    def test_tuple_format_raises_error(self):
        scale_transformations = {
            "z": (3.2, "micrometer"),
            "y": (0.75, "micrometer"),
            "x": (0.75, "micrometer"),
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

    def test_partial_specification_supported(self):
        scale_transformations = {
            "y": 0.75,
            "x": 0.75,
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

        expected_scale = [1.0, 1.0, 0.75, 0.75] #[c, z, y, x]
        assert transform.scale == expected_scale

    def test_different_dimension_order_supported(self):
        test_image = np.random.randint(0, 255, size=(10, 64, 64), dtype=np.uint8)
        dims = ["z", "y", "x"]

        scale_transformations = {"z": 1.0, "y": 0.75, "x": 0.75}

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
        expected_scale = [1.0, 0.75, 0.75]  # [z, y, x]
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

    def test_raises_error_for_invalid_scale_value(self):
        scale_transformations = {
            "z": -0.25,  
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

    def test_raises_error_for_invalid_tuple_format(self):
        scale_transformations = {
            "z": (0.25,),  
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

    def test_none_scale_transformations_supported(self):
        ome_zarr_image = OmeZarrImage(
            path=self.output_path,
            image=self.test_image,
            dims=self.dims,
            axis_units=self.axis_units,
            scale_transformations=None,
            overwrite=True,
        )

        assert ome_zarr_image.coordinate_transformations is None

    def test_empty_dict_scale_transformations_defaults(self):
        scale_transformations = {}

        ome_zarr_image = OmeZarrImage(
            path=self.output_path,
            image=self.test_image,
            dims=self.dims,
            axis_units=self.axis_units,
            scale_transformations=scale_transformations,
            overwrite=True,
        )

        assert ome_zarr_image.coordinate_transformations is not None
        assert len(ome_zarr_image.coordinate_transformations) == 1

        transform = ome_zarr_image.coordinate_transformations[0]
        assert isinstance(transform, ScaleTransformation)
        expected_scale = [1.0, 1.0, 1.0, 1.0]
        assert transform.scale == expected_scale

    def test_mixed_types_in_dict_not_supported(self):
        scale_transformations = {
            "z": (0.25, "micrometer"), 
            "y": 0.1,  
            "x": 0.1,  
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

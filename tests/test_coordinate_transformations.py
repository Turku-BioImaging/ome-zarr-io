"""Test coordinate transformations for multiscale levels."""

from typing import List, Union
import numpy as np

from ome_zarr_writer.image import OmeZarrImage
from ome_zarr_writer.schema_models import ScaleTransformation, TranslationTransformation


class TestCoordinateTransformations:
    """Test coordinate transformation calculations for multiscale levels."""

    def test_scale_transformation_adjustment(self, tmp_path):
        """Test that scale transformations are correctly adjusted for each level."""
        # Create test image
        image = np.random.randint(0, 255, size=(2, 3, 64, 64), dtype=np.uint8)
        dims = ["c", "z", "y", "x"]
        
        # Define original coordinate transformations
        coordinate_transformations: List[Union[ScaleTransformation, TranslationTransformation]] = [
            ScaleTransformation(scale=[1.0, 0.25, 0.1, 0.1])  # c, z, y, x
        ]
        
        # Define axis units
        axis_units = {"unit": "micrometer"}
        
        # Create OME-Zarr image
        ome_zarr_image = OmeZarrImage(
            path=tmp_path / "test.zarr",
            image=image,
            dims=dims,
            axis_units=axis_units,
            coordinate_transformations=coordinate_transformations,
            downscale_levels=2,
            downscale_factor=2,
            overwrite=True
        )
        
        # Get coordinate transformations for each level
        level_transformations = ome_zarr_image._create_coordinate_transformations_for_levels()
        
        # Should have 3 levels (original + 2 downscale levels)
        assert len(level_transformations) == 3
        
        # Check level 0 (original)
        level_0_scale = level_transformations[0][0]
        assert isinstance(level_0_scale, ScaleTransformation)
        assert level_0_scale.scale == [1.0, 0.25, 0.1, 0.1]
        
        # Check level 1 (2x downscale)
        level_1_scale = level_transformations[1][0]
        assert isinstance(level_1_scale, ScaleTransformation)
        assert level_1_scale.scale == [1.0, 0.25, 0.2, 0.2]  # Y and X doubled
        
        # Check level 2 (4x downscale)
        level_2_scale = level_transformations[2][0]
        assert isinstance(level_2_scale, ScaleTransformation)
        assert level_2_scale.scale == [1.0, 0.25, 0.4, 0.4]  # Y and X quadrupled

    def test_translation_transformation_unchanged(self, tmp_path):
        """Test that translation transformations remain unchanged across levels."""
        # Create test image
        image = np.random.randint(0, 255, size=(32, 32), dtype=np.uint8)
        dims = ["y", "x"]
        
        # Define original coordinate transformations
        coordinate_transformations: List[Union[ScaleTransformation, TranslationTransformation]] = [
            TranslationTransformation(translation=[5.0, 10.0])  # y, x offsets
        ]
        
        # Define axis units
        axis_units = {"unit": "micrometer"}
        
        # Create OME-Zarr image
        ome_zarr_image = OmeZarrImage(
            path=tmp_path / "test.zarr",
            image=image,
            dims=dims,
            axis_units=axis_units,
            coordinate_transformations=coordinate_transformations,
            downscale_levels=2,
            downscale_factor=2,
            overwrite=True
        )
        
        # Get coordinate transformations for each level
        level_transformations = ome_zarr_image._create_coordinate_transformations_for_levels()
        
        # Should have 3 levels
        assert len(level_transformations) == 3
        
        # Check that translation is the same at all levels
        for level in range(3):
            level_translation = level_transformations[level][0]
            assert isinstance(level_translation, TranslationTransformation)
            assert level_translation.translation == [5.0, 10.0]

    def test_mixed_transformations(self, tmp_path):
        """Test both scale and translation transformations together."""
        # Create test image
        image = np.random.randint(0, 255, size=(3, 32, 32), dtype=np.uint8)
        dims = ["z", "y", "x"]
        
        # Define original coordinate transformations
        coordinate_transformations: List[Union[ScaleTransformation, TranslationTransformation]] = [
            ScaleTransformation(scale=[0.5, 0.2, 0.2]),  # z, y, x
            TranslationTransformation(translation=[1.0, 2.0, 3.0])  # z, y, x offsets
        ]
        
        # Define axis units
        axis_units = {"unit": "micrometer"}
        
        # Create OME-Zarr image
        ome_zarr_image = OmeZarrImage(
            path=tmp_path / "test.zarr",
            image=image,
            dims=dims,
            axis_units=axis_units,
            coordinate_transformations=coordinate_transformations,
            downscale_levels=1,
            downscale_factor=2,
            overwrite=True
        )
        
        # Get coordinate transformations for each level
        level_transformations = ome_zarr_image._create_coordinate_transformations_for_levels()
        
        # Should have 2 levels
        assert len(level_transformations) == 2
        
        # Check level 0
        level_0_scale = level_transformations[0][0]
        level_0_translation = level_transformations[0][1]
        assert isinstance(level_0_scale, ScaleTransformation)
        assert isinstance(level_0_translation, TranslationTransformation)
        assert level_0_scale.scale == [0.5, 0.2, 0.2]
        assert level_0_translation.translation == [1.0, 2.0, 3.0]
        
        # Check level 1
        level_1_scale = level_transformations[1][0]
        level_1_translation = level_transformations[1][1]
        assert isinstance(level_1_scale, ScaleTransformation)
        assert isinstance(level_1_translation, TranslationTransformation)
        assert level_1_scale.scale == [0.5, 0.4, 0.4]  # Y and X doubled
        assert level_1_translation.translation == [1.0, 2.0, 3.0]  # Unchanged

    def test_no_coordinate_transformations(self, tmp_path):
        """Test that empty list is returned when no coordinate transformations provided."""
        # Create test image
        image = np.random.randint(0, 255, size=(32, 32), dtype=np.uint8)
        dims = ["y", "x"]
        
        # Define axis units
        axis_units = {"unit": "micrometer"}
        
        # Create OME-Zarr image without coordinate transformations
        ome_zarr_image = OmeZarrImage(
            path=tmp_path / "test.zarr",
            image=image,
            dims=dims,
            axis_units=axis_units,
            coordinate_transformations=None,  # No transformations
            downscale_levels=2,
            downscale_factor=2,
            overwrite=True
        )
        
        # Get coordinate transformations for each level
        level_transformations = ome_zarr_image._create_coordinate_transformations_for_levels()
        
        # Should return empty list
        assert level_transformations == []

    def test_different_downscale_factors(self, tmp_path):
        """Test coordinate transformations with different downscale factors."""
        # Create test image
        image = np.random.randint(0, 255, size=(81, 81), dtype=np.uint8)  # 81 = 3^4, allows factor 3
        dims = ["y", "x"]
        
        # Define original coordinate transformations
        coordinate_transformations: List[Union[ScaleTransformation, TranslationTransformation]] = [
            ScaleTransformation(scale=[0.1, 0.1])  # y, x
        ]
        
        # Define axis units
        axis_units = {"unit": "micrometer"}
        
        # Create OME-Zarr image with downscale factor 3
        ome_zarr_image = OmeZarrImage(
            path=tmp_path / "test.zarr",
            image=image,
            dims=dims,
            axis_units=axis_units,
            coordinate_transformations=coordinate_transformations,
            downscale_levels=2,
            downscale_factor=3,
            overwrite=True
        )
        
        # Get coordinate transformations for each level
        level_transformations = ome_zarr_image._create_coordinate_transformations_for_levels()
        
        # Should have 3 levels
        assert len(level_transformations) == 3
        
        # Check scaling progression
        level_0_scale = level_transformations[0][0]
        level_1_scale = level_transformations[1][0]
        level_2_scale = level_transformations[2][0]
        
        assert isinstance(level_0_scale, ScaleTransformation)
        assert isinstance(level_1_scale, ScaleTransformation)
        assert isinstance(level_2_scale, ScaleTransformation)
        
        assert level_0_scale.scale == [0.1, 0.1]    # 1x
        assert abs(level_1_scale.scale[0] - 0.3) < 1e-10    # 3x
        assert abs(level_1_scale.scale[1] - 0.3) < 1e-10    # 3x
        assert abs(level_2_scale.scale[0] - 0.9) < 1e-10    # 9x
        assert abs(level_2_scale.scale[1] - 0.9) < 1e-10    # 9x

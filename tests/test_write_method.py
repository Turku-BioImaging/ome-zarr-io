"""Test the write() method functionality."""

from typing import List, Union
import numpy as np
import zarr
from pathlib import Path

from ome_zarr_writer.image import OmeZarrImage
from ome_zarr_writer.schema_models import ScaleTransformation, TranslationTransformation


class TestWriteMethod:
    """Test the write() method implementation."""

    def test_write_simple_2d_image(self, tmp_path):
        """Test writing a simple 2D image."""
        # Create test image
        image = np.random.randint(0, 255, size=(128, 128), dtype=np.uint8)
        dims = ["y", "x"]
        axis_units = {"unit": "micrometer"}
        
        output_path = tmp_path / "test_2d.zarr"
        
        # Create and write OME-Zarr image
        ome_zarr_image = OmeZarrImage(
            path=output_path,
            image=image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True
        )
        
        # Write the image
        ome_zarr_image.write()
        
        # Verify the output exists
        assert output_path.exists()
        assert output_path.is_dir()
        
        # Open and verify the zarr group
        group = zarr.open_group(str(output_path), mode='r')
        
        # Check that level 0 exists
        assert '0' in group
        level_0_array = group['0']
        # Verify it's a zarr array, not a group
        assert isinstance(level_0_array, zarr.Array)
        assert level_0_array.shape == image.shape
        np.testing.assert_array_equal(level_0_array[:], image)
        
        # Check metadata
        assert 'ome' in group.attrs
        ome_metadata = group.attrs['ome']
        assert ome_metadata['version'] == '0.5'
        assert len(ome_metadata['multiscales']) == 1
        
        multiscale = ome_metadata['multiscales'][0]
        assert len(multiscale['datasets']) == 1
        assert len(multiscale['axes']) == 2
        
        # Check axes
        axes = multiscale['axes']
        assert axes[0]['name'] == 'y'
        assert axes[0]['type'] == 'space'
        assert axes[0]['unit'] == 'micrometer'
        assert axes[1]['name'] == 'x'
        assert axes[1]['type'] == 'space'
        assert axes[1]['unit'] == 'micrometer'
        
        # Check dataset
        dataset = multiscale['datasets'][0]
        assert dataset['path'] == '0'
        assert len(dataset['coordinateTransformations']) == 1
        assert dataset['coordinateTransformations'][0]['type'] == 'scale'

    def test_write_multiscale_image(self, tmp_path):
        """Test writing a multiscale image."""
        # Create test image
        image = np.random.randint(0, 255, size=(2, 3, 64, 64), dtype=np.uint8)
        dims = ["c", "z", "y", "x"]
        axis_units = {"unit": "micrometer"}
        
        output_path = tmp_path / "test_multiscale.zarr"
        
        # Create OME-Zarr image with downscaling
        ome_zarr_image = OmeZarrImage(
            path=output_path,
            image=image,
            dims=dims,
            axis_units=axis_units,
            downscale_levels=2,
            downscale_factor=2,
            overwrite=True
        )
        
        # Write the image
        ome_zarr_image.write()
        
        # Verify the output exists
        assert output_path.exists()
        
        # Open and verify the zarr group
        group = zarr.open_group(str(output_path), mode='r')
        
        # Check that multiple levels exist
        assert '0' in group  # Original
        assert '1' in group  # 2x downscaled
        assert '2' in group  # 4x downscaled
        
        # Check shapes
        level_0 = group['0']
        level_1 = group['1'] 
        level_2 = group['2']
        
        assert level_0.shape == (2, 3, 64, 64)
        assert level_1.shape == (2, 3, 32, 32)
        assert level_2.shape == (2, 3, 16, 16)
        
        # Check metadata
        ome_metadata = group.attrs['ome']
        multiscale = ome_metadata['multiscales'][0]
        assert len(multiscale['datasets']) == 3
        assert len(multiscale['axes']) == 4

    def test_write_with_coordinate_transformations(self, tmp_path):
        """Test writing with coordinate transformations."""
        # Create test image
        image = np.random.randint(0, 255, size=(32, 32), dtype=np.uint8)
        dims = ["y", "x"]
        axis_units = {"unit": "micrometer"}
        
        # Define coordinate transformations
        coordinate_transformations: List[Union[ScaleTransformation, TranslationTransformation]] = [
            ScaleTransformation(scale=[0.1, 0.1]),  # 0.1 μm pixel size
            TranslationTransformation(translation=[5.0, 10.0])  # 5,10 μm offset
        ]
        
        output_path = tmp_path / "test_transforms.zarr"
        
        # Create OME-Zarr image
        ome_zarr_image = OmeZarrImage(
            path=output_path,
            image=image,
            dims=dims,
            axis_units=axis_units,
            coordinate_transformations=coordinate_transformations,
            overwrite=True
        )
        
        # Write the image
        ome_zarr_image.write()
        
        # Verify transformations in metadata
        group = zarr.open_group(str(output_path), mode='r')
        ome_metadata = group.attrs['ome']
        multiscale = ome_metadata['multiscales'][0]
        dataset = multiscale['datasets'][0]
        
        transformations = dataset['coordinateTransformations']
        assert len(transformations) == 2
        
        # Check scale transformation
        scale_transform = transformations[0]
        assert scale_transform['type'] == 'scale'
        assert scale_transform['scale'] == [0.1, 0.1]
        
        # Check translation transformation
        translation_transform = transformations[1]
        assert translation_transform['type'] == 'translation'
        assert translation_transform['translation'] == [5.0, 10.0]

    def test_write_overwrite_existing(self, tmp_path):
        """Test overwriting existing files."""
        # Create test image
        image = np.random.randint(0, 255, size=(32, 32), dtype=np.uint8)
        dims = ["y", "x"]
        axis_units = {"unit": "micrometer"}
        
        output_path = tmp_path / "test_overwrite.zarr"
        
        # Create first image
        ome_zarr_image1 = OmeZarrImage(
            path=output_path,
            image=image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True
        )
        ome_zarr_image1.write()
        
        # Verify it exists
        assert output_path.exists()
        
        # Create second image with different data
        image2 = np.random.randint(100, 200, size=(32, 32), dtype=np.uint8)
        ome_zarr_image2 = OmeZarrImage(
            path=output_path,
            image=image2,
            dims=dims,
            axis_units=axis_units,
            overwrite=True
        )
        ome_zarr_image2.write()
        
        # Verify the file was overwritten
        group = zarr.open_group(str(output_path), mode='r')
        level_0 = group['0']
        np.testing.assert_array_equal(level_0[:], image2)

    def test_write_preserves_original_image(self, tmp_path):
        """Test that writing doesn't modify the original image."""
        # Create test image
        original_image = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
        image_copy = original_image.copy()
        dims = ["y", "x"]
        axis_units = {"unit": "micrometer"}
        
        output_path = tmp_path / "test_preserve.zarr"
        
        # Create and write OME-Zarr image
        ome_zarr_image = OmeZarrImage(
            path=output_path,
            image=original_image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True
        )
        ome_zarr_image.write()
        
        # Verify original image is unchanged
        np.testing.assert_array_equal(original_image, image_copy)

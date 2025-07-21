"""Tests for OMERO metadata functionality in OmeZarrImage."""

import numpy as np
import tempfile
import pytest
from pathlib import Path
import zarr

from src.ome_zarr_writer.image import OmeZarrImage
from src.ome_zarr_writer.schema_models import (
    Omero,
    Channel,
    Window,
    create_axes
)


class TestOmeroMetadata:
    """Test OMERO metadata functionality."""

    def test_omero_metadata_basic(self):
        """Test basic OMERO metadata integration."""
        # Create test image data (multichannel: C, Y, X)
        image = np.random.randint(0, 255, size=(3, 100, 100), dtype=np.uint8)
        
        # Create OMERO metadata with channel information
        channels = []
        for i in range(3):
            window = Window(start=0.0, min=0.0, end=255.0, max=255.0)
            channel = Channel(
                window=window,
                label=f"Channel {i+1}",
                color=["FF0000", "00FF00", "0000FF"][i],
                active=True
            )
            channels.append(channel)
        
        omero_metadata = Omero(channels=channels)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_omero.ome.zarr"
            
            # Create OME-Zarr image with OMERO metadata
            ome_zarr = OmeZarrImage(
                path=output_path,
                image=image,
                dims=["c", "y", "x"],
                axis_units={"c": None, "y": "micrometer", "x": "micrometer"},
                scale_transformations={"y": 0.1, "x": 0.1},
                omero_metadata=omero_metadata,
                overwrite=True
            )
            
            # Write the OME-Zarr
            ome_zarr.write()
            
            # Verify the file was created
            assert output_path.exists()
            
            # Check zarr metadata contains OMERO information
            root = zarr.open(str(output_path), mode='r')
            assert 'ome' in root.attrs
            ome_attrs = root.attrs['ome']
            assert 'omero' in ome_attrs
            
            omero_attrs = ome_attrs['omero']
            assert 'channels' in omero_attrs
            assert len(omero_attrs['channels']) == 3
            
            # Verify channel details
            for i, channel in enumerate(omero_attrs['channels']):
                assert 'window' in channel
                assert 'label' in channel
                assert 'color' in channel
                assert 'active' in channel
                assert channel['label'] == f"Channel {i+1}"
                assert channel['color'] == ["FF0000", "00FF00", "0000FF"][i]
                assert channel['active'] is True

    def test_omero_metadata_optional_fields(self):
        """Test OMERO metadata with optional fields."""
        image = np.random.randint(0, 255, size=(2, 50, 50), dtype=np.uint8)
        
        # Create channels with optional fields
        channel1 = Channel(
            window=Window(start=10.0, min=0.0, end=200.0, max=255.0),
            label="Red Channel",
            color="FF0000",
            active=True,
            family="linear"
        )
        
        channel2 = Channel(
            window=Window(start=5.0, min=0.0, end=150.0, max=255.0),
            label="Green Channel",
            color="00FF00",
            active=False
        )
        
        omero_metadata = Omero(channels=[channel1, channel2])
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_omero_optional.ome.zarr"
            
            ome_zarr = OmeZarrImage(
                path=output_path,
                image=image,
                dims=["c", "y", "x"],
                axis_units={"c": None, "y": "micrometer", "x": "micrometer"},
                omero_metadata=omero_metadata,
                overwrite=True
            )
            
            ome_zarr.write()
            
            # Verify metadata
            root = zarr.open(str(output_path), mode='r')
            omero_attrs = root.attrs['ome']['omero']
            
            # Check first channel (has family)
            ch1 = omero_attrs['channels'][0]
            assert ch1['family'] == "linear"
            assert ch1['active'] is True
            
            # Check second channel (no family, inactive)
            ch2 = omero_attrs['channels'][1]
            assert 'family' not in ch2 or ch2['family'] is None
            assert ch2['active'] is False

    def test_without_omero_metadata(self):
        """Test that OME-Zarr works correctly without OMERO metadata."""
        image = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_no_omero.ome.zarr"
            
            # Create OME-Zarr without OMERO metadata
            ome_zarr = OmeZarrImage(
                path=output_path,
                image=image,
                dims=["y", "x"],
                axis_units={"y": "micrometer", "x": "micrometer"},
                overwrite=True
            )
            
            ome_zarr.write()
            
            # Verify file created without OMERO metadata
            root = zarr.open(str(output_path), mode='r')
            ome_attrs = root.attrs['ome']
            assert 'omero' not in ome_attrs

    def test_omero_metadata_with_multiscale(self):
        """Test OMERO metadata with multiscale images."""
        image = np.random.randint(0, 255, size=(2, 128, 128), dtype=np.uint8)
        
        # Create OMERO metadata
        channels = [
            Channel(
                window=Window(start=0.0, min=0.0, end=255.0, max=255.0),
                label="Ch1",
                color="FF0000",
                active=True
            ),
            Channel(
                window=Window(start=0.0, min=0.0, end=255.0, max=255.0),
                label="Ch2",
                color="00FF00",
                active=True
            )
        ]
        omero_metadata = Omero(channels=channels)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_omero_multiscale.ome.zarr"
            
            # Create multiscale OME-Zarr with OMERO metadata
            ome_zarr = OmeZarrImage(
                path=output_path,
                image=image,
                dims=["c", "y", "x"],
                axis_units={"c": None, "y": "micrometer", "x": "micrometer"},
                scale_transformations={"y": 0.1, "x": 0.1},
                downscale_levels=2,
                downscale_factor=2,
                omero_metadata=omero_metadata,
                overwrite=True
            )
            
            ome_zarr.write()
            
            # Verify multiscale structure with OMERO metadata
            root = zarr.open(str(output_path), mode='r')
            
            # Check that multiple resolution levels exist
            assert '0' in root
            assert '1' in root
            assert '2' in root
            
            # Check OMERO metadata is present
            ome_attrs = root.attrs['ome']
            assert 'omero' in ome_attrs
            omero_attrs = ome_attrs['omero']
            assert len(omero_attrs['channels']) == 2

    def test_omero_metadata_parameter_validation(self):
        """Test that omero_metadata parameter validation works correctly."""
        image = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_validation.ome.zarr"
            
            # Test with None (should work)
            ome_zarr = OmeZarrImage(
                path=output_path,
                image=image,
                dims=["y", "x"],
                axis_units={"y": "micrometer", "x": "micrometer"},
                omero_metadata=None,
                overwrite=True
            )
            
            # Should not raise any errors
            assert ome_zarr.omero_metadata is None

    def test_omero_metadata_stored_correctly(self):
        """Test that omero_metadata is stored as instance variable."""
        image = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
        
        channel = Channel(
            window=Window(start=0.0, min=0.0, end=255.0, max=255.0),
            label="Test Channel",
            color="FF0000",
            active=True
        )
        omero_metadata = Omero(channels=[channel])
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_storage.ome.zarr"
            
            ome_zarr = OmeZarrImage(
                path=output_path,
                image=image,
                dims=["y", "x"],
                axis_units={"y": "micrometer", "x": "micrometer"},
                omero_metadata=omero_metadata,
                overwrite=True
            )
            
            # Check that omero_metadata is stored correctly
            assert ome_zarr.omero_metadata is omero_metadata
            assert len(ome_zarr.omero_metadata.channels) == 1
            assert ome_zarr.omero_metadata.channels[0].label == "Test Channel"

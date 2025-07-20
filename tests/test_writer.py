"""Tests for the OME-Zarr writer."""

import pytest
import numpy as np
from pathlib import Path
from ome_zarr_writer import OMEZarrWriter


def test_writer_initialization(temp_dir):
    """Test that the writer can be initialized."""
    writer = OMEZarrWriter(temp_dir / "test.zarr")
    assert writer.path == temp_dir / "test.zarr"
    assert not writer.overwrite


def test_writer_initialization_with_overwrite(temp_dir):
    """Test that the writer can be initialized with overwrite option."""
    writer = OMEZarrWriter(temp_dir / "test.zarr", overwrite=True)
    assert writer.overwrite


def test_write_image_not_implemented(temp_dir, sample_image):
    """Test that write_image raises NotImplementedError."""
    writer = OMEZarrWriter(temp_dir / "test.zarr")
    with pytest.raises(NotImplementedError):
        writer.write_image(sample_image, pixel_size=(1.0, 1.0))


def test_create_multiscale_group_not_implemented(temp_dir, sample_image):
    """Test that create_multiscale_group raises NotImplementedError."""
    writer = OMEZarrWriter(temp_dir / "test.zarr")
    with pytest.raises(NotImplementedError):
        writer.create_multiscale_group(
            arrays=[sample_image],
            axes=[{"name": "y", "type": "space", "unit": "micrometer"}]
        )

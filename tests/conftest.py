"""Test configuration and fixtures."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)


@pytest.fixture
def valid_fileset(tmp_path):
    """A small valid OME-Zarr fileset with two OMERO channels and one label image."""
    from ome_zarr_io import Writer
    from ome_zarr_io.schema_models import Channel, Omero, Window

    path = tmp_path / "valid.ome.zarr"
    writer = Writer(
        path=path,
        image=np.zeros((2, 20, 20), dtype=np.uint8),
        dims=["c", "y", "x"],
        axis_units={"y": "micrometer", "x": "micrometer"},
        scale_transformations={"y": 0.5, "x": 0.5},
        downscale_levels=1,
        omero_metadata=Omero(
            channels=[
                Channel(
                    label="DAPI", color="0000FF", window=Window(0, 0, 255, 255)
                ),
                Channel(label="GFP", color="00FF00"),
            ]
        ),
    )
    writer.write()
    writer.add_labels(
        name="nuclei",
        array=np.ones((2, 20, 20), dtype=np.uint8),
        colors=[{"label-value": 1, "rgba": [255, 0, 0, 255]}],
    )
    return path

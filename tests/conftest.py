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
def rename_levels():
    """Rename the level arrays of an image or label group and update its metadata.

    ``rename_levels(group_path, {"0": "s0", "1": "s1"})`` moves the array directories
    and rewrites the ``path`` of each entry in ``multiscales[0].datasets``, as
    another OME-Zarr writer might have named them.
    """
    import json

    def _rename(group_path, new_names):
        group_path = Path(group_path)
        meta_file = group_path / "zarr.json"
        meta = json.loads(meta_file.read_text())
        for dataset in meta["attributes"]["ome"]["multiscales"][0]["datasets"]:
            old, new = dataset["path"], new_names[dataset["path"]]
            shutil.move(str(group_path / old), str(group_path / new))
            dataset["path"] = new
        meta_file.write_text(json.dumps(meta))

    return _rename


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

"""Tests for the Reader class."""

import numpy as np
import pytest

from ome_zarr_io.writer import Writer
from ome_zarr_io.reader import Reader
from ome_zarr_io.schema_models import (
    Axis,
    Channel,
    Dataset,
    Multiscale,
    OMEMetadata,
    OMEZarrImageMetadata,
    Omero,
    ScaleTransformation,
)


@pytest.fixture
def channel_image():
    """Sample multichannel (C, Y, X) image data."""
    return np.random.randint(0, 255, size=(2, 50, 50), dtype=np.uint8)


@pytest.fixture
def label_array():
    """Sample label array matching `channel_image`'s shape."""
    return np.random.randint(0, 5, size=(2, 50, 50), dtype=np.uint8)


@pytest.fixture
def written_image_path(tmp_path, channel_image, label_array):
    """Write a small OME-Zarr fileset with channels and a label, return its path."""
    path = tmp_path / "test.zarr"

    omero_metadata = Omero(
        channels=[Channel(label="DAPI"), Channel(label="GFP")]
    )

    writer = Writer(
        path=path,
        image=channel_image,
        dims=["c", "y", "x"],
        axis_units={"y": "micrometer", "x": "micrometer"},
        scale_transformations={"y": 0.5, "x": 0.5},
        downscale_levels=1,
        downscale_factor=2.0,
        omero_metadata=omero_metadata,
        overwrite=True,
    )
    writer.write()
    writer.add_labels(name="nuclei", array=label_array)

    return path


@pytest.fixture
def reader(written_image_path):
    """A Reader opened on the fixture fileset."""
    return Reader(written_image_path)


class TestInit:
    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Reader(tmp_path / "does_not_exist.zarr")


class TestMetadataAccessors:
    def test_dims(self, reader):
        assert reader.dims == ["c", "y", "x"]

    def test_axes_types(self, reader):
        axis_types = [axis.type for axis in reader.axes]
        assert axis_types == ["channel", "space", "space"]

    def test_n_levels(self, reader):
        assert reader.n_levels == 2


class TestChannels:
    def test_channel_names(self, reader):
        assert reader.channel_names == ["DAPI", "GFP"]

    def test_get_channel_dask_matches_source(self, reader, channel_image):
        result = reader.get_channel("GFP")
        assert result.shape == (50, 50)
        np.testing.assert_array_equal(result.compute(), channel_image[1])

    def test_get_channel_numpy(self, reader, channel_image):
        result = reader.get_channel("DAPI", as_type="numpy")
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, channel_image[0])

    def test_get_channel_unknown_raises_key_error(self, reader):
        with pytest.raises(KeyError):
            reader.get_channel("does-not-exist")

    def test_get_channel_index_unaffected_by_unlabeled_channels(self, tmp_path):
        """A preceding unlabeled channel must not shift later channels' indices."""
        path = tmp_path / "mixed_labels.zarr"
        image = np.random.randint(0, 255, size=(2, 50, 50), dtype=np.uint8)
        omero_metadata = Omero(channels=[Channel(label=None), Channel(label="GFP")])

        writer = Writer(
            path=path,
            image=image,
            dims=["c", "y", "x"],
            axis_units={"y": "micrometer", "x": "micrometer"},
            omero_metadata=omero_metadata,
            overwrite=True,
        )
        writer.write()

        reader = Reader(path)
        result = reader.get_channel("GFP", as_type="numpy")
        np.testing.assert_array_equal(result, image[1])

    def test_get_channel_no_channel_axis_raises(self, tmp_path):
        path = tmp_path / "no_channel.zarr"
        image = np.random.randint(0, 255, size=(50, 50), dtype=np.uint8)
        writer = Writer(
            path=path,
            image=image,
            dims=["y", "x"],
            axis_units={"y": "micrometer", "x": "micrometer"},
            overwrite=True,
        )
        writer.write()

        reader = Reader(path)
        with pytest.raises(ValueError):
            reader.get_channel("anything")


class TestLabels:
    def test_label_names(self, reader):
        assert reader.label_names == ["nuclei"]

    def test_get_label_dask_matches_source(self, reader, label_array):
        result = reader.get_label("nuclei")
        assert result.shape == label_array.shape
        np.testing.assert_array_equal(result.compute(), label_array)

    def test_get_label_numpy(self, reader, label_array):
        result = reader.get_label("nuclei", as_type="numpy")
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, label_array)

    def test_get_label_unknown_raises_key_error(self, reader):
        with pytest.raises(KeyError):
            reader.get_label("does-not-exist")

    def test_no_labels_returns_empty_list(self, tmp_path):
        path = tmp_path / "no_labels.zarr"
        image = np.random.randint(0, 255, size=(50, 50), dtype=np.uint8)
        writer = Writer(
            path=path,
            image=image,
            dims=["y", "x"],
            axis_units={"y": "micrometer", "x": "micrometer"},
            overwrite=True,
        )
        writer.write()

        reader = Reader(path)
        assert reader.label_names == []


class TestValidation:
    def test_validate_valid_fileset(self, reader):
        assert reader.validate() is True

    def test_get_validation_errors_empty_for_valid_fileset(self, reader):
        assert reader.get_validation_errors() == []


class TestPhysicalSize:
    def test_get_physical_size_level_0(self, reader):
        sizes = reader.get_physical_size(level=0)
        assert set(sizes.keys()) == {"y", "x"}
        assert sizes["y"].value == pytest.approx(0.5)
        assert sizes["x"].value == pytest.approx(0.5)
        assert sizes["y"].unit == "micrometer"
        assert sizes["x"].unit == "micrometer"

    def test_get_physical_size_downscaled_level(self, reader):
        sizes = reader.get_physical_size(level=1)
        # Level 1 was downscaled by a factor of 2.0
        assert sizes["y"].value == pytest.approx(1.0)
        assert sizes["x"].value == pytest.approx(1.0)

    def test_get_physical_size_out_of_range_raises(self, reader):
        with pytest.raises(IndexError):
            reader.get_physical_size(level=99)

    def test_get_voxel_size(self, reader):
        voxel_size = reader.get_voxel_size()
        assert voxel_size == {"y": pytest.approx(0.5), "x": pytest.approx(0.5)}

    def test_get_voxel_size_missing_unit_raises(self, tmp_path):
        path = tmp_path / "no_unit.zarr"
        axes = [
            Axis(name="y", type="space", unit=None),
            Axis(name="x", type="space", unit=None),
        ]
        dataset = Dataset(
            path="0", coordinateTransformations=[ScaleTransformation(scale=[1.0, 1.0])]
        )
        multiscale = Multiscale(datasets=[dataset], axes=axes)
        metadata = OMEZarrImageMetadata(ome=OMEMetadata(multiscales=[multiscale], version="0.5"))

        import zarr

        root = zarr.create_group(str(path), zarr_format=3)
        root.create_array(name="0", shape=(10, 10), dtype="uint8")
        root.attrs.update(metadata.to_dict())

        reader = Reader(path)
        with pytest.raises(ValueError):
            reader.get_voxel_size()

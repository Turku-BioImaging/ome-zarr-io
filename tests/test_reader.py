"""Tests for the Reader class."""

import json
import shutil

import numpy as np
import pytest
from jsonschema.exceptions import ValidationError
from invalid_cases import make_broken_copy

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


def _write_fileset_with_paths(path, levels, dataset_paths):
    """Write a (C, Y, X) OME-Zarr fileset whose level arrays use `dataset_paths`.

    A matching `labels/nuclei` label image is written with the same paths.
    """
    import zarr

    axes = [
        Axis(name="c", type="channel"),
        Axis(name="y", type="space", unit="micrometer"),
        Axis(name="x", type="space", unit="micrometer"),
    ]
    datasets = [
        Dataset(
            path=dataset_path,
            coordinateTransformations=[ScaleTransformation(scale=[1.0, 2.0**i, 2.0**i])],
        )
        for i, dataset_path in enumerate(dataset_paths)
    ]
    metadata = OMEZarrImageMetadata(
        ome=OMEMetadata(
            multiscales=[Multiscale(datasets=datasets, axes=axes)],
            version="0.5",
            omero=Omero(channels=[Channel(label="DAPI"), Channel(label="GFP")]),
        )
    )

    root = zarr.create_group(str(path), zarr_format=3)
    root.attrs.update(metadata.to_dict())
    for dataset_path, level in zip(dataset_paths, levels):
        root.create_array(name=dataset_path, shape=level.shape, dtype=level.dtype)[:] = level

    labels = root.create_group("labels")
    labels.attrs.update({"ome": {"version": "0.5", "labels": ["nuclei"]}})
    label_group = labels.create_group("nuclei")
    label_group.attrs.update(
        {"ome": {**metadata.to_dict()["ome"], "image-label": {}}}
    )
    for dataset_path, level in zip(dataset_paths, levels):
        label_group.create_array(name=dataset_path, shape=level.shape, dtype=level.dtype)[:] = level


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

    def test_get_channel_uses_dataset_path(self, tmp_path):
        """Levels are located via `datasets[level].path`, not the level index."""
        path = tmp_path / "custom_paths.zarr"
        levels = [
            np.random.randint(0, 255, size=(2, 20, 20), dtype=np.uint8),
            np.random.randint(0, 255, size=(2, 10, 10), dtype=np.uint8),
        ]
        _write_fileset_with_paths(path, levels, ["s0", "s1"])

        reader = Reader(path)
        np.testing.assert_array_equal(
            reader.get_channel("GFP", level=0, as_type="numpy"), levels[0][1]
        )
        np.testing.assert_array_equal(
            reader.get_channel("DAPI", level=1, as_type="numpy"), levels[1][0]
        )

    def test_get_channel_negative_level(self, reader):
        assert reader.get_channel("DAPI", level=-1).shape == reader.get_channel(
            "DAPI", level=1
        ).shape

    def test_get_channel_level_out_of_range_raises(self, reader):
        with pytest.raises(IndexError):
            reader.get_channel("DAPI", level=99)


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

    def test_get_label_uses_dataset_path(self, tmp_path):
        """Label levels are located via the label group's own `datasets[level].path`."""
        path = tmp_path / "custom_label_paths.zarr"
        levels = [
            np.random.randint(0, 5, size=(2, 20, 20), dtype=np.uint8),
            np.random.randint(0, 5, size=(2, 10, 10), dtype=np.uint8),
        ]
        _write_fileset_with_paths(path, levels, ["s0", "s1"])

        reader = Reader(path)
        np.testing.assert_array_equal(
            reader.get_label("nuclei", level=1, as_type="numpy"), levels[1]
        )

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

    # Broken images the Reader can still open (metadata parses but is invalid).
    @pytest.mark.parametrize(
        "mutate,label",
        [
            (lambda a: a["ome"].update(version="0.4"), None),
            (lambda a: a["ome"]["multiscales"][0]["datasets"][0].update(path=0), None),
            (lambda a: a["ome"].pop("image-label"), "nuclei"),
            (lambda a: a["ome"]["image-label"].update(source="x"), "nuclei"),
        ],
        ids=["old_version", "path_not_string", "label_no_image_label", "label_source_not_object"],
    )
    def test_validate_rejects_broken_fileset(self, written_image_path, tmp_path, mutate, label):
        broken = make_broken_copy(written_image_path, tmp_path / "broken.zarr", mutate, label)
        reader = Reader(broken)

        with pytest.raises(ValidationError):
            reader.validate()
        assert reader.get_validation_errors() != []
        assert not reader.report()

    def test_errors_in_label_are_found_even_if_image_is_valid(self, written_image_path, tmp_path):
        broken = make_broken_copy(
            written_image_path, tmp_path / "broken.zarr", lambda a: a["ome"].pop("image-label"), "nuclei"
        )
        errors = Reader(broken).get_validation_errors()
        assert len(errors) == 1 and "image-label" in errors[0]


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


class TestLevelArrayNames:
    """Level arrays are found through the metadata, not assumed to be named 0, 1, ..."""

    def test_renamed_arrays_are_read(self, written_image_path, rename_levels):
        before = Reader(written_image_path)
        expected = {
            level: before.get_channel("GFP", level=level, as_type="numpy")
            for level in (0, 1)
        }
        expected_label = before.get_label("nuclei", level=1, as_type="numpy")

        rename_levels(written_image_path, {"0": "s0", "1": "s1"})
        rename_levels(written_image_path / "labels" / "nuclei", {"0": "l0", "1": "l1"})

        after = Reader(written_image_path)
        for level in (0, 1):
            np.testing.assert_array_equal(
                after.get_channel("GFP", level=level, as_type="numpy"), expected[level]
            )
        np.testing.assert_array_equal(
            after.get_label("nuclei", level=1, as_type="numpy"), expected_label
        )
        assert after.validate()

    def test_swapped_names_follow_the_metadata_order(self, tmp_path):
        """Level 0 is the first dataset in the metadata even if it is named "1"."""
        path = tmp_path / "swapped.zarr"
        levels = [
            np.full((2, 20, 20), 7, dtype=np.uint8),
            np.full((2, 10, 10), 9, dtype=np.uint8),
        ]
        _write_fileset_with_paths(path, levels, ["1", "0"])
        reader = Reader(path)
        assert reader.get_channel("DAPI", level=0, as_type="numpy").shape == (20, 20)
        assert (reader.get_channel("DAPI", level=0, as_type="numpy") == 7).all()
        assert (reader.get_channel("DAPI", level=1, as_type="numpy") == 9).all()

    def test_missing_array_is_a_clear_error(self, written_image_path):
        shutil.rmtree(written_image_path / "1")
        with pytest.raises(
            KeyError, match="Array '1' listed in the multiscales metadata for level 1"
        ):
            Reader(written_image_path).get_channel("DAPI", level=1)

    def test_channel_level_out_of_range_message(self, reader):
        with pytest.raises(IndexError, match="Level 99 is out of range: .* 2 level"):
            reader.get_channel("DAPI", level=99)

    def test_negative_level_counts_from_the_coarsest(self, reader):
        last = reader.get_channel("DAPI", level=-1, as_type="numpy")
        np.testing.assert_array_equal(
            last, reader.get_channel("DAPI", level=1, as_type="numpy")
        )
        assert last.shape == (25, 25)


class TestLabelLevels:
    def test_negative_level(self, reader):
        np.testing.assert_array_equal(
            reader.get_label("nuclei", level=-1, as_type="numpy"),
            reader.get_label("nuclei", level=1, as_type="numpy"),
        )

    def test_level_out_of_range(self, reader):
        with pytest.raises(IndexError, match="out of range"):
            reader.get_label("nuclei", level=2)

    def test_label_without_multiscales_metadata(self, written_image_path):
        label_meta = written_image_path / "labels" / "nuclei" / "zarr.json"
        meta = json.loads(label_meta.read_text())
        del meta["attributes"]["ome"]["multiscales"]
        label_meta.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="Label 'nuclei' has no valid multiscales"):
            Reader(written_image_path).get_label("nuclei")

    def test_label_missing_array(self, written_image_path):
        shutil.rmtree(written_image_path / "labels" / "nuclei" / "1")
        with pytest.raises(KeyError, match="Array '1' listed in the multiscales"):
            Reader(written_image_path).get_label("nuclei", level=1)

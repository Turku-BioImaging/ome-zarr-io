"""Tests for Writer.from_existing: adding labels to a fileset that is already on disk."""

import json
import shutil

import dask.array as da
import numpy as np
import pytest
import zarr

from ome_zarr_io import PlateWriter, Reader, Writer, validate

UNITS = {"t": "second", "z": "micrometer", "y": "micrometer", "x": "micrometer"}


def write_fileset(path, shape, dims, scale=None, **kwargs):
    units = {d: UNITS[d] for d in dims if d in UNITS}
    data = np.random.default_rng(0).integers(0, 255, shape).astype(np.uint8)
    writer = Writer(
        path, data, list(dims), units, scale_transformations=scale, **kwargs
    )
    writer.write()
    return writer


def label_shape(shape, dims):
    """A label with every non-spatial axis of size 1, as the spec allows."""
    return tuple(1 if d in "tcz" else s for d, s in zip(dims, shape))


def read_ome(group_path):
    meta = json.loads((group_path / "zarr.json").read_text())
    return meta["attributes"]["ome"]


def edit_ome(group_path, change):
    file = group_path / "zarr.json"
    meta = json.loads(file.read_text())
    change(meta["attributes"]["ome"])
    file.write_text(json.dumps(meta))


def label_levels(path, name):
    n = len(read_ome(path / "labels" / name)["multiscales"][0]["datasets"])
    return [zarr.open_array(str(path / "labels" / name / str(i)))[:] for i in range(n)]


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "img.ome.zarr"
    write_fileset(
        path,
        (2, 3, 64, 64),
        "czyx",
        scale={"z": 2.0, "y": 0.5, "x": 0.5},
        downscale_levels=2,
    )
    return path


class TestEquivalentToTheOriginalWriter:
    """A label added through from_existing must match one added by the Writer that wrote the image."""

    CASES = [
        ("yx", (64, 64), 2, 2, {"y": 0.25, "x": 0.25}),
        ("zyx", (4, 64, 64), 3, 3, {"z": 1.5, "y": 0.1, "x": 0.1}),
        ("czyx", (2, 3, 81, 81), 2, 3, {"z": 2.0, "y": 0.5, "x": 0.5}),
        ("tczyx", (2, 2, 2, 100, 63), 2, 4, {"t": 3.0, "z": 1.0, "y": 0.3, "x": 0.3}),
        ("yx", (65, 65), 3, 2, None),  # odd size, no pixel size given
        ("zyx", (3, 128, 96), 4, 3, {"z": 5.0, "y": 0.325, "x": 0.325}),
    ]

    @pytest.mark.parametrize("dims, shape, levels, factor, scale", CASES)
    def test_label_matches(self, tmp_path, dims, shape, levels, factor, scale):
        path = tmp_path / "img.zarr"
        original = write_fileset(
            path,
            shape,
            dims,
            scale=scale,
            downscale_levels=levels,
            downscale_factor=factor,
        )
        mask = np.random.default_rng(1).integers(
            0, 5, label_shape(shape, dims)
        ).astype(np.uint8)

        original.add_labels("reference", mask)
        Writer.from_existing(path).add_labels("reopened", mask)

        ref, new = (read_ome(path / "labels" / n) for n in ("reference", "reopened"))
        assert new["multiscales"][0]["datasets"] == ref["multiscales"][0]["datasets"]
        assert new["multiscales"][0]["axes"] == ref["multiscales"][0]["axes"]
        assert new["image-label"] == ref["image-label"]
        ref_levels, new_levels = (
            label_levels(path, n) for n in ("reference", "reopened")
        )
        assert len(new_levels) == len(ref_levels) == levels + 1
        for a, b in zip(ref_levels, new_levels):
            np.testing.assert_array_equal(a, b)
        assert validate(path).is_valid

    def test_label_scales_follow_the_image(self, image):
        Writer.from_existing(image).add_labels("cells", np.ones((1, 1, 64, 64), "uint8"))
        image_ds = read_ome(image)["multiscales"][0]["datasets"]
        label_ds = read_ome(image / "labels" / "cells")["multiscales"][0]["datasets"]
        assert label_ds == image_ds
        assert image_ds[0]["coordinateTransformations"][0]["scale"] == [1.0, 2.0, 0.5, 0.5]


class TestWhatIsReadFromDisk:
    def test_attributes(self, tmp_path):
        path = tmp_path / "img.zarr"
        write_fileset(
            path,
            (2, 3, 81, 81),
            "czyx",
            scale={"z": 2.0, "y": 0.5, "x": 0.5},
            downscale_levels=2,
            downscale_factor=3,
            name="my image",
        )
        writer = Writer.from_existing(path)
        assert writer.path == path
        assert writer.dims == ["c", "z", "y", "x"]
        assert [a.name for a in writer.axes] == ["c", "z", "y", "x"]
        assert writer.image.shape == (2, 3, 81, 81)
        assert writer.downscale_factor == 3
        assert writer.downscale_levels == 2
        assert writer.name == "my image"
        assert writer.overwrite is False
        assert writer.coordinate_transformations[0].scale == [1.0, 2.0, 0.5, 0.5]

    def test_single_level_image(self, tmp_path):
        path = tmp_path / "img.zarr"
        write_fileset(path, (64, 64), "yx", scale={"y": 0.5, "x": 0.5})
        writer = Writer.from_existing(path)
        assert writer.downscale_levels == 0
        writer.add_labels("mask", np.ones((64, 64), "uint8"))
        assert len(label_levels(path, "mask")) == 1
        assert validate(path).is_valid

    def test_factor_is_inferred_from_the_scales(self, tmp_path):
        """Both 5//3 and 5//4 are 1, so the shapes alone cannot tell the factor."""
        path = tmp_path / "img.zarr"
        write_fileset(path, (5, 5), "yx", downscale_levels=1, downscale_factor=4)
        assert Writer.from_existing(path).downscale_factor == 4

    def test_pixel_data_is_not_read(self, image):
        """Chunk files are deleted: opening and labelling needs only the metadata."""
        for level in ("0", "1", "2"):
            for chunk in (image / level).iterdir():
                if chunk.name != "zarr.json":
                    shutil.rmtree(chunk) if chunk.is_dir() else chunk.unlink()
        writer = Writer.from_existing(image)
        writer.add_labels("cells", np.ones((1, 1, 64, 64), "uint8"))
        assert validate(image).is_valid

    def test_image_metadata_is_not_modified(self, image):
        before = (image / "zarr.json").read_text()
        Writer.from_existing(image).add_labels("cells", np.ones((1, 1, 64, 64), "uint8"))
        after = json.loads((image / "zarr.json").read_text())
        assert after["attributes"]["ome"]["multiscales"] == json.loads(before)[
            "attributes"
        ]["ome"]["multiscales"]
        assert (image / "labels").exists()

    def test_accepts_str_and_path(self, image):
        assert Writer.from_existing(str(image)).path == image


class TestUsingTheOpenedWriter:
    def test_several_labels(self, image):
        writer = Writer.from_existing(image)
        writer.add_labels("a", np.ones((1, 1, 64, 64), "uint8"))
        writer.add_labels("b", np.full((1, 1, 64, 64), 2, "uint16"))
        assert Reader(image).label_names == ["a", "b"]
        assert validate(image).is_valid

    def test_overwrite_a_label(self, image):
        writer = Writer.from_existing(image)
        writer.add_labels("a", np.ones((1, 1, 64, 64), "uint8"))
        writer.add_labels("a", np.full((1, 1, 64, 64), 7, "uint8"), overwrite=True)
        assert (label_levels(image, "a")[0] == 7).all()
        assert Reader(image).label_names == ["a"]

    def test_label_with_the_full_shape(self, image):
        Writer.from_existing(image).add_labels("a", np.ones((2, 3, 64, 64), "uint8"))
        assert label_levels(image, "a")[0].shape == (2, 3, 64, 64)

    def test_dask_label(self, image):
        mask = da.ones((1, 1, 64, 64), dtype="uint8", chunks=(1, 1, 32, 32))
        Writer.from_existing(image).add_labels("a", mask)
        assert validate(image).is_valid

    def test_wrong_label_shape_is_still_rejected(self, image):
        with pytest.raises(ValueError, match="incompatible with source image shape"):
            Writer.from_existing(image).add_labels("a", np.ones((1, 1, 32, 32), "uint8"))
        assert not (image / "labels").exists()

    def test_label_level_count_matches_the_image(self, image):
        writer = Writer.from_existing(image)
        with pytest.raises(ValueError, match="same number of levels as its image"):
            writer.add_labels("a", np.ones((1, 1, 64, 64), "uint8"), downscale_levels=0)

    def test_write_is_disabled(self, image):
        writer = Writer.from_existing(image)
        before = sorted(p.relative_to(image).as_posix() for p in image.rglob("*"))
        with pytest.raises(RuntimeError, match="write\\(\\) is disabled"):
            writer.write()
        assert sorted(p.relative_to(image).as_posix() for p in image.rglob("*")) == before

    def test_write_is_still_disabled_with_overwrite(self, image):
        writer = Writer.from_existing(image)
        writer.overwrite = True
        with pytest.raises(RuntimeError):
            writer.write()
        assert (image / "0").exists()

    def test_a_normal_writer_can_still_write(self, tmp_path):
        write_fileset(tmp_path / "ok.zarr", (16, 16), "yx")
        assert (tmp_path / "ok.zarr" / "0").exists()


class TestPlateFields:
    def test_label_on_a_plate_field(self, tmp_path):
        path = tmp_path / "plate.zarr"
        image = np.zeros((2, 32, 32), dtype=np.uint8)
        with PlateWriter(path, rows=["A"], columns=["1"]) as plate:
            plate.add_field(
                "A",
                "1",
                image,
                ["c", "y", "x"],
                {"y": "micrometer", "x": "micrometer"},
                scale_transformations={"y": 0.5, "x": 0.5},
                downscale_levels=1,
            )
        Writer.from_existing(path / "A" / "1" / "0").add_labels(
            "cells", np.ones((1, 32, 32), "uint8")
        )
        report = validate(path)
        assert report.is_valid, report.errors
        assert Reader(path / "A" / "1" / "0").label_names == ["cells"]

    def test_plate_root_is_rejected_with_a_hint(self, tmp_path):
        path = tmp_path / "plate.zarr"
        with PlateWriter(path, rows=["A"], columns=["1"]) as plate:
            plate.add_field(
                "A", "1", np.zeros((8, 8), "uint8"), ["y", "x"],
                {"y": "micrometer", "x": "micrometer"},
            )
        with pytest.raises(ValueError, match="open one of its fields"):
            Writer.from_existing(path)


class TestImageFromAnotherTool:
    """An image built directly with zarr, as other OME-Zarr writers would."""

    def test_hand_built_image(self, tmp_path):
        path = tmp_path / "other.zarr"
        group = zarr.create_group(str(path), zarr_format=3)
        shapes = [(3, 200, 200), (3, 100, 100), (3, 50, 50)]
        for i, shape in enumerate(shapes):
            group.create_array(str(i), shape=shape, dtype="uint16", dimension_names=["c", "y", "x"])
        group.attrs["ome"] = {
            "version": "0.5",
            "multiscales": [
                {
                    "name": "from another tool",
                    "axes": [
                        {"name": "c", "type": "channel"},
                        {"name": "y", "type": "space", "unit": "micrometer"},
                        {"name": "x", "type": "space", "unit": "micrometer"},
                    ],
                    "datasets": [
                        {
                            "path": str(i),
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1.0, 0.325 * 2**i, 0.325 * 2**i]}
                            ],
                        }
                        for i in range(3)
                    ],
                }
            ],
        }
        writer = Writer.from_existing(path)
        assert writer.downscale_factor == 2 and writer.downscale_levels == 2
        writer.add_labels("nuclei", np.ones((1, 200, 200), "uint32"))
        label_ds = read_ome(path / "labels" / "nuclei")["multiscales"][0]["datasets"]
        image_ds = read_ome(path)["multiscales"][0]["datasets"]
        assert label_ds == image_ds
        report = validate(path)
        assert report.is_valid, report.errors


class TestUnsupportedInputs:
    def test_missing_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Writer.from_existing(tmp_path / "nope.zarr")

    def test_empty_directory(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(ValueError, match="not an OME-Zarr 0.5 image group"):
            Writer.from_existing(tmp_path / "empty")

    def test_zarr_array_path(self, image):
        with pytest.raises(ValueError, match="not an OME-Zarr 0.5 image group"):
            Writer.from_existing(image / "0")

    def test_group_without_ome_metadata(self, tmp_path):
        zarr.create_group(str(tmp_path / "plain.zarr"), zarr_format=3)
        with pytest.raises(ValueError, match="not an OME-Zarr 0.5 image group"):
            Writer.from_existing(tmp_path / "plain.zarr")

    def test_level_array_missing(self, image):
        shutil.rmtree(image / "1")
        with pytest.raises(ValueError, match="array '1' listed for level 1 does not exist"):
            Writer.from_existing(image)

    def test_translation(self, image):
        edit_ome(
            image,
            lambda o: o["multiscales"][0]["datasets"][0]["coordinateTransformations"].append(
                {"type": "translation", "translation": [0.0, 0.0, 5.0, 5.0]}
            ),
        )
        with pytest.raises(ValueError, match="level 0 has a translation"):
            Writer.from_existing(image)

    def test_multiscale_level_transformations(self, image):
        edit_ome(
            image,
            lambda o: o["multiscales"][0].update(
                coordinateTransformations=[{"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0]}]
            ),
        )
        with pytest.raises(ValueError, match="multiscale has its own coordinateTransformations"):
            Writer.from_existing(image)

    def test_z_is_downscaled(self, image):
        def change(o):
            o["multiscales"][0]["datasets"][1]["coordinateTransformations"][0]["scale"][1] = 4.0

        edit_ome(image, change)
        with pytest.raises(ValueError, match="level 1 has scale .* gives"):
            Writer.from_existing(image)

    def test_non_integer_factor(self, image):
        def change(o):
            for level, factor in ((1, 2.5), (2, 6.25)):
                scale = o["multiscales"][0]["datasets"][level]["coordinateTransformations"][0]["scale"]
                scale[-2:] = [0.5 * factor, 0.5 * factor]

        edit_ome(image, change)
        with pytest.raises(ValueError, match="not an integer factor"):
            Writer.from_existing(image)

    def test_shapes_do_not_follow_the_factor(self, image):
        shutil.rmtree(image / "1")
        group = zarr.open_group(str(image), mode="a")
        group.create_array("1", shape=(2, 3, 30, 30), dtype="uint8", dimension_names=["c", "z", "y", "x"])
        with pytest.raises(ValueError, match="level 1 has shape .* gives"):
            Writer.from_existing(image)

    def test_non_spatial_axis_is_downscaled(self, image):
        shutil.rmtree(image / "1")
        group = zarr.open_group(str(image), mode="a")
        group.create_array("1", shape=(2, 2, 32, 32), dtype="uint8", dimension_names=["c", "z", "y", "x"])
        with pytest.raises(ValueError, match="level 1 has shape"):
            Writer.from_existing(image)

    def test_error_names_the_image(self, image):
        edit_ome(
            image,
            lambda o: o["multiscales"][0]["datasets"][0]["coordinateTransformations"].append(
                {"type": "translation", "translation": [0.0, 0.0, 1.0, 1.0]}
            ),
        )
        with pytest.raises(ValueError, match="Cannot add labels to .*img.ome.zarr"):
            Writer.from_existing(image)

    def test_nothing_is_written_when_rejected(self, image):
        edit_ome(
            image,
            lambda o: o["multiscales"][0].update(coordinateTransformations=[{"type": "scale", "scale": [1.0] * 4}]),
        )
        before = sorted(p.relative_to(image).as_posix() for p in image.rglob("*"))
        with pytest.raises(ValueError):
            Writer.from_existing(image)
        assert sorted(p.relative_to(image).as_posix() for p in image.rglob("*")) == before


class TestImageWithRenamedLevelArrays:
    """Level arrays named by another tool (not "0", "1", ...) work too."""

    def test_labels_can_be_added(self, tmp_path, rename_levels):
        path = tmp_path / "img.zarr"
        write_fileset(
            path,
            (2, 3, 64, 64),
            "czyx",
            scale={"z": 2.0, "y": 0.5, "x": 0.5},
            downscale_levels=2,
            channels=["a", "b"],
        )
        rename_levels(path, {"0": "s0", "1": "s1", "2": "s2"})

        writer = Writer.from_existing(path)
        assert writer.image.shape == (2, 3, 64, 64)  # level 0, not another level
        assert writer.downscale_levels == 2
        writer.add_labels("mask", np.ones((1, 3, 64, 64), dtype=np.uint8))

        assert len(label_levels(path, "mask")) == 3
        assert sorted(p.name for p in path.iterdir() if p.is_dir()) == [
            "labels",
            "s0",
            "s1",
            "s2",
        ]  # the image's arrays are untouched
        assert validate(path).is_valid

    def test_pixel_data_comes_from_the_renamed_level_zero(self, tmp_path, rename_levels):
        path = tmp_path / "img.zarr"
        write_fileset(path, (64, 64), "yx", downscale_levels=1)
        rename_levels(path, {"0": "full", "1": "half"})
        writer = Writer.from_existing(path)
        np.testing.assert_array_equal(
            writer.image.compute(), zarr.open_array(str(path / "full"))[:]
        )

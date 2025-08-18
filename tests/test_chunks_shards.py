"""This module tests chunking and sharding functionality in the OME-Zarr writer."""

import pytest
import numpy as np
import zarr
from ome_zarr_writer.image import OmeZarrImage


class TestChunksAndShardsAPI:
    """ 
    This is the documentation for TestChunksAndShardsAPI class, used for testing
    chunking and sharding functionality with the new write() method API.
    """

    @pytest.fixture
    def sample_2D_image(self):
        return np.random.randint(0, 255, size=(128, 128), dtype=np.uint8)

    @pytest.fixture
    def sample_4D_image(self):
        return np.random.randint(0, 255, size=(2, 5, 64, 64), dtype=np.uint16)

    def test_chunk_size_at_level0_matching_custom_chunks(self,tmp_path,sample_2D_image):    
        output_path = tmp_path / "test_chunks_write.zarr"
        dims = ["y", "x"]
        axis_units = {"y": "micrometer", "x": "micrometer"}
        custom_chunks = (32, 32)

        writer = OmeZarrImage(
            path=output_path,
            image=sample_2D_image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )
        writer.write(chunks=custom_chunks)

        group = zarr.open_group(str(output_path), mode="r")
        level_0 = group["0"]
        assert level_0.chunks == custom_chunks

    def test_successful_writing_if_shards_parameter_in_write_method(self, tmp_path, sample_2D_image):
        """If supported by zarr version, tests writing with shards parameter."""
        output_path = tmp_path / "test_shards_write.zarr"
        dims = ["y", "x"]
        axis_units = {"y": "micrometer", "x": "micrometer"}
        custom_shards = (128, 128)

        writer = OmeZarrImage(
            path=output_path,
            image=sample_2D_image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )
        writer.write(shards=custom_shards)

        group = zarr.open_group(str(output_path), mode="r")
        level_0 = group["0"]

        assert level_0.shape == sample_2D_image.shape

    def test_chunks_and_shards_compatibility(self, tmp_path, sample_4D_image):
        output_path = tmp_path / "test_chunks_shards_together.zarr"
        dims = ["c", "z", "y", "x"]
        axis_units = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}

        # Use shards that are divisible by chunks
        custom_chunks = (1, 2, 8, 8)
        custom_shards = (2, 4, 64, 64)

        writer = OmeZarrImage(
            path=output_path,
            image=sample_4D_image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )

        writer.write(chunks=custom_chunks, shards=custom_shards)

        assert output_path.exists()
        group = zarr.open_group(str(output_path), mode="r")
        assert "0" in group

    def test_chunk_size_at_each_level(self, tmp_path, sample_2D_image):
        output_path = tmp_path / "test_chunks_multiscale.zarr"
        dims = ["y", "x"]
        axis_units = {"y": "micrometer", "x": "micrometer"}
        custom_chunks = (32, 32)

        writer = OmeZarrImage(
            path=output_path,
            image=sample_2D_image,
            dims=dims,
            axis_units=axis_units,
            downscale_levels=2,
            overwrite=True,
        )
        writer.write(chunks=custom_chunks)

        group = zarr.open_group(str(output_path), mode="r")
        for level_name in group.keys():
            if level_name.isdigit():
                level_array = group[level_name]
                assert level_array.chunks == custom_chunks

    def test_writing_without_chunks_and_shards(self, tmp_path, sample_2D_image):
        output_path = tmp_path / "test_constructor.zarr"
        dims = ["y", "x"]
        axis_units = {"y": "micrometer", "x": "micrometer"}

        writer = OmeZarrImage(
            path=output_path,
            image=sample_2D_image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )

        assert not hasattr(writer, "chunks")
        assert not hasattr(writer, "shards")

        writer.write()
        assert output_path.exists()

    def test_data_integrity_if_writing_with_chunks(self, tmp_path, sample_2D_image):
        output_path = tmp_path / "test_integrity.zarr"
        dims = ["y", "x"]
        axis_units = {"y": "micrometer", "x": "micrometer"}
        custom_chunks = (64, 64)

        writer = OmeZarrImage(
            path=output_path,
            image=sample_2D_image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )

        writer.write(chunks=custom_chunks)

        # Verify data integrity
        group = zarr.open_group(str(output_path), mode="r")
        level_0 = group["0"]
        np.testing.assert_array_equal(level_0[:], sample_2D_image)

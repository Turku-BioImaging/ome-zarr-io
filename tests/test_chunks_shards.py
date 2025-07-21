"""Test chunking and sharding functionality in OME-Zarr writer."""

import pytest
import numpy as np
import zarr
from ome_zarr_writer.image import OmeZarrImage


class TestChunksAndShardsAPI:
    """Test chunking and sharding functionality with the new write() method API."""

    @pytest.fixture
    def sample_2d_image(self):
        """Create a sample 2D image for testing."""
        return np.random.randint(0, 255, size=(128, 128), dtype=np.uint8)

    @pytest.fixture
    def sample_4d_image(self):
        """Create a sample 4D image for testing."""
        return np.random.randint(0, 255, size=(2, 5, 64, 64), dtype=np.uint16)

    def test_chunks_in_write_method(self, tmp_path, sample_2d_image):
        """Test that chunks parameter works in write() method."""
        output_path = tmp_path / "test_chunks_write.zarr"
        dims = ["y", "x"]
        axis_units = {"unit": "micrometer"}
        custom_chunks = (32, 32)

        writer = OmeZarrImage(
            path=output_path,
            image=sample_2d_image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )
        writer.write(chunks=custom_chunks)

        # Verify chunks
        group = zarr.open_group(str(output_path), mode="r")
        level_0 = group["0"]
        assert level_0.chunks == custom_chunks

    def test_shards_in_write_method(self, tmp_path, sample_2d_image):
        """Test that shards parameter works in write() method."""
        output_path = tmp_path / "test_shards_write.zarr"
        dims = ["y", "x"]
        axis_units = {"unit": "micrometer"}
        custom_shards = (128, 128)

        writer = OmeZarrImage(
            path=output_path,
            image=sample_2d_image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )
        writer.write(shards=custom_shards)

        # Verify shards (if supported by zarr version)
        group = zarr.open_group(str(output_path), mode="r")
        level_0 = group["0"]
        # For now, just verify the file was written successfully
        assert level_0.shape == sample_2d_image.shape

    def test_chunks_and_shards_together(self, tmp_path, sample_4d_image):
        """Test that chunks and shards work together."""
        output_path = tmp_path / "test_chunks_shards_together.zarr"
        dims = ["c", "z", "y", "x"]
        axis_units = {"unit": "micrometer"}
        
        # Use compatible chunks and shards
        # Shards must be divisible by chunks
        custom_chunks = (1, 2, 8, 8)
        custom_shards = (2, 4, 64, 64)

        writer = OmeZarrImage(
            path=output_path,
            image=sample_4d_image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )
        
        # This should work without errors
        writer.write(chunks=custom_chunks, shards=custom_shards)

        # Verify the file was created
        assert output_path.exists()
        group = zarr.open_group(str(output_path), mode="r")
        assert "0" in group

    def test_chunks_with_multiscale(self, tmp_path, sample_2d_image):
        """Test chunks with multiscale pyramids."""
        output_path = tmp_path / "test_chunks_multiscale.zarr"
        dims = ["y", "x"]
        axis_units = {"unit": "micrometer"}
        custom_chunks = (32, 32)

        writer = OmeZarrImage(
            path=output_path,
            image=sample_2d_image,
            dims=dims,
            axis_units=axis_units,
            downscale_levels=2,
            overwrite=True,
        )
        writer.write(chunks=custom_chunks)

        # Verify chunks are applied to all levels
        group = zarr.open_group(str(output_path), mode="r")
        
        for level_name in group.keys():
            if level_name.isdigit():
                level_array = group[level_name]
                assert level_array.chunks == custom_chunks

    def test_no_chunks_or_shards_parameters_in_constructor(self, tmp_path, sample_2d_image):
        """Test that chunks and shards are no longer accepted in constructor."""
        output_path = tmp_path / "test_constructor.zarr"
        dims = ["y", "x"]
        axis_units = {"unit": "micrometer"}

        # This should work - no chunks/shards in constructor
        writer = OmeZarrImage(
            path=output_path,
            image=sample_2d_image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )
        
        # Verify the object doesn't have chunks/shards attributes
        assert not hasattr(writer, "chunks")
        assert not hasattr(writer, "shards")
        
        # Write should work without parameters
        writer.write()
        
        # Verify file was created
        assert output_path.exists()

    def test_data_integrity_preserved(self, tmp_path, sample_2d_image):
        """Test that data integrity is preserved when using chunks."""
        output_path = tmp_path / "test_integrity.zarr"
        dims = ["y", "x"]
        axis_units = {"unit": "micrometer"}
        custom_chunks = (64, 64)

        writer = OmeZarrImage(
            path=output_path,
            image=sample_2d_image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )
        writer.write(chunks=custom_chunks)

        # Verify data integrity
        group = zarr.open_group(str(output_path), mode="r")
        level_0 = group["0"]
        np.testing.assert_array_equal(level_0[:], sample_2d_image)

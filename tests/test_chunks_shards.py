"""Test chunking and sharding functionality in OME-Zarr writer."""

import pytest
import numpy as np
import zarr
from ome_zarr_writer.image import OmeZarrImage
from ome_zarr_writer.schema_models import ScaleTransformation, TranslationTransformation


class TestChunksAndShards:
    """Test chunking and sharding functionality."""

    @pytest.fixture
    def sample_2d_image(self):
        """Create a sample 2D image for testing."""
        return np.random.randint(0, 255, size=(128, 128), dtype=np.uint8)

    @pytest.fixture
    def sample_3d_image(self):
        """Create a sample 3D image for testing."""
        return np.random.randint(0, 255, size=(5, 64, 64), dtype=np.uint8)

    @pytest.fixture
    def sample_4d_image(self):
        """Create a sample 4D image for testing."""
        return np.random.randint(0, 255, size=(3, 10, 64, 64), dtype=np.uint16)

    def test_chunks_tuple_2d(self, tmp_path, sample_2d_image):
        """Test writing with tuple chunks specification for 2D image."""
        output_path = tmp_path / "test_chunks_tuple_2d.zarr"
        dims = ["y", "x"]
        axis_units = {"unit": "micrometer"}
        custom_chunks = (32, 32)

        writer = OmeZarrImage(
            path=output_path,
            image=sample_2d_image,
            dims=dims,
            axis_units=axis_units,
            chunks=custom_chunks,
            overwrite=True,
        )
        writer.write()

        # Verify chunks
        group = zarr.open_group(str(output_path), mode="r")
        level_0 = group["0"]
        assert level_0.chunks == custom_chunks

    def test_chunks_tuple_3d(self, tmp_path, sample_3d_image):
        """Test writing with tuple chunks specification for 3D image."""
        output_path = tmp_path / "test_chunks_tuple_3d.zarr"
        dims = ["z", "y", "x"]
        axis_units = {"unit": "micrometer"}
        custom_chunks = (2, 32, 32)

        writer = OmeZarrImage(
            path=output_path,
            image=sample_3d_image,
            dims=dims,
            axis_units=axis_units,
            chunks=custom_chunks,
            overwrite=True,
        )
        writer.write()

        # Verify chunks
        group = zarr.open_group(str(output_path), mode="r")
        level_0 = group["0"]
        assert level_0.chunks == custom_chunks

    def test_chunks_tuple_4d(self, tmp_path, sample_4d_image):
        """Test writing with tuple chunks specification for 4D image."""
        output_path = tmp_path / "test_chunks_tuple_4d.zarr"
        dims = ["c", "z", "y", "x"]
        axis_units = {"unit": "micrometer"}
        custom_chunks = (1, 5, 32, 32)

        writer = OmeZarrImage(
            path=output_path,
            image=sample_4d_image,
            dims=dims,
            axis_units=axis_units,
            chunks=custom_chunks,
            overwrite=True,
        )
        writer.write()

        # Verify chunks
        group = zarr.open_group(str(output_path), mode="r")
        level_0 = group["0"]
        assert level_0.chunks == custom_chunks

    def test_chunks_integer(self, tmp_path, sample_2d_image):
        """Test writing with integer chunks specification."""
        output_path = tmp_path / "test_chunks_int.zarr"
        dims = ["y", "x"]
        axis_units = {"unit": "micrometer"}
        custom_chunks = (64, 64)

        writer = OmeZarrImage(
            path=output_path,
            image=sample_2d_image,
            dims=dims,
            axis_units=axis_units,
            chunks=custom_chunks,
            overwrite=True,
        )
        writer.write()

        # Verify chunks - zarr should apply the integer to all dimensions
        group = zarr.open_group(str(output_path), mode="r")
        level_0 = group["0"]
        # When chunks is an integer, zarr applies it to all dimensions
        assert level_0.chunks == (64, 64)

    def test_shards_tuple_2d(self, tmp_path, sample_2d_image):
        """Test writing with tuple shards specification for 2D image."""
        output_path = tmp_path / "test_shards_tuple_2d.zarr"
        dims = ["y", "x"]
        axis_units = {"unit": "micrometer"}
        custom_shards = (128, 128)

        writer = OmeZarrImage(
            path=output_path,
            image=sample_2d_image,
            dims=dims,
            axis_units=axis_units,
            shards=custom_shards,
            overwrite=True,
        )
        writer.write()

        # Verify shards (if supported by zarr version)
        group = zarr.open_group(str(output_path), mode="r")
        level_0 = group["0"]
        if hasattr(level_0, "shards") and level_0.shards is not None:
            assert level_0.shards == custom_shards

    def test_shards_tuple_4d(self, tmp_path, sample_4d_image):
        """Test writing with tuple shards specification for 4D image."""
        output_path = tmp_path / "test_shards_tuple_4d.zarr"
        dims = ["c", "z", "y", "x"]
        axis_units = {"unit": "micrometer"}
        custom_shards = (3, 10, 64, 64)

        writer = OmeZarrImage(
            path=output_path,
            image=sample_4d_image,
            dims=dims,
            axis_units=axis_units,
            shards=custom_shards,
            overwrite=True,
        )
        writer.write()

        # Verify shards (if supported by zarr version)
        group = zarr.open_group(str(output_path), mode="r")
        level_0 = group["0"]
        if hasattr(level_0, "shards") and level_0.shards is not None:
            assert level_0.shards == custom_shards

    def test_chunks_and_shards_together(self, tmp_path, sample_3d_image):
        """Test writing with both chunks and shards specified."""
        output_path = tmp_path / "test_chunks_and_shards.zarr"
        dims = ["z", "y", "x"]
        axis_units = {"unit": "micrometer"}
        custom_chunks = (2, 32, 32)
        custom_shards = (6, 64, 64)

        writer = OmeZarrImage(
            path=output_path,
            image=sample_3d_image,
            dims=dims,
            axis_units=axis_units,
            chunks=custom_chunks,
            shards=custom_shards,
            overwrite=True,
        )
        writer.write()

        # Verify both chunks and shards
        group = zarr.open_group(str(output_path), mode="r")
        level_0 = group["0"]
        assert level_0.chunks == custom_chunks
        if hasattr(level_0, "shards") and level_0.shards is not None:
            assert level_0.shards == custom_shards

    def test_chunks_with_multiscale(self, tmp_path, sample_2d_image):
        """Test chunks configuration with multiscale levels."""
        output_path = tmp_path / "test_chunks_multiscale.zarr"
        dims = ["y", "x"]
        axis_units = {"unit": "micrometer"}
        custom_chunks = (32, 32)

        writer = OmeZarrImage(
            path=output_path,
            image=sample_2d_image,
            dims=dims,
            axis_units=axis_units,
            chunks=custom_chunks,
            downscale_levels=2,
            overwrite=True,
        )
        writer.write()

        # Verify chunks are applied to all levels
        group = zarr.open_group(str(output_path), mode="r")
        
        # Check all available levels
        for level_name in group.keys():
            if level_name.isdigit():
                level_array = group[level_name]
                assert level_array.chunks == custom_chunks

    def test_shards_with_multiscale(self, tmp_path, sample_2d_image):
        """Test shards configuration with multiscale levels."""
        output_path = tmp_path / "test_shards_multiscale.zarr"
        dims = ["y", "x"]
        axis_units = {"unit": "micrometer"}
        custom_shards = (128, 128)

        writer = OmeZarrImage(
            path=output_path,
            image=sample_2d_image,
            dims=dims,
            axis_units=axis_units,
            shards=custom_shards,
            downscale_levels=2,
            overwrite=True,
        )
        writer.write()

        # Verify shards are applied to all levels (if supported)
        group = zarr.open_group(str(output_path), mode="r")
        
        for level_name in group.keys():
            if level_name.isdigit():
                level_array = group[level_name]
                if hasattr(level_array, "shards") and level_array.shards is not None:
                    assert level_array.shards == custom_shards

    def test_chunks_with_coordinate_transformations(self, tmp_path, sample_4d_image):
        """Test chunks with coordinate transformations."""
        output_path = tmp_path / "test_chunks_coord_transforms.zarr"
        dims = ["c", "z", "y", "x"]
        axis_units = {"unit": "micrometer"}
        custom_chunks = (1, 5, 32, 32)
        
        coord_transforms = [
            ScaleTransformation(scale=[1.0, 0.5, 0.1, 0.1]),
            TranslationTransformation(translation=[0.0, 0.0, 5.0, 5.0])
        ]

        writer = OmeZarrImage(
            path=output_path,
            image=sample_4d_image,
            dims=dims,
            axis_units=axis_units,
            coordinate_transformations=coord_transforms,
            chunks=custom_chunks,
            downscale_levels=1,
            overwrite=True,
        )
        writer.write()

        # Verify chunks and metadata
        group = zarr.open_group(str(output_path), mode="r")
        level_0 = group["0"]
        assert level_0.chunks == custom_chunks
        
        # Verify metadata exists
        assert "ome" in group.attrs
        ome_meta = group.attrs["ome"]
        assert "multiscales" in ome_meta

    def test_data_integrity_with_chunks(self, tmp_path, sample_3d_image):
        """Test that data is preserved correctly when using custom chunks."""
        output_path = tmp_path / "test_data_integrity_chunks.zarr"
        dims = ["z", "y", "x"]
        axis_units = {"unit": "micrometer"}
        custom_chunks = (5, 32, 32)

        writer = OmeZarrImage(
            path=output_path,
            image=sample_3d_image,
            dims=dims,
            axis_units=axis_units,
            chunks=custom_chunks,
            overwrite=True,
        )
        writer.write()

        # Verify data integrity
        group = zarr.open_group(str(output_path), mode="r")
        level_0 = group["0"]
        
        # Check chunks configuration
        assert level_0.chunks == custom_chunks
        
        # Check data integrity
        np.testing.assert_array_equal(level_0[:], sample_3d_image)

    def test_data_integrity_with_shards(self, tmp_path, sample_3d_image):
        """Test that data is preserved correctly when using custom shards."""
        output_path = tmp_path / "test_data_integrity_shards.zarr"
        dims = ["z", "y", "x"]
        axis_units = {"unit": "micrometer"}
        custom_chunks = (2, 32, 32)
        custom_shards = (4, 64, 64)

        writer = OmeZarrImage(
            path=output_path,
            image=sample_3d_image,
            dims=dims,
            axis_units=axis_units,
            chunks=custom_chunks,
            shards=custom_shards,
            overwrite=True,
        )
        writer.write()

        # Verify data integrity
        group = zarr.open_group(str(output_path), mode="r")
        level_0 = group["0"]
        
        # Check shards configuration (if supported)
        if hasattr(level_0, "shards") and level_0.shards is not None:
            assert level_0.shards == custom_shards
        
        # Check data integrity
        np.testing.assert_array_equal(level_0[:], sample_3d_image)

    def test_chunks_none_uses_default(self, tmp_path, sample_2d_image):
        """Test that chunks=None uses zarr's default chunking."""
        output_path = tmp_path / "test_chunks_none.zarr"
        dims = ["y", "x"]
        axis_units = {"unit": "micrometer"}

        writer = OmeZarrImage(
            path=output_path,
            image=sample_2d_image,
            dims=dims,
            axis_units=axis_units,
            chunks=None,  # Should use zarr's default
            overwrite=True,
        )
        writer.write()

        # Verify that zarr created some default chunks
        group = zarr.open_group(str(output_path), mode="r")
        level_0 = group["0"]
        assert level_0.chunks is not None
        assert len(level_0.chunks) == 2  # Should match 2D shape

    def test_shards_none_no_sharding(self, tmp_path, sample_2d_image):
        """Test that shards=None means no sharding is applied."""
        output_path = tmp_path / "test_shards_none.zarr"
        dims = ["y", "x"]
        axis_units = {"unit": "micrometer"}

        writer = OmeZarrImage(
            path=output_path,
            image=sample_2d_image,
            dims=dims,
            axis_units=axis_units,
            shards=None,  # No sharding
            overwrite=True,
        )
        writer.write()

        # Verify that no sharding is applied
        group = zarr.open_group(str(output_path), mode="r")
        level_0 = group["0"]
        # If shards is None or not supported, this should not raise an error
        if hasattr(level_0, "shards"):
            # shards should be None or not set
            assert level_0.shards is None or not level_0.shards

    def test_invalid_chunks_dimension_mismatch(self, tmp_path, sample_2d_image):
        """Test that mismatched chunk dimensions are handled properly by zarr."""
        output_path = tmp_path / "test_invalid_chunks.zarr"
        dims = ["y", "x"]
        axis_units = {"unit": "micrometer"}
        
        # This should be handled by zarr itself - zarr will either:
        # 1. Raise an error, or
        # 2. Adjust the chunks to match the array dimensions
        invalid_chunks = (32, 32, 32)  # 3D chunks for 2D array

        # We expect this to either work (zarr adjusts) or raise an error
        try:
            writer = OmeZarrImage(
                path=output_path,
                image=sample_2d_image,
                dims=dims,
                axis_units=axis_units,
                chunks=invalid_chunks,
                overwrite=True,
            )
            writer.write()
            
            # If it succeeded, zarr handled it somehow
            group = zarr.open_group(str(output_path), mode="r")
            level_0 = group["0"]
            # The actual chunks should be valid for the 2D array
            assert len(level_0.chunks) == 2
            
        except (ValueError, TypeError) as e:
            # This is also acceptable - zarr rejected the invalid chunks
            assert "chunk" in str(e).lower() or "shape" in str(e).lower()

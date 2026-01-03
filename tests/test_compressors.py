"""Tests for compressors parameter in write method."""

import numpy as np
import tempfile
from pathlib import Path
import zarr
from zarr.codecs import BloscCodec, GzipCodec, ZstdCodec

from ome_zarr_writer import OmeZarrImage


class TestCompressorsParameter:
    """Test class for testing the compressors parameter."""

    def test_write_with_single_compressor(self) -> None:
        """Test writing with a single compressor."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            image = np.random.randint(0, 255, size=(128, 128), dtype=np.uint8)
            dims = ["y", "x"]
            axis_units = {"y": "micrometer", "x": "micrometer"}
            output_path = Path(tmp_dir) / "single_compressor.zarr"

            writer = OmeZarrImage(
                path=output_path,
                image=image,
                dims=dims,
                axis_units=axis_units,
                overwrite=True,
            )

            # Test with BloscCodec
            writer.write(compressors=BloscCodec())

            # Verify file was created
            assert output_path.exists()

            # Verify zarr group structure
            group = zarr.open_group(str(output_path), mode="r")
            assert "0" in group
            array_0 = group["0"]
            assert array_0.shape == image.shape

    def test_write_with_multiple_compressors(self) -> None:
        """Test writing with multiple compressors."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            image = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
            dims = ["y", "x"]
            axis_units = {"y": "micrometer", "x": "micrometer"}
            output_path = Path(tmp_dir) / "multiple_compressors.zarr"

            writer = OmeZarrImage(
                path=output_path,
                image=image,
                dims=dims,
                axis_units=axis_units,
                overwrite=True,
            )

            # Test with list of compressors
            writer.write(compressors=[GzipCodec(), BloscCodec()])

            # Verify file was created
            assert output_path.exists()

            # Verify zarr group structure
            group = zarr.open_group(str(output_path), mode="r")
            assert "0" in group
            array_0 = group["0"]
            assert array_0.shape == image.shape

    def test_write_with_zstd_compressor(self) -> None:
        """Test writing with ZstdCodec compressor."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            image = np.random.randint(0, 255, size=(2, 32, 32), dtype=np.uint8)
            dims = ["c", "y", "x"]
            axis_units = {"y": "micrometer", "x": "micrometer"}
            output_path = Path(tmp_dir) / "zstd_compressor.zarr"

            writer = OmeZarrImage(
                path=output_path,
                image=image,
                dims=dims,
                axis_units=axis_units,
                overwrite=True,
            )

            # Test with ZstdCodec with custom level
            writer.write(compressors=ZstdCodec(level=5))

            # Verify file was created
            assert output_path.exists()

            # Verify zarr group structure
            group = zarr.open_group(str(output_path), mode="r")
            assert "0" in group
            array_0 = group["0"]
            assert array_0.shape == image.shape

    def test_write_without_compressors_still_works(self) -> None:
        """Test that writing without compressors parameter still works (backward compatibility)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            image = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
            dims = ["y", "x"]
            axis_units = {"y": "micrometer", "x": "micrometer"}
            output_path = Path(tmp_dir) / "no_compressor.zarr"

            writer = OmeZarrImage(
                path=output_path,
                image=image,
                dims=dims,
                axis_units=axis_units,
                overwrite=True,
            )

            # Test without compressors parameter (should use defaults)
            writer.write()

            # Verify file was created
            assert output_path.exists()

            # Verify zarr group structure
            group = zarr.open_group(str(output_path), mode="r")
            assert "0" in group
            array_0 = group["0"]
            assert array_0.shape == image.shape

    def test_write_compressors_with_multiscale(self) -> None:
        """Test compressors parameter with multiscale downscaling."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            image = np.random.randint(0, 255, size=(2, 128, 128), dtype=np.uint8)
            dims = ["c", "y", "x"]
            axis_units = {"y": "micrometer", "x": "micrometer"}
            output_path = Path(tmp_dir) / "multiscale_compressor.zarr"

            writer = OmeZarrImage(
                path=output_path,
                image=image,
                dims=dims,
                axis_units=axis_units,
                downscale_levels=2,
                downscale_method="nearest",
                overwrite=True,
            )

            # Test with compressor and multiscale
            writer.write(compressors=BloscCodec(cname="zstd"))

            # Verify file was created
            assert output_path.exists()

            # Verify zarr group structure with multiple levels
            group = zarr.open_group(str(output_path), mode="r")
            assert "0" in group  # Original level
            assert "1" in group  # First downscale level
            assert "2" in group  # Second downscale level

            # Verify shapes decrease as expected
            array_0 = group["0"]
            array_1 = group["1"]
            array_2 = group["2"]
            assert array_0.shape == image.shape
            assert array_1.shape[1:] == (64, 64)  # Half size in Y,X
            assert array_2.shape[1:] == (32, 32)  # Quarter size in Y,X

    def test_write_compressors_with_chunks_and_shards(self) -> None:
        """Test compressors parameter combined with chunks and shards."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            image = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
            dims = ["y", "x"]
            axis_units = {"y": "micrometer", "x": "micrometer"}
            output_path = Path(tmp_dir) / "compressor_chunks_shards.zarr"

            writer = OmeZarrImage(
                path=output_path,
                image=image,
                dims=dims,
                axis_units=axis_units,
                overwrite=True,
            )

            # Test with all parameters
            writer.write(
                compressors=GzipCodec(level=6), chunks=(32, 32), shards=(64, 64)
            )

            # Verify file was created
            assert output_path.exists()

            # Verify zarr group structure
            group = zarr.open_group(str(output_path), mode="r")
            assert "0" in group
            array_0 = group["0"]
            assert array_0.shape == image.shape

"""This module tests compressors parameter in the OME-Zarr writer."""

import numpy as np
import tempfile
from pathlib import Path
import zarr
from zarr.codecs import BloscCodec, GzipCodec, ZstdCodec

from ome_zarr_writer import OmeZarrImage


class TestCompressorsParameter:
    """ 
    This is the documentation for TestCompressorsParameter class, used for testing the 'compressors' parameter with the new write() method API.
    """

    def test_BloscCodec_compressor(self):
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

            writer.write(compressors=BloscCodec())

            assert output_path.exists()

            group = zarr.open_group(str(output_path), mode="r")
            assert "0" in group
            array_0 = group["0"]
            assert array_0.shape == image.shape

    def test_BloscCodec_and_GzipCodec_compressors(self):
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

            writer.write(compressors=[GzipCodec(), BloscCodec()])

            assert output_path.exists()

            group = zarr.open_group(str(output_path), mode="r")
            assert "0" in group
            array_0 = group["0"]
            assert array_0.shape == image.shape

    def test_zstd_compressor(self):
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

            writer.write(compressors=ZstdCodec(level=5))

            assert output_path.exists()

            group = zarr.open_group(str(output_path), mode="r")
            assert "0" in group
            array_0 = group["0"]
            assert array_0.shape == image.shape

    def test_without_defined_compressors(self):
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

            writer.write()

            assert output_path.exists()

            group = zarr.open_group(str(output_path), mode="r")
            assert "0" in group
            array_0 = group["0"]
            assert array_0.shape == image.shape

    def test_BloscCodec_compressor_with_multiscale(self):
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
                overwrite=True,
            )

            writer.write(compressors=BloscCodec(cname="zstd"))

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

    def test_GzipCodec_compressors_with_chunking_and_sharding(self):
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

            writer.write(
                compressors=GzipCodec(level=6), chunks=(32, 32), shards=(64, 64)
            )

            assert output_path.exists()

            group = zarr.open_group(str(output_path), mode="r")
            assert "0" in group
            array_0 = group["0"]
            assert array_0.shape == image.shape
            assert array_0.chunks == (32, 32)
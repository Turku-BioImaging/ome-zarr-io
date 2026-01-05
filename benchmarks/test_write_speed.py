"""
Benchmark writing to OME-Zarr using zarr-python vs. Rust zarrs backend.
"""

import shutil
from time import time
# from dask.distributed import Client
import dask
import dask.array as da
import numpy as np
import zarr
import zarr.codecs

from ome_zarr_writer import OmeZarrImage

dask.config.set(scheduler='threads')

if __name__ == '__main__':


    np.random.seed(88971)

    # Declare 3D array.
    # Benchmark writing using both zarrs (Rust) and zarr-python backends.
    # array = np.random.randint(0, 65535, size=(300, 2048, 2048), dtype=np.uint16)
    array = da.random.randint(0, 65535, size=(300, 2048, 2048), dtype=np.uint16, chunks=(10, 128, 128)).compute()
    dims = ["z", "y", "x"]
    axis_units = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}

    scale_transformations = {"z": 0.325, "y": 0.15, "x": 0.15}

    image = OmeZarrImage(
        path="benchmark.ome.zarr",
        image=array,
        dims=dims,
        axis_units=axis_units,
        scale_transformations=scale_transformations,
        downscale_levels=4,
        zarr_backend="zarrs",
        overwrite=True,
    )

    start_time = time()
    image.write(
        chunks=(10, 128, 128),
        shards=(30, 1024, 1024),
        compressors=[
            zarr.codecs.BloscCodec(
                cname="zstd", clevel=5, shuffle=zarr.codecs.BloscShuffle.bitshuffle
            )
        ],
    )

    end_time = time()
    print(f"Rust - Elapsed time: {end_time - start_time:.2f} seconds")


    # image = OmeZarrImage(
    #     path="benchmark.ome.zarr",
    #     image=array,
    #     dims=dims,
    #     axis_units=axis_units,
    #     scale_transformations=scale_transformations,
    #     downscale_levels=4,
    #     zarr_backend="zarr-python",
    #     overwrite=True,
    # )

    # start_time = time()
    # image.write(
    #     chunks=(10, 128, 128),
    #     shards=(30, 1024, 1024),
    #     compressors=[
    #         zarr.codecs.BloscCodec(
    #             cname="zstd", clevel=5, shuffle=zarr.codecs.BloscShuffle.bitshuffle
    #         )
    #     ],
    # )

    # end_time = time()
    # print(f"Python - Elapsed time: {end_time - start_time:.2f} seconds")

    # shutil.rmtree("benchmark.ome.zarr")

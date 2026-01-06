"""
Test script for writing stacked NumPy arrays.
Note the use of array indexing to write the dask array
without calling `.compute()` explicitly.
"""

import numpy as np
import zarr
import zarr.codecs
import tempfile
import dask.array as da
import os
import time

np.random.seed(1013)

# create temporary directory to store npy files
npy_dir = tempfile.TemporaryDirectory()

for i in range(10):
    arr = np.random.randint(0, 255, size=(20000, 20000), dtype=np.uint8)
    print(f"slice_{i:03d}.npy", f"{(arr.nbytes / (1024 * 1024)):.2f} MB")
    np.save(os.path.join(npy_dir.name, f"slice_{i:03d}.npy"), arr)

# Load the npy files as a stacked zarr array
npy_files = [os.path.join(npy_dir.name, f"slice_{i:03d}.npy") for i in range(10)]

lazy_arrays = [
    da.from_array(np.load(f, mmap_mode="r"), chunks=(600, 600)) for f in npy_files # type: ignore
]
combined_data = da.stack(lazy_arrays, axis=0).rechunk((5, 600, 600))

print(
    combined_data.shape,
    combined_data.chunksize,
    f"{(combined_data.nbytes / (1024 * 1024)):.2f} MB",
    f"Single chunk size: {(np.prod(combined_data.chunksize) * combined_data.dtype.itemsize / (1024 * 1024)):.2f} MB",
)

print("Writing stacked array to Zarr")
start_time = time.time()
arr = zarr.create_array(
    store="test.zarr",
    shape=combined_data.shape,
    chunks=combined_data.chunksize,
    shards=(10, 3000, 3000),
    dtype=combined_data.dtype,
    compressors=zarr.codecs.BloscCodec(
        cname="zstd", clevel=5, shuffle=zarr.codecs.BloscShuffle.bitshuffle
    ),
    overwrite=True,
)

arr[:] = combined_data # Store without calling .compute()

end_time = time.time()
print(f"Elapsed time: {end_time - start_time:.2f} seconds")

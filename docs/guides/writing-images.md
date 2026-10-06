# Writing an image

```python
import numpy as np
import zarr
from ome_zarr_io import Writer

# Create sample multichannel 3D confocal image data
image = np.random.randint(0, 255, size=(2, 32, 512, 512), dtype=np.uint8)

dims = ["c", "z", "y", "x"]

# Define axis units for each dimension explicitly
axis_units = {
    "z": "micrometer",
    "y": "micrometer",
    "x": "micrometer"
}

# Units should be one of the OME-Zarr 0.5 recommended names (e.g. "micrometer").
# "pixel" is also accepted for uncalibrated images, but is not in the spec's list
# and emits a UserWarning.

# Define scale transformations (pixel/voxel sizes)
# Units are determined by axis_units above
scale_transformations = {
    "z": 0.325,  # 0.325 μm z-step size
    "y": 0.15,   # 0.15 μm pixel size in Y
    "x": 0.15    # 0.15 μm pixel size in X
}

writer = Writer(
    path='example.ome.zarr',
    image=image,
    dims=dims,
    axis_units=axis_units,
    scale_transformations=scale_transformations,
    downscale_method='mean',  # Block average; use 'nearest' for label images
    downscale_levels=3,  # Create 3 additional downscale levels
    downscale_factor=2,  # Integer >= 2; each level is 2x smaller in Y/X
    overwrite=True
)

# Optional: configure chunking, sharding, and compression for large images
compressors = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle='bitshuffle')

writer.write(
    chunks=(1, 8, 256, 256),      # Optimize chunk size for access patterns
    shards=(2, 32, 512, 512),     # Group chunks into shards for efficiency
    compressors=compressors
)

```

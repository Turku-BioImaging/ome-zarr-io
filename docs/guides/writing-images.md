# Writing an image

```python
import numpy as np
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

writer.write()
```

The `downscale_levels`, `downscale_factor` and `downscale_method` options are explained under
[Resolution level](../getting-started.md#terminology) in the terminology section.

## Chunks, shards and compression

`write()` takes three optional arguments that control how the image data is stored. They apply to every
resolution level. Without them, zarr chooses the chunk shape and compresses with Zstd.

```python
writer = Writer(
    path="chunked.ome.zarr",
    image=image,
    dims=dims,
    axis_units=axis_units,
    scale_transformations=scale_transformations,
    overwrite=True,
)

writer.write(
    chunks=(1, 8, 256, 256),   # the unit that is read and decompressed on its own
    shards=(2, 32, 512, 512),  # groups chunks into fewer, larger files
    compressors={"name": "zstd", "configuration": {"level": 5}},
)
```

Each shard shape has to be a whole multiple of the chunk shape. Sharding keeps the number of files
manageable for large images.

### Choosing a compressor

`compressors` takes a codec as a plain dictionary or as an object from zarr. You do not need to import
anything from zarr to use the dictionary form. It follows the
[Zarr v3 codec names](https://zarr-specs.readthedocs.io/en/latest/v3/codecs/index.html), a `name` and an optional `configuration`:

| You want | Pass |
|---|---|
| zarr's default (Zstd, level 0) | nothing, or `None` |
| Zstd, stronger | `{"name": "zstd", "configuration": {"level": 5}}` |
| Blosc with Zstd and bit shuffling | `{"name": "blosc", "configuration": {"cname": "zstd", "clevel": 5, "shuffle": "bitshuffle"}}` |
| Gzip | `{"name": "gzip", "configuration": {"level": 6}}` |
| Several codecs in a row | a list of such dictionaries |
| No compression | `[]` |

The same options as objects, if you already work with zarr:

```python
import zarr

writer = Writer(
    path="blosc.ome.zarr",
    image=image,
    dims=dims,
    axis_units=axis_units,
    scale_transformations=scale_transformations,
    overwrite=True,
)

writer.write(
    compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle"),
)
```

`add_labels` accepts the same `chunks`, `shards` and `compressors` arguments for label images.

# Reading an image

Use the `Reader` class to validate an image and retrieve channel or label data as NumPy/Dask arrays. The examples
below read this small image, which has two channels and one label image:

```python
import numpy as np
from ome_zarr_io import Writer

writer = Writer(
    path="example.ome.zarr",
    image=np.random.randint(0, 255, size=(2, 8, 128, 128), dtype=np.uint8),
    dims=["c", "z", "y", "x"],
    axis_units={"z": "micrometer", "y": "micrometer", "x": "micrometer"},
    scale_transformations={"z": 0.325, "y": 0.15, "x": 0.15},
    channels=["DAPI", "GFP"],
    downscale_levels=2,
    overwrite=True,
)
writer.write()
writer.add_labels("cell_space_segmentation", np.ones((1, 8, 128, 128), dtype=np.uint8))
```

```python
from ome_zarr_io import Reader

reader = Reader("example.ome.zarr")

# Validate against the OME-Zarr 0.5 spec
reader.validate()  # raises jsonschema.exceptions.ValidationError if invalid

# Inspect metadata
print(reader.dims)            # e.g. ["c", "z", "y", "x"]
print(reader.channel_names)   # e.g. ["DAPI", "GFP"]
print(reader.label_names)     # e.g. ["cell_space_segmentation"]

# Get channel data as a lazy Dask array (default) or eager NumPy array
dapi = reader.get_channel("DAPI")                       # dask.array.Array
dapi_np = reader.get_channel("DAPI", as_type="numpy")    # numpy.ndarray

# Read a lower-resolution pyramid level
dapi_level1 = reader.get_channel("DAPI", level=1)

# Get a specific label by name
segmentation = reader.get_label("cell_space_segmentation", as_type="numpy")

# Query physical pixel/voxel size
print(reader.get_physical_size())  # {"z": PhysicalSize(0.325, "micrometer"), ...}
print(reader.get_voxel_size())     # {"z": 0.325, "y": 0.15, "x": 0.15}

# Full physical transform of one pyramid level (an int index)
lvl = reader.get_level_transform(1)  # metadata only, no image data is read
print(lvl.physical_voxel_size)     # spatial axes only: {"y": PhysicalSize(0.3, "micrometer"), ...}
print(lvl.physical_pixel_size)     # same thing, handy for 2D data
print(lvl.common_spatial_unit)     # "micrometer" (None if absent or mixed)
print(lvl.scale, lvl.translation)  # tuples aligned with lvl.axes
print(lvl.downsample_factor)       # scale relative to level 0, per axis
```

`reader.level_transforms` returns these objects for every level. `scale` and `translation`
already include any multiscale-level `coordinateTransformations`.

## Resampling between pyramid levels

Read data at one level, process it, and bring the result onto the grid of another
level with `Reader.resample`:

```python
# Process a coarse level (cheap), then upscale the result to full resolution
from scipy.ndimage import gaussian_filter

smooth = gaussian_filter(reader.get_channel("DAPI", level=2, as_type="numpy"), sigma=2)
smooth_full = reader.resample(smooth, from_level=2, to_level=0)

# Labels always use nearest neighbour, so no new label ids are created
labels_coarse = reader.resample(reader.get_label("cell_space_segmentation"), 0, 2, kind="label")
```

The factors come from the levels' scale transforms and the output has exactly the
stored shape of `to_level`. Images are downscaled by block mean and upscaled
linearly by default; pass `method="nearest"` to change that. Dask input stays lazy,
and `as_type` selects a Dask or NumPy result. Levels refer to the image's pyramid.

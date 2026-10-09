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
```

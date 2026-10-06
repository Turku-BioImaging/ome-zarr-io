# ome-zarr-io

Read, write, and validate OME-Zarr 0.5 images and high-content-screening plates: from a NumPy array to a spec-compliant OME-Zarr image in a few lines of Python.

Microscopy datasets are growing faster than desktop tools can handle. [OME-Zarr](https://ngff.openmicroscopy.org/)
is the community format for storing them in a chunked, cloud-friendly way that viewers such as napari and Vizarr
can open. Getting the metadata exactly right is fiddly. `ome-zarr-io` handles it for you, so you can spend your
time on the analysis instead of the file format.

`ome-zarr-io` writes NumPy or Dask arrays as OME-Zarr 0.5 images with axes, units, pixel sizes, multiscale pyramids, channels and segmentation labels, and arranges such images into high-content-screening plates. It also reads images back and validates images and plates against the OME-Zarr 0.5 schemas, from Python or the command line.

## Quick start

```python
import numpy as np
from ome_zarr_io import Writer, Reader, validate

image = np.random.randint(0, 255, size=(2, 16, 256, 256), dtype=np.uint8)

Writer(
    path="example.ome.zarr", image=image, dims=["c", "z", "y", "x"],
    axis_units={"z": "micrometer", "y": "micrometer", "x": "micrometer"},
    scale_transformations={"z": 0.5, "y": 0.2, "x": 0.2},
    channels={"DAPI": {"color": "0000FF"}, "GFP": {"color": "00FF00"}},
    downscale_levels=2, overwrite=True,
).write()

print(validate("example.ome.zarr"))                      # spec check
print(Reader("example.ome.zarr").get_voxel_size())       # physical pixel size
```

```{toctree}
:caption: Getting started
:maxdepth: 1

getting-started
```

```{toctree}
:caption: Guides
:maxdepth: 1

guides/reading
guides/writing-images
guides/channels
guides/labels
guides/plates
guides/validating
guides/cli
```

```{toctree}
:caption: Reference
:maxdepth: 1

api/index
```

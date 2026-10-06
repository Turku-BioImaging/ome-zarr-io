# ome-zarr-io

Read, write, and validate OME-Zarr 0.5 images and high-content-screening plates: from a NumPy array to a spec-compliant OME-Zarr image in a few lines of Python.

[![CI/CD](https://github.com/Turku-BioImaging/ome-zarr-io/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Turku-BioImaging/ome-zarr-io/actions/workflows/ci-cd.yml)
[![Documentation](https://img.shields.io/badge/docs-online-00aeef)](https://turku-bioimaging.github.io/ome-zarr-io/)
[![Python 3.11 | 3.12 | 3.13 | 3.14](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/ome-zarr-io)](https://pypi.org/project/ome-zarr-io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Microscopy datasets are growing faster than desktop tools can handle. [OME-Zarr](https://ngff.openmicroscopy.org/)
is the community format for storing them in a chunked, cloud-friendly way that viewers such as napari and Vizarr
can open. Getting the metadata exactly right is fiddly. `ome-zarr-io` handles it for you, so you can spend your
time on the analysis instead of the file format.

`ome-zarr-io` writes NumPy or Dask arrays as OME-Zarr 0.5 images with axes, units, pixel sizes, multiscale pyramids, channels and segmentation labels, and arranges such images into high-content-screening plates. It also reads images back and validates images and plates against the OME-Zarr 0.5 schemas, from Python or the command line.

## Installation

```bash
pip install ome-zarr-io
```

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

The same validation is available on the command line:

```bash
ome-zarr-io validate example.ome.zarr
```

## Documentation

The full documentation is at **<https://turku-bioimaging.github.io/ome-zarr-io/>**:

- [Getting started](https://turku-bioimaging.github.io/ome-zarr-io/getting-started.html): installation, including
  remote (URL) support, and the terminology used throughout.
- Guides for [reading](https://turku-bioimaging.github.io/ome-zarr-io/guides/reading.html) and
  [writing](https://turku-bioimaging.github.io/ome-zarr-io/guides/writing-images.html) images,
  [channels](https://turku-bioimaging.github.io/ome-zarr-io/guides/channels.html),
  [labels](https://turku-bioimaging.github.io/ome-zarr-io/guides/labels.html),
  [high-content-screening plates](https://turku-bioimaging.github.io/ome-zarr-io/guides/plates.html),
  [validation](https://turku-bioimaging.github.io/ome-zarr-io/guides/validating.html) and the
  [command line](https://turku-bioimaging.github.io/ome-zarr-io/guides/cli.html).
- The [API reference](https://turku-bioimaging.github.io/ome-zarr-io/api/index.html).

![Stack](https://go-skill-icons.vercel.app/api/icons?i=py,numpy,dask,pytest,githubactions&theme=dark)

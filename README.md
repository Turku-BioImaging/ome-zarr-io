# ome-zarr-io

Read, write, and validate OME-Zarr 0.5 multiscale images: from a NumPy array to a spec-compliant dataset in a few lines of Python.

[![CI/CD](https://github.com/Turku-BioImaging/ome-zarr-writer/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Turku-BioImaging/ome-zarr-writer/actions/workflows/ci-cd.yml)
[![Python 3.11 | 3.12 | 3.13 | 3.14](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/ome-zarr-io)](https://pypi.org/project/ome-zarr-io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Microscopy datasets are growing faster than desktop tools can handle. [OME-Zarr](https://ngff.openmicroscopy.org/)
is the community format for storing them in a chunked, cloud-friendly way that viewers such as napari and Vizarr
can open. Getting the metadata exactly right is fiddly. `ome-zarr-io` handles it for you, so you can spend your
time on the analysis instead of the file format.

`ome-zarr-io` writes NumPy or Dask arrays as OME-Zarr 0.5 datasets with axes, units, pixel sizes, multiscale pyramids, channels and segmentation labels. It also reads them back and validates them against the OME-Zarr 0.5 schemas, from Python or the command line.

**Quick start**

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

![Stack](https://go-skill-icons.vercel.app/api/icons?i=py,numpy,dask,pytest,githubactions&theme=dark)


## Usage

### Writing an OME-Zarr 0.5 fileset

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
    # Note: 'c' (channel) dimension doesn't need a unit
}

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
    downscale_method='gaussian',  # Use Gaussian filtering for intensity images
    downscale_levels=3,  # Create 3 additional downscale levels
    downscale_factor=2,
    overwrite=True
)

# Optional: configure chunking, sharding, and compression for large datasets
compressors = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle='bitshuffle')

writer.write(
    chunks=(1, 8, 256, 256),      # Optimize chunk size for access patterns
    shards=(2, 32, 512, 512),     # Group chunks into shards for efficiency
    compressors=compressors
)

```

#### Channels and display windows

```python
writer = Writer(
    path="example.ome.zarr",
    image=image,
    dims=dims,
    axis_units=axis_units,
    channels={"DAPI": {"color": "0000FF", "window": (0, 200)}, "GFP": {}},
    colors="random",  # distinct colors for channels without one; color_seed=... changes the palette
)
```

`channels` accepts a dict keyed by label, a list of labels, or a list of dicts, and requires a `c` axis with one
entry per channel. Per-channel keys:

- `color`: hex without "#", or `"random"`
- `window`: `"auto"` (default), `"minmax"`, `(start, end)`, a `Window`, or `None`
- `family` and `active`

Each window's `min`/`max` are the data's min/max; with `"auto"`, `start`/`end` follow Fiji's auto-contrast
(`"minmax"` uses the full range). `colors="random"` gives every channel without a color a distinct one
(`color_seed=...` changes the palette); `ome_zarr_io.random_colors(n, seed=0)` exposes the same generator.

`Omero(...)` objects, `omero_metadata=`, and the top-level `Channel`/`Window`/`Omero` exports are deprecated or
removed from `ome_zarr_io`. Passing `omero_metadata=` (or an `Omero` in `channels=`) emits a `DeprecationWarning`;
use `channels=` instead. `Channel`, `Window` and `Omero` remain importable from `ome_zarr_io.schema_models`.

### Adding labels to an existing OME-Zarr 0.5 fileset

```python
import numpy as np
from ome_zarr_io import Writer

image = np.random.randint(0, 255, size=(512, 512), dtype=np.uint8)
label_mask = np.zeros_like(image, dtype=np.uint8)
label_mask[100:200, 100:200] = 1

writer = Writer(
    path="example_labels.ome.zarr",
    image=image,
    dims=["y", "x"],
    axis_units={"y": "micrometer", "x": "micrometer"},
    downscale_levels=2,
    overwrite=True,
)
writer.write()

writer.add_labels(
    name="cell_space_segmentation",
    array=label_mask,
    colors=[
        {"label-value": 0, "rgba": [0, 0, 128, 128]},
        {"label-value": 1, "rgba": [0, 128, 0, 128]},
    ],
    properties=[
        {"label-value": 0, "class": "intercellular space"},
        {"label-value": 1, "class": "cell"},
    ],
)
```

This creates a nested `labels/cell_space_segmentation` group under the image and writes NGFF 0.5 label metadata in the parent `ome.labels` list and the label group's `ome.image-label` block. See [OME-Zarr 0.5 spec](https://ngff.openmicroscopy.org/specifications/0.5/index.html#labels-metadata) for more details.

### Reading an OME-Zarr fileset

Use the `Reader` class to validate a fileset and retrieve channel or label data as NumPy/Dask arrays.

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

### Validating an OME-Zarr fileset

`validate()` checks a fileset against the OME-Zarr 0.5 schemas and returns a `FilesetReport`. It never
raises on malformed metadata; problems are collected in `report.errors`.

```python
from ome_zarr_io import validate

report = validate("example.ome.zarr")  # local path or URL

if report:  # same as report.is_valid
    print(report.spec_version, [a["name"] for a in report.axes])
    print([c.label for c in report.channels], [lb.name for lb in report.labels])
else:
    for issue in report.errors:
        print(issue.location, issue.path, issue.message)

print(report)               # human-readable summary
report.to_dict()            # JSON-serializable
report.raise_if_invalid()   # raises jsonschema.exceptions.ValidationError
```

Use `strict=True` for the stricter schemas.

### Command line

The same validation is available as a command (also `python -m ome_zarr_io ...`):

```bash
ome-zarr-io validate example.ome.zarr                  # human-readable summary
ome-zarr-io validate example.ome.zarr --strict --quiet && echo ok
```

Exit status is `0` if valid, `1` if invalid, and `2` on a usage error or if the target cannot be read.
Remote URLs need `pip install "ome-zarr-io[remote]"`.

## Downscaling Methods

- `"gaussian"` (default): Gaussian blur before downscaling. Use for intensity images.
- `"nearest"`: nearest-neighbor. Preserves discrete values; use for labels and segmentation masks.

## Installation

```bash
pip install ome-zarr-io
```

To validate remote (URL) filesets, install the optional extra:

```bash
pip install "ome-zarr-io[remote]"
```

### From source

To get the latest development version:

```bash
git clone https://github.com/Turku-BioImaging/ome-zarr-io.git
cd ome-zarr-io
pip install -e .
```

## Development

### Setting up development environment

```bash
git clone https://github.com/Turku-BioImaging/ome-zarr-io.git
cd ome-zarr-io
./setup-dev.sh
```

This will:
- Create a virtual environment
- Install the package in development mode
- Install development dependencies
- Set up pre-commit hooks

### Running tests

```bash
# Basic tests
pytest

# With coverage
./run_tests.sh cov

# All tests + examples
./run_tests.sh all
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This package builds upon the excellent work of the [zarr-python](https://github.com/zarr-developers/zarr-python) community.

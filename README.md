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

`ome-zarr-io` writes NumPy or Dask arrays as OME-Zarr 0.5 datasets with axes, units, pixel sizes, multiscale pyramids, channels, segmentation labels and high-content-screening plates. It also reads them back and validates them against the OME-Zarr 0.5 schemas, from Python or the command line.

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
    channels={"DAPI": {"color": "0000FF", "window": (0, 200)}, "GFP": {}},  # GFP gets an automatic color
    color_seed=0,  # change to get a different automatic palette
)
```

`channels` accepts a dict keyed by label, a list of labels, or a list of dicts, and requires a `c` axis with one
entry per channel. Per-channel keys:

- `color`: hex without "#". If omitted (or `"random"`), the channel gets a distinct automatic color, because the
  spec requires every channel to have one
- `window`: `"auto"` (default), `"minmax"`, `(start, end)`, a `Window`, or `None`. The spec requires every channel
  to have a window, so `None` is written as the full data range, like `"minmax"`
- `family` and `active`

Each window's `min`/`max` are the data's min/max; with `"auto"`, `start`/`end` follow Fiji's auto-contrast
(`"minmax"` uses the full range). Automatic colors avoid the hues of the colors you set, and are reproducible:
the same `color_seed` gives the same colors (`color_seed=...` changes the palette).
`ome_zarr_io.random_colors(n, seed=0)` exposes the same generator. `colors="random"` is still accepted but no
longer needed.

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

### Writing a high-content-screening (HCS) plate

`PlateWriter` writes a plate as the `plate/<row>/<column>/<field>` layout defined by the OME-Zarr 0.5 spec. Each field of
view is an ordinary multiscale image written with `Writer`, so it accepts the same options.

```python
import numpy as np
from ome_zarr_io import PlateWriter

units = {"y": "micrometer", "x": "micrometer"}

with PlateWriter(
    "screen.ome.zarr",
    rows=["A", "B"],
    columns=["1", "2", "3"],
    name="Screen 1",
    acquisitions=[{"id": 0, "name": "t0"}],  # optional
) as plate:
    for row, column in [("A", "1"), ("A", "2"), ("B", "3")]:
        for _ in range(2):  # two fields of view per well
            image = np.random.randint(0, 255, size=(2, 256, 256), dtype=np.uint8)
            plate.add_field(
                row, column, image, dims=["c", "y", "x"], axis_units=units,
                channels=["DAPI", "GFP"], downscale_levels=2, acquisition=0,
            )
```

- Rows and columns are declared up front. Their order sets each well's `rowIndex` and `columnIndex`.
- `add_field` takes the same keyword arguments as `Writer` (except `overwrite`, which belongs to the plate). It creates
  the row and well groups as needed. `field=` sets the field index; by default the next free one is used.
- If you declare more than one acquisition, every `add_field` call must say which one it belongs to (`acquisition=`),
  as the spec requires.
- The well and plate metadata are rewritten after every field, so an interrupted run leaves a valid partial plate. The
  plate metadata first appears with the first field, because the spec requires at least one well.
- By default every field must match the first one in dims, axes and channels. Pass `check_consistency=False` to allow
  mixed fields.
- `add_field` returns the field's `Writer`, so you can attach labels with `add_labels`.
- Leaving the `with` block validates the plate and all wells against the schemas. Without `with`, call `plate.close()`.
- `overwrite=True` replaces an existing plate.

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

A plate is validated all the way down: its own metadata, every well it lists, and every field of view and label image
in those wells. `issue.location` says where each problem is, for example `plate`, `A/1` (a well), `A/1/0` (a field) or
`A/1/0/labels/cells`. Besides the schemas, it checks that each well's `path`, `rowIndex` and `columnIndex` agree, that
listed wells and fields exist, that field acquisitions are defined by the plate, and that the OME-Zarr version is the
same throughout. Validating a well on its own checks its fields too.

### Command line

The same validation is available as a command (also `python -m ome_zarr_io ...`):

```bash
ome-zarr-io validate example.ome.zarr                  # human-readable summary
ome-zarr-io validate example.ome.zarr --strict --quiet && echo ok
```

Exit status is `0` if valid, `1` if invalid, and `2` on a usage error or if the target cannot be read.
Remote URLs need `pip install "ome-zarr-io[remote]"`.

## Downscaling Methods

Level `L` of the pyramid is `downscale_factor ** L` times smaller than the original in Y and X
(`downscale_factor` must be an integer >= 2; default `2`). Pixels that don't fill a whole block at
the bottom/right edge are dropped at coarser levels.

- `"mean"` (default): averages each block of pixels (a box filter followed by subsampling). Use for intensity images.
- `"nearest"`: takes one pixel per block. Preserves discrete values; use for labels and segmentation masks.
- `"gaussian"`: deprecated alias for `"mean"`.

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


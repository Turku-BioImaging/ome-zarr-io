# ome-zarr-io

Write valid OME-Zarr 0.5 multiscale images

[![CI/CD](https://github.com/Turku-BioImaging/ome-zarr-writer/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Turku-BioImaging/ome-zarr-writer/actions/workflows/ci-cd.yml)
[![Python 3.11 | 3.12 | 3.13 | 3.14](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/)

A Python package for reading and writing OME-Zarr 0.5 multiscale images using NumPy and Dask. This library provides a simple interface for creating cloud-optimized bioimaging data in the OME-Zarr 0.5 format.

![Stack](https://go-skill-icons.vercel.app/api/icons?i=py,numpy,dask&theme=dark)

## Installation

### From source

```bash
git clone https://github.com/Turku-BioImaging/ome-zarr-writer.git
cd ome-zarr-writer
pip install -e .
```

### From pip requirements.txt

```bash
# Add to your requirements.txt file
git+https://github.com/Turku-BioImaging/ome-zarr-writer.git

# Then install with pip
pip install -r requirements.txt
```

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

## Downscaling Methods

### Gaussian Filtering (Default)
- **Best for:** Intensity images (fluorescence, brightfield, etc.)
- **Method:** Applies Gaussian blur before downscaling to prevent aliasing artifacts
- **Use case:** Most microscopy images where preserving smooth intensity variations is important

### Nearest-Neighbor Interpolation
- **Best for:** Label/segmentation images with discrete values
- **Method:** Preserves exact pixel values during downscaling
- **Use case:** Segmentation masks, label images where each value represents a distinct object/region


## Development

### Setting up development environment

```bash
git clone https://github.com/Turku-BioImaging/ome-zarr-writer.git
cd ome-zarr-writer
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

### Code formatting and linting

```bash
# Format and lint
black src/ tests/ examples/
flake8 src/ tests/
mypy src/
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This package builds upon the excellent work of the [zarr-python](https://github.com/zarr-developers/zarr-python) community.

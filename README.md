# ome-zarr-writer

Write valid OME-Zarr 0.5 multiscale images

[![CI/CD](https://github.com/Turku-BioImaging/ome-zarr-writer/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Turku-BioImaging/ome-zarr-writer/actions/workflows/ci-cd.yml)
[![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)

A Python package for writing valid OME-Zarr 0.5 multiscale images. This library provides a simple interface for creating cloud-optimized bioimaging data in the OME-Zarr 0.5 format.

## Features

- Write OME-Zarr 0.5 compliant multiscale images
- Integration with Zarr version 3.
- Type-safe Python dataclasses implementing the OME-Zarr schema
- JSON Schema validation for metadata compliance
- Strict TCZYX dimension ordering
- Space and time axis axis unit validation
- Two downscaling methods: Gaussian filtering (default) and nearest-neighbor interpolation


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

## Downscaling Methods

### Gaussian Filtering (Default)
- **Best for:** Intensity images (fluorescence, brightfield, etc.)
- **Method:** Applies Gaussian blur before downscaling to prevent aliasing artifacts
- **Use case:** Most microscopy images where preserving smooth intensity variations is important

### Nearest-Neighbor Interpolation
- **Best for:** Label/segmentation images with discrete values
- **Method:** Preserves exact pixel values during downscaling
- **Use case:** Segmentation masks, label images where each value represents a distinct object/region

## Quick Start

### Basic Image Writing

```python
import numpy as np
from ome_zarr_writer import OmeZarrImage

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

ome_zarr_image = OmeZarrImage(
    path='example.ome.zarr',
    image=image,
    dims=dims,
    axis_units=axis_units,
    scale_transformations=scale_transformations,
    downscale_method='gaussian',  # Use Gaussian filtering for intensity images
    downscale_levels=3,  # Create 3 additional downscale levels
    overwrite=True
)

# Write the OME-Zarr file
ome_zarr_image.write()

```

### Advanced Storage Configuration

```python
import numpy as np
from ome_zarr_writer import OmeZarrImage

# Create sample large 3D dataset
image = np.random.randint(0, 65535, size=(64, 1024, 1024), dtype=np.uint16)

dims = ["z", "y", "x"]

axis_units = {
    "z": "micrometer",
    "y": "micrometer",
    "x": "micrometer"
}

scale_transformations = {
    "z": 0.2,   # 0.2 μm z-step
    "y": 0.1,   # 0.1 μm pixel size
    "x": 0.1    # 0.1 μm pixel size
}

# Configure chunking, sharding, and compression
ome_zarr_image = OmeZarrImage(
    path='advanced.ome.zarr',
    image=image,
    dims=dims,
    axis_units=axis_units,
    scale_transformations=scale_transformations,
    downscale_method='gaussian',  # Default Gaussian filtering for intensity data
    downscale_levels=3,
    downscale_factor=2,
    overwrite=True
)

# Setup compression options
compressors = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle=zarr.codecs.BloscShuffle.shuffle)

ome_zarr_image.write(
    chunks=(8, 256, 256),      # Optimize chunk size for access patterns
    shards=(32, 512, 512),     # Group chunks into shards for efficiency
    compressors=compressors
)

print("✅ Successfully created advanced.ome.zarr with optimized storage")
```


### Valid Dimension Combinations

All input images must follow the **TCZYX** order where T (time), C (channel), and Z are optional:

| Dimensions | Order | Example Use Case | Input Shape Example |
|------------|-------|------------------|-------------------|
| **YX** | Y, X | 2D grayscale image | `(512, 512)` |
| **ZYX** | Z, Y, X | 3D confocal stack | `(20, 256, 256)` |
| **CYX** | C, Y, X | RGB/multichannel 2D | `(3, 512, 512)` |
| **CZYX** | C, Z, Y, X | Multiscale 3D stack | `(2, 15, 256, 256)` |
| **TYX** | T, Y, X | Time-lapse 2D | `(100, 256, 256)` |
| **TZYX** | T, Z, Y, X | 4D live imaging | `(50, 10, 128, 128)` |
| **TCYX** | T, C, Y, X | Time-lapse multichannel | `(50, 2, 128, 128)` |
| **TCZYX** | T, C, Z, Y, X | 5D live cell imaging | `(20, 3, 8, 64, 64)` |


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

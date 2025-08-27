# ome-zarr-writer

Write valid OME-Zarr 0.5 multiscale images

[![CI/CD](https://github.com/Turku-BioImaging/ome-zarr-writer/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Turku-BioImaging/ome-zarr-writer/actions/workflows/ci-cd.yml)
[![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)

<p float='left'>
    <a href='https://bioimaging.fi' target='_blank'>
        <img src='https://github.com/user-attachments/assets/429d4dac-7af6-40f3-abe6-b4ba999185b0' style="height:75px;width:auto;"/>
    </a>
</p>

A Python package for writing valid OME-Zarr 0.5 multiscale images. This library provides a simple interface for creating cloud-optimized bioimaging data in the OME-Zarr 0.5 format.
  

## Features

- Write OME-Zarr 0.5 compliant multiscale images
- Integration with Zarr version 3.
- Type-safe Python dataclasses implementing the OME-Zarr schema
- JSON Schema validation for metadata compliance
- Strict TCZYX dimension ordering
- Space and time axis unit validation (26 supported units: angstrom, micrometer, meter, etc.)
- Two downscaling methods: Gaussian filtering (default) and nearest-neighbor interpolation

## Quick Start

### Basic Usage

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
# Setup compression options
compressors = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle=zarr.codecs.BloscShuffle.shuffle)

ome_zarr_image.write(
    chunks=(8, 256, 256),      # Optimize chunk size for access patterns
    shards=(32, 512, 512),     # Group chunks into shards for efficiency
    compressors=compressors
)
```

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
The OME-Zarr multiscales specification requires generating downscaled (lower resolution) images from the original images. This allows OME-Zarr compatible software to read only the required resolution depending on the current zoom level.

To generate these downscaled images, Gaussian filtering (default) is applied before downscaling in order to avoid scaling artifacts, especially in high-frequency images.

### Gaussian Filtering (Default)
- **Best for:** Intensity images (fluorescence, brightfield, etc.)
- **Method:** Applies Gaussian blur before downscaling to prevent aliasing artifacts
- **Use case:** Most microscopy images where preserving smooth intensity variations is important

### Nearest-Neighbor Interpolation  
- **Best for:** Label/segmentation images with discrete values
- **Method:** Preserves exact pixel values during downscaling
- **Use case:** Segmentation masks, label images where each value represents a distinct object/region

```python
ome_zarr_image = OmeZarrImage(
    path='example.ome.zarr',
    image=image,
    dims=dims,
    axis_units=axis_units,
    scale_transformations=scale_transformations,
    downscale_method='gaussian',  # or `nearest`
    downscale_levels=3,  # Create 3 additional downscale levels
    overwrite=True
)
```

## GPU Support (Optional)

For GPU-accelerated downscaling, install with CuPy:

```bash
# Install with GPU support
pip install -e ".[gpu]"

# Or install CuPy for your CUDA version separately
pip install cupy-cuda11x
pip install cupy-cuda12x
```

**Requirements:** NVIDIA CUDA-compatible hardware and drivers. If CuPy is not available, the library automatically falls back to CPU computation.

**Performance:** GPU acceleration provides significant speedups for large images and complex downscaling operations:
- **Large images (≥4096×4096)**: Up to 2.5× faster with multiple downscale levels
- **Small images (<1024×1024)**: CPU may be faster due to GPU overhead
- **Optimal for**: Multi-level downscaling, large multi-dimensional datasets

#### Device Selection

The library automatically chooses the optimal device, but you can also specify manually:

```python
from ome_zarr_writer import OmeZarrImage
from ome_zarr_writer.downscaler import Downscaler

# Check available GPU devices
devices = Downscaler.list_cuda_devices()
for device in devices:
    print(f"GPU {device['device_id']}: {device['device_name']} "
          f"({device['total_memory_gb']} GB)")

# Use specific GPU device
ome_zarr_image = OmeZarrImage(
    path='multi_gpu.ome.zarr',
    image=image,
    dims=dims,
    device='cuda',
    cuda_device_id=0,  # Use first GPU
    overwrite=True
)
```


## Examples

The `examples/` directory contains **4 comprehensive examples** designed for progressive learning:

- **`examples/getting_started.py`** - **Start here!** Essential usage patterns
  - Simple 2D and 3D image writing
  - Time-lapse data handling
  - Basic compression and chunking

- **`examples/metadata_examples.py`** - Advanced metadata capabilities
  - Metadata configuration
  - Multichannel setups
  - Validation examples


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
# Basic tests (excludes GPU tests)
pytest

# With coverage (excludes GPU tests)
./run_tests.sh cov

# GPU tests only (requires CUDA/CuPy)
./run_tests.sh gpu

# All tests including GPU tests
./run_tests.sh all-gpu

# All tests + examples (excludes GPU)
./run_tests.sh all
```

**Note:** GPU tests are automatically skipped when CUDA/CuPy is not available. In CI/CD environments without GPU hardware, use `-m "not gpu"` to exclude GPU tests.

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

This package was largely inspired by [ngff-zarr](https://github.com/thewtex/ngff-zarr) and builds upon excellent work of the [zarr-python](https://github.com/zarr-developers/zarr-python) community.

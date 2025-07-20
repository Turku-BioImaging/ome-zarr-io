# ome-zarr-writer

Write valid OME-Zarr 0.5 multiscale images

[![CI/CD](https://github.com/Turku-BioImaging/ome-zarr-writer/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Turku-BioImaging/ome-zarr-writer/actions/workflows/ci-cd.yml)
[![PyPI version](https://badge.fury.io/py/ome-zarr-writer.svg)](https://badge.fury.io/py/ome-zarr-writer)
[![Python versions](https://img.shields.io/pypi/pyversions/ome-zarr-writer.svg)](https://pypi.org/project/ome-zarr-writer/)

A Python package for writing valid OME-Zarr 0.5 multiscale images. This library provides a simple interface for creating cloud-optimized bioimaging data in the OME-Zarr format.

## Features

- Write OME-Zarr 0.5 compliant multiscale images
- Support for various image formats and data types
- Efficient handling of large image datasets
- Integration with the zarr ecosystem

## Installation

### From PyPI (recommended)

```bash
pip install ome-zarr-writer
```

### From source

```bash
git clone https://github.com/Turku-BioImaging/ome-zarr-writer.git
cd ome-zarr-writer
pip install -e .
```

## Quick Start

```python
import numpy as np
from ome_zarr_writer import OMEZarrWriter

# Create sample image data
image = np.random.randint(0, 255, size=(512, 512, 3), dtype=np.uint8)

# Initialize writer
writer = OMEZarrWriter("output.zarr")

# Write image with pixel size information
writer.write_image(
    image=image,
    pixel_size=(0.5, 0.5),  # micrometers per pixel
    units=["micrometer", "micrometer"],
    channel_names=["Red", "Green", "Blue"]
)
```

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
pytest tests/
```

### Code formatting and linting

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/ome_zarr_writer/
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This package builds upon the excellent work of the [ome-zarr-py](https://github.com/ome/ome-zarr-py) and [zarr-python](https://github.com/zarr-developers/zarr-python) communities.

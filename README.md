# ome-zarr-writer

Write valid OME-Zarr 0.5 multiscale images

[![CI/CD](https://github.com/Turku-BioImaging/ome-zarr-writer/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Turku-BioImaging/ome-zarr-writer/actions/workflows/ci-cd.yml)
[![PyPI version](https://badge.fury.io/py/ome-zarr-writer.svg)](https://badge.fury.io/py/ome-zarr-writer)
[![Python versions](https://img.shields.io/pypi/pyversions/ome-zarr-writer.svg)](https://pypi.org/project/ome-zarr-writer/)

A Python package for writing valid OME-Zarr 0.5 multiscale images. This library provides a simple interface for creating cloud-optimized bioimaging data in the OME-Zarr format.

## Features

- Write OME-Zarr 0.5 compliant multiscale images
- Type-safe Python dataclasses implementing the OME-Zarr schema
- JSON Schema validation for metadata compliance
- Support for various image formats and data types
- **Multichannel image support** (CYX, CZYX, TYX, TCYX dimensions)
- Efficient handling of large image datasets
- Integration with the zarr ecosystem
- OMERO display settings support

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

### Basic Image Writing

```python
import numpy as np
from ome_zarr_writer import OMEZarrImage

# Create sample image data
image = np.random.randint(0, 255, size=(512, 512), dtype=np.uint8)

# Initialize writer
writer = OMEZarrImage(
    path="output.zarr",
    image=image,
    dims=["y", "x"]
)

# Write image with pixel size information
writer.write(
    image=image,
    pixel_size=(0.5, 0.5),  # micrometers per pixel
    units=["micrometer", "micrometer"]
)
```

### Using Schema Dataclasses

```python
from ome_zarr_writer.schema_models import (
    OMEZarrImageMetadata,
    OMEMetadata,
    Multiscale,
    Dataset,
    ScaleTransformation,
    create_2d_axes,
    create_cyx_axes  # For multichannel images
)

# Create metadata using type-safe dataclasses
axes = create_2d_axes(0.1, 0.1, unit="micrometer")
scale_transform = ScaleTransformation(scale=[0.1, 0.1])
dataset = Dataset(path="0", coordinateTransformations=[scale_transform])
multiscale = Multiscale(datasets=[dataset], axes=axes, name="My Image")
ome_metadata = OMEMetadata(multiscales=[multiscale], version="0.5")
metadata = OMEZarrImageMetadata(ome=ome_metadata)

# Convert to dictionary for zarr attrs
attrs = metadata.to_dict()
```

### Multichannel Images

```python
from ome_zarr_writer.schema_models import (
    create_cyx_axes,    # Channel, Y, X
    create_czyx_axes,   # Channel, Z, Y, X
    create_tyx_axes,    # Time, Y, X
    create_tcyx_axes,   # Time, Channel, Y, X
    create_scale_transformation
)

# Create CYX (multichannel) metadata
axes = create_cyx_axes(0.1, 0.1, unit="micrometer")
scale_transform = create_scale_transformation([1.0, 0.1, 0.1])  # channel, y, x

# For fluorescence Z-stack
axes = create_czyx_axes(0.1, 0.1, 0.3, unit="micrometer")  # x, y, z pixel sizes
scale_transform = create_scale_transformation([1.0, 0.3, 0.1, 0.1])  # c, z, y, x
```

### Validation

```python
from ome_zarr_writer import OMEZarrValidator

# Validate metadata against OME-Zarr schema
validator = OMEZarrValidator()
is_valid = validator.validate_image_metadata(attrs)

# Get detailed validation errors
errors = validator.get_validation_errors(attrs, "image")
```

## Examples

See the `examples/` directory for comprehensive usage examples:

- `examples/schema_example.py` - Complete schema dataclass examples
- `examples/cyx_example.py` - Multichannel image examples (CYX, CZYX, TYX, TCYX)
- `examples/unit_validation_example.py` - Space axis unit validation demo
- `examples/integration_example.py` - Integration with existing code
- `examples/basic_example.py` - Basic writing operations
- `examples/validation_example.py` - Validation workflows

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

Multiple ways to run tests:

```bash
# Using pytest directly
pytest

# Using the test script
./run_tests.sh

# With coverage
./run_tests.sh cov

# Fast tests only
./run_tests.sh fast

# All tests + examples
./run_tests.sh all

# Using make (if available)
make test
make test-cov
```

### Code formatting and linting

```bash
# Format code
black src/ tests/ examples/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Using make
make format
make lint
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This package builds upon the excellent work of the [ome-zarr-py](https://github.com/ome/ome-zarr-py) and [zarr-python](https://github.com/zarr-developers/zarr-python) communities.

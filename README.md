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
- **Strict TCZYX dimension ordering** - Input images always follow Time-Channel-Z-Y-X order (T, C, Z optional)
- **Multichannel image support** (CYX, CZYX, TYX, TCYX, TCZYX dimensions)
- **Space axis unit validation** (26 supported units: angstrom, micrometer, meter, etc.)
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

### Using Schema Dataclasses with TCZYX Ordering

```python
from ome_zarr_writer.schema_models import (
    OMEZarrImageMetadata,
    OMEMetadata,
    Multiscale,
    Dataset,
    ScaleTransformation,
    create_yx_axes,   # 2D spatial
    create_cyx_axes,  # Multichannel 2D
    create_tczyx_axes, # Full 5D (Time-Channel-Z-Y-X)
    validate_tczyx_axis_ordering
)

# All axis creation functions follow strict TCZYX ordering
# Valid combinations: YX, ZYX, CYX, CZYX, TYX, TZYX, TCYX, TCZYX

# Create metadata using type-safe dataclasses
axes = create_yx_axes(0.1, 0.1, unit="micrometer")
scale_transform = ScaleTransformation(scale=[0.1, 0.1])
dataset = Dataset(path="0", coordinateTransformations=[scale_transform])
multiscale = Multiscale(datasets=[dataset], axes=axes, name="My Image")
ome_metadata = OMEMetadata(multiscales=[multiscale], version="0.5")
metadata = OMEZarrImageMetadata(ome=ome_metadata)

# Validate dimension ordering (automatically done in Multiscale.__post_init__)
validate_tczyx_axis_ordering(axes)

# Convert to dictionary for zarr attrs
attrs = metadata.to_dict()
```

### TCZYX Dimension Examples

```python
from ome_zarr_writer.schema_models import (
    create_axes,       # NEW: Single unified function!
    create_yx_axes,    # YX: 2D spatial
    create_zyx_axes,   # ZYX: 3D spatial  
    create_cyx_axes,   # CYX: Multichannel 2D
    create_czyx_axes,  # CZYX: Multichannel 3D
    create_tyx_axes,   # TYX: Time-series 2D
    create_tzyx_axes,  # TZYX: Time-series 3D
    create_tcyx_axes,  # TCYX: Time-series multichannel 2D
    create_tczyx_axes, # TCZYX: Time-series multichannel 3D (maximum)
    create_scale_transformation
)

# NEW UNIFIED API - Single function for all dimension combinations:
axes = create_axes("czyx", 0.1, 0.1, 0.3, unit="micrometer")  # Multichannel 3D
axes = create_axes("tcyx", 0.2, 0.2, unit="micrometer")       # Time-series multichannel 2D
axes = create_axes("tczyx", 0.25, 0.25, 0.5, unit="micrometer") # Full 5D

# Example: Create CZYX (multichannel 3D) metadata
# Input image shape: (3, 20, 256, 256) = (Channel, Z, Y, X)
axes = create_czyx_axes(0.1, 0.1, 0.3, unit="micrometer")  # x, y, z pixel sizes
scale_transform = create_scale_transformation([1.0, 0.3, 0.1, 0.1])  # c, z, y, x
```

## TCZYX Dimension Ordering

This library enforces **strict TCZYX dimension ordering** for input images:

### Valid Dimension Combinations

All input images must follow the **TCZYX** order where T (time), C (channel), and Z are optional:

| Dimensions | Order | Example Use Case | Input Shape Example |
|------------|-------|------------------|-------------------|
| **YX** | Y, X | 2D grayscale image | `(512, 512)` |
| **ZYX** | Z, Y, X | 3D confocal stack | `(20, 256, 256)` |
| **CYX** | C, Y, X | RGB/multichannel 2D | `(3, 512, 512)` |
| **CZYX** | C, Z, Y, X | Multichannel 3D stack | `(2, 15, 256, 256)` |
| **TYX** | T, Y, X | Time-lapse 2D | `(100, 256, 256)` |
| **TZYX** | T, Z, Y, X | 4D live imaging | `(50, 10, 128, 128)` |
| **TCYX** | T, C, Y, X | Time-lapse multichannel | `(50, 2, 128, 128)` |
| **TCZYX** | T, C, Z, Y, X | 5D live cell imaging | `(20, 3, 8, 64, 64)` |

### Invalid Orderings (Rejected)

```python
# ❌ These will raise ValueError:
create_xy_axes(...)      # X, Y - wrong order  
create_zct_axes(...)     # Z, C, T - wrong order
create_czdt_axes(...)    # Invalid 'D' dimension
```

### Space Axis Units

Space axes (X, Y, Z) support 26 validated units:

```python
valid_units = [
    "angstrom", "attometer", "centimeter", "decimeter", "exameter", 
    "femtometer", "foot", "gigameter", "hectometer", "inch", "kilometer", 
    "megameter", "meter", "micrometer", "millimeter", "nanometer", 
    "parsec", "petameter", "picometer", "terameter", "yard", "yoctometer", 
    "yottameter", "zeptometer", "zettameter", "reference_frame"
]
```

### Validation

```python
from ome_zarr_writer.schema_models import validate_tczyx_axis_ordering, create_axes

# Automatic validation in unified function
axes = create_axes("tczyx", 0.1, 0.1, 0.3, unit="micrometer")  # ✓ Valid

# Manual validation
validate_tczyx_axis_ordering(axes)  # Raises ValueError if invalid
```

### Complete Example

```python
# Run the comprehensive examples:
python examples/unified_create_axes_example.py  # NEW: Unified API demonstration
python examples/tczyx_ordering_example.py       # Complete TCZYX examples
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

- `examples/unified_create_axes_example.py` - **NEW**: Single function API demonstration  
- `examples/tczyx_ordering_example.py` - Complete TCZYX dimension ordering demonstration
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

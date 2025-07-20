# OME-Zarr Schema Dataclasses

This document describes the Python dataclasses that implement the OME-Zarr 0.5 `image.schema` specification.

## Overview

The `schema_models.py` module provides type-safe Python dataclasses that correspond to the JSON schema definitions in the OME-Zarr 0.5 specification. These dataclasses:

- Provide type hints for better IDE support and development experience
- Include validation to ensure schema compliance
- Can convert to/from dictionary representations for zarr metadata
- Support both simple and complex OME-Zarr configurations

## Main Classes

### `OMEZarrImageMetadata`
The top-level container for OME-Zarr image metadata.

```python
from ome_zarr_writer.schema_models import OMEZarrImageMetadata

# Use to_dict() to get zarr attrs
attrs = metadata.to_dict()

# Use from_dict() to parse existing metadata
metadata = OMEZarrImageMetadata.from_dict(attrs_dict)
```

### `OMEMetadata`
Contains the core OME metadata including multiscales, version, and optional OMERO settings.

### `Multiscale`
Defines a multiscale image with datasets, axes, and transformations.

### `Dataset`
Represents a single resolution level with path and coordinate transformations.

### `Axis`
Defines image dimensions (spatial, temporal, channel, etc.).

### Coordinate Transformations
- `ScaleTransformation`: Defines pixel/voxel size scaling
- `TranslationTransformation`: Defines spatial offset

### OMERO Display Settings
- `Omero`: Container for display metadata
- `Channel`: Individual channel display settings
- `Window`: Intensity display window settings

## Quick Start Examples

### Simple 2D Image

```python
from ome_zarr_writer.schema_models import *

# Create axes
axes = create_2d_axes(0.1, 0.1, unit="micrometer")

# Create coordinate transformation
scale_transform = create_scale_transformation([0.1, 0.1])

# Create dataset
dataset = Dataset(path="0", coordinateTransformations=[scale_transform])

# Create multiscale
multiscale = Multiscale(datasets=[dataset], axes=axes, name="My Image")

# Create OME metadata
ome_metadata = OMEMetadata(multiscales=[multiscale], version="0.5")

# Create final metadata
metadata = OMEZarrImageMetadata(ome=ome_metadata)

# Get dictionary for zarr
attrs = metadata.to_dict()
```

### Multiscale 3D Image

```python
# Create 3D axes
axes = create_3d_axes(0.1, 0.1, 0.2, unit="micrometer")

# Create multiple resolution levels
datasets = []
for level in range(4):
    scale_factor = 2 ** level
    scale_transform = ScaleTransformation(scale=[
        0.2 * scale_factor,  # Z
        0.1 * scale_factor,  # Y
        0.1 * scale_factor   # X
    ])
    dataset = Dataset(path=str(level), coordinateTransformations=[scale_transform])
    datasets.append(dataset)

multiscale = Multiscale(datasets=datasets, axes=axes)
# ... rest as above
```

### Multichannel with OMERO Settings

```python
# Create multichannel axes
axes = [
    Axis(name="c", type="channel"),
    Axis(name="y", type="space", unit="micrometer"),
    Axis(name="x", type="space", unit="micrometer")
]

# Create OMERO channel settings
channels = [
    Channel(
        label="DAPI",
        color="0000FF", 
        window=Window(start=0, min=0, end=4095, max=4095),
        active=True
    ),
    Channel(
        label="GFP",
        color="00FF00",
        window=Window(start=100, min=0, end=3000, max=4095),
        active=True
    )
]

omero = Omero(channels=channels)

# Include OMERO in OME metadata
ome_metadata = OMEMetadata(
    multiscales=[multiscale],
    version="0.5", 
    omero=omero
)
```

### With Translation

```python
# Create transformations with both scale and translation
transformations = [
    ScaleTransformation(scale=[0.1, 0.1]),
    TranslationTransformation(translation=[100.0, 50.0])  # Offset in μm
]

dataset = Dataset(path="0", coordinateTransformations=transformations)
```

## Validation

The dataclasses include comprehensive validation:

- **Multiscale**: Must have at least one dataset, 2-5 axes, 2-3 space axes
- **Dataset**: Must have at least one coordinate transformation, at least one scale transformation
- **Axes**: Names must be unique, appropriate types for space/time/channel
- **Transformations**: Scale and translation arrays must have ≥2 elements
- **OMERO**: Window values must be properly ordered (start ≤ min ≤ end ≤ max)

Validation errors are raised as `ValueError` with descriptive messages.

## Utility Functions

```python
# Convenience functions for common configurations
axes_2d = create_2d_axes(pixel_size_x, pixel_size_y, unit="micrometer")
axes_3d = create_3d_axes(pixel_size_x, pixel_size_y, pixel_size_z, unit="micrometer")
scale_transform = create_scale_transformation([scale_z, scale_y, scale_x])
```

## Integration with Zarr

```python
import zarr

# Create metadata using dataclasses
metadata = OMEZarrImageMetadata(ome=ome_metadata)

# Write to zarr group
group = zarr.open("/path/to/image.zarr", mode="w")
group.attrs.update(metadata.to_dict())

# Read from zarr group
loaded_metadata = OMEZarrImageMetadata.from_dict(dict(group.attrs))
```

## Examples

See `examples/schema_example.py` for complete working examples and `tests/test_schema_models.py` for comprehensive test cases demonstrating all features.

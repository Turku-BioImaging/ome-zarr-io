# Coordinate Transformations for Multiscale Images

The OME-Zarr writer now supports automatic calculation of coordinate transformations for multiscale pyramid levels.

## How it works

When you provide coordinate transformations for the original image, the library automatically calculates the appropriate transformations for each downscale level:

- **Scale transformations**: Spatial dimensions (Y and X) are multiplied by the downscale factor for each level
- **Translation transformations**: Remain unchanged across all levels

## Example

```python
from ome_zarr_writer import OmeZarrImage
from ome_zarr_writer.schema_models import ScaleTransformation, TranslationTransformation
import numpy as np

# Create sample image
image = np.random.randint(0, 255, size=(2, 5, 256, 256), dtype=np.uint8)  # CZYX
dims = ["c", "z", "y", "x"]

# Define coordinate transformations for the original image
coordinate_transformations = [
    # Pixel sizes: c=1.0, z=0.25μm, y=0.1μm, x=0.1μm
    ScaleTransformation(scale=[1.0, 0.25, 0.1, 0.1]),
    
    # Physical offset: 5μm in Y and X
    TranslationTransformation(translation=[0.0, 0.0, 5.0, 5.0])
]

# Create OME-Zarr with multiscale levels
ome_zarr_image = OmeZarrImage(
    path="example.zarr",
    image=image,
    dims=dims,
    coordinate_transformations=coordinate_transformations,
    downscale_levels=3,
    downscale_factor=2,
    overwrite=True
)

# Get coordinate transformations for each level
level_transformations = ome_zarr_image._create_coordinate_transformations_for_levels()
```

## Result

The above code will automatically generate coordinate transformations for 4 levels:

- **Level 0** (1x): Scale [1.0, 0.25, 0.1, 0.1], Translation [0.0, 0.0, 5.0, 5.0]
- **Level 1** (2x): Scale [1.0, 0.25, 0.2, 0.2], Translation [0.0, 0.0, 5.0, 5.0]
- **Level 2** (4x): Scale [1.0, 0.25, 0.4, 0.4], Translation [0.0, 0.0, 5.0, 5.0]
- **Level 3** (8x): Scale [1.0, 0.25, 0.8, 0.8], Translation [0.0, 0.0, 5.0, 5.0]

Note how only the Y and X components of the scale transformation change, while the Z and C components remain constant, and the translation remains completely unchanged.

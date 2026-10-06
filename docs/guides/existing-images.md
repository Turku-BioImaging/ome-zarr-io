# Adding labels to an existing image

This continues [Adding labels](labels.md).

`Writer.from_existing` attaches a `Writer` to an OME-Zarr image you did not just write, so you can add labels to it
without re-supplying its dimensions, pixel sizes or pyramid settings:

```python
import numpy as np
from ome_zarr_io import Writer

writer = Writer.from_existing("example.ome.zarr")  # or a plate field: "screen.ome.zarr/A/1/0"
writer.add_labels("cell_space_segmentation", label_mask)
```

Everything `add_labels` needs is read from the image's metadata (axes, shape, pixel sizes, number of levels and
downscale factor); no pixel data is read, and the image is not modified. `write()` is disabled on the returned
`Writer`, so the image cannot be overwritten by accident.

The image's pyramid must be one this library could have written: only Y and X downscaled, by a constant integer
factor, with no `translation` or multiscale-level `coordinateTransformations`. Otherwise `from_existing` raises a
`ValueError` explaining what differs, because labels written for it would not line up with the image. Pass a field
of a plate (`.../A/1/0`), not the plate itself.

# Adding labels

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

A label always gets the same number of resolution levels as the image on disk, as the spec requires. The
`downscale_levels` argument of `add_labels` is deprecated.

Label arrays must have an integer data type (`uint8`, `int8`, `uint16`, `int16`, `uint32`, `int32`, `uint64` or
`int64`). Float and boolean arrays are rejected; convert a boolean mask with `mask.astype("uint8")`. `validate()`
reports a label with another data type, or with a different number of levels than its image, as invalid.

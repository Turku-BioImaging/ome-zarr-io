# High-content-screening plates

`PlateWriter` writes a plate as the `plate/<row>/<column>/<field>` layout defined by the OME-Zarr 0.5 spec. Each field of
view is an ordinary multiscale image written with `Writer`, so it accepts the same options.

```python
import numpy as np
from ome_zarr_io import PlateWriter

units = {"y": "micrometer", "x": "micrometer"}

with PlateWriter(
    "screen.ome.zarr",
    rows=["A", "B"],
    columns=["1", "2", "3"],
    name="Screen 1",
    acquisitions=[{"id": 0, "name": "t0"}],  # optional
) as plate:
    for row, column in [("A", "1"), ("A", "2"), ("B", "3")]:
        for _ in range(2):  # two fields of view per well
            image = np.random.randint(0, 255, size=(2, 256, 256), dtype=np.uint8)
            plate.add_field(
                row, column, image, dims=["c", "y", "x"], axis_units=units,
                channels=["DAPI", "GFP"], downscale_levels=2, acquisition=0,
            )
```

- Rows and columns are declared up front. Their order sets each well's `rowIndex` and `columnIndex`.
- `add_field` takes the same keyword arguments as `Writer` (except `overwrite`, which belongs to the plate). It creates
  the row and well groups as needed. `field=` sets the field index; by default the next free one is used.
- If you declare more than one acquisition, every `add_field` call must say which one it belongs to (`acquisition=`),
  as the spec requires.
- The well and plate metadata are rewritten after every field, so an interrupted run leaves a valid partial plate. The
  plate metadata first appears with the first field, because the spec requires at least one well.
- By default every field must match the first one in dims, axes and channels. Pass `check_consistency=False` to allow
  mixed fields.
- `add_field` returns the field's `Writer`, so you can attach labels with `add_labels`.
- Leaving the `with` block validates the plate and all wells against the schemas. Without `with`, call `plate.close()`.
- `overwrite=True` replaces an existing plate.

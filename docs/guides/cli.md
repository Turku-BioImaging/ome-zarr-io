# Command line

The same validation is available as a command (also `python -m ome_zarr_io ...`). To try it, first write a small image:

```python
import numpy as np
from ome_zarr_io import Writer

Writer(
    path="example.ome.zarr",
    image=np.zeros((2, 64, 64), dtype=np.uint8),
    dims=["c", "y", "x"],
    axis_units={"y": "micrometer", "x": "micrometer"},
    channels=["DAPI", "GFP"],
    overwrite=True,
).write()
```

Then validate it:

```bash
ome-zarr-io validate example.ome.zarr                  # human-readable summary
ome-zarr-io validate example.ome.zarr --strict --quiet && echo ok
```

Exit status is `0` if valid, `1` if invalid, and `2` on a usage error or if the target cannot be read.
Remote URLs need `pip install "ome-zarr-io[remote]"`.

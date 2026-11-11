# Channels and display windows

```python
import numpy as np
from ome_zarr_io import Writer

image = np.random.randint(0, 255, size=(2, 8, 256, 256), dtype=np.uint8)

writer = Writer(
    path="example.ome.zarr",
    image=image,
    dims=["c", "z", "y", "x"],
    axis_units={"z": "micrometer", "y": "micrometer", "x": "micrometer"},
    channels={"DAPI": {"color": "0000FF", "window": (0, 200)}, "GFP": {}},  # GFP gets an automatic color
    color_seed=0,  # change to get a different automatic palette
    overwrite=True,
)
writer.write()
```

`channels` accepts a dict keyed by label, a list of labels, or a list of dicts, and requires a `c` axis with one
entry per channel. Per-channel keys:

- `color`: hex without "#". If omitted (or `"random"`), the channel gets a distinct automatic color, because the
  spec requires every channel to have one
- `window`: `"auto"` (default), `"minmax"`, `(start, end)`, a `Window`, or `None`. The spec requires every channel
  to have a window, so `None` is written as the full data range, like `"minmax"`
- `family` and `active`

Each window's `min`/`max` are the data's min/max; with `"auto"`, `start`/`end` follow Fiji's auto-contrast
(`"minmax"` uses the full range). Automatic colors avoid the hues of the colors you set, and are reproducible:
the same `color_seed` gives the same colors (`color_seed=...` changes the palette).
`ome_zarr_io.random_colors(n, seed=0)` exposes the same generator. `colors="random"` is still accepted but no
longer needed.

`Omero(...)` objects, `omero_metadata=`, and the top-level `Channel`/`Window`/`Omero` exports are deprecated or
removed from `ome_zarr_io`. Passing `omero_metadata=` (or an `Omero` in `channels=`) emits a `DeprecationWarning`;
use `channels=` instead. `Channel`, `Window` and `Omero` remain importable from `ome_zarr_io.schema_models`.

## Adding channels to an existing image

To store a derived version of the data next to the original, for example a background-subtracted copy, append it
as new channels. The new data is level 0 only; its pyramid is built with the image's own downscale factor and
number of levels.

```python
import numpy as np
from ome_zarr_io import Writer

raw = np.random.randint(100, 4000, size=(1, 256, 256), dtype=np.uint16)
Writer(
    "add_channels_example.ome.zarr",
    raw,
    ["c", "y", "x"],
    {"y": "micrometer", "x": "micrometer"},
    downscale_levels=2,
    channels=["DAPI"],
    overwrite=True,
).write()

background_subtracted = np.clip(raw.astype(np.int32) - 100, 0, None).astype(np.uint16)
Writer.from_existing("add_channels_example.ome.zarr").add_channels(
    background_subtracted,
    [{"label": "DAPI_bgsub", "active": False}],  # hidden by default in viewers
)
```

The array must match the image in every axis but `c` and must have the same dtype, because all channels of a zarr
array share one dtype. Labels must be new. Colors that you leave out are chosen to differ from the existing ones.
The existing channels are not rewritten, and if writing fails the `c` axes are shrunk back. The image needs a `c`
axis and `omero` channel metadata with one entry per existing channel.

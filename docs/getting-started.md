# Getting started

## Installation

```bash
pip install ome-zarr-io
```

To validate remote (URL) filesets, install the optional extra:

```bash
pip install "ome-zarr-io[remote]"
```

### From source

To get the latest development version:

```bash
git clone https://github.com/Turku-BioImaging/ome-zarr-io.git
cd ome-zarr-io
pip install -e .
```

## Terminology

This project follows the [OME-Zarr 0.5 specification](https://ngff.openmicroscopy.org/0.5/):

- **Image**: a Zarr group with `multiscales` metadata. It holds a pyramid of resolution levels (one array per
  level) and may also carry channel metadata and labels. `Writer` writes images and `Reader` reads them.
- **Image data**: the NumPy or Dask array you pass in (the `image=` argument), as opposed to the image on disk.
- **Resolution level**: one array of an image's pyramid. Level `L` is `downscale_factor ** L` times smaller than the
  original in Y and X (`downscale_factor` must be an integer >= 2; default `2`). Pixels that don't fill a whole block
  at the bottom/right edge are dropped at coarser levels. `Writer` builds the levels with `downscale_levels`,
  `downscale_factor` and `downscale_method`:
  - `"mean"` (default): averages each block of pixels (a box filter followed by subsampling). Use for intensity
    images.
  - `"nearest"`: takes one pixel per block. Preserves discrete values; use for labels and segmentation masks.
  - `"gaussian"`: deprecated alias for `"mean"`.
- **Label image**: an image stored under an image's `labels/` group.
- **Plate, well, field**: the high-content-screening layout `plate/<row>/<column>/<field>`. A field (of view) is an
  image.
- **Fileset**: the whole hierarchy of Zarr groups under one path. It can be an image, a plate, a well or a
  bioformats2raw collection. `validate()` and the command line accept any fileset; `Reader` and
  `Writer.from_existing` need an image, so for a plate pass one of its fields.

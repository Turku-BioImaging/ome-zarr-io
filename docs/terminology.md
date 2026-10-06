# Terminology

This project follows the [OME-Zarr 0.5 specification](https://ngff.openmicroscopy.org/0.5/):

- **Image**: a Zarr group with `multiscales` metadata. It holds a pyramid of resolution levels (one array per
  level) and may also carry channel metadata and labels. `Writer` writes images and `Reader` reads them.
- **Image data**: the NumPy or Dask array you pass in (the `image=` argument), as opposed to the image on disk.
- **Label image**: an image stored under an image's `labels/` group.
- **Plate, well, field**: the high-content-screening layout `plate/<row>/<column>/<field>`. A field (of view) is an
  image.
- **Fileset**: the whole hierarchy of Zarr groups under one path. It can be an image, a plate, a well or a
  bioformats2raw collection. `validate()` and the command line accept any fileset; `Reader` and
  `Writer.from_existing` need an image, so for a plate pass one of its fields.

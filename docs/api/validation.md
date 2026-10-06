# Validation

{py:func}`~ome_zarr_io.validate` checks any fileset (an image, a plate, a well or a bioformats2raw collection) and
returns a {py:class}`~ome_zarr_io.FilesetReport`. The same check is available on the
[command line](../guides/cli.md).

```{eval-rst}
.. autofunction:: ome_zarr_io.validate
```

```{eval-rst}
.. autoclass:: ome_zarr_io.FilesetReport
   :members:
```

```{eval-rst}
.. autoclass:: ome_zarr_io.ValidationIssue
   :members:
   :undoc-members:
```

## Report details

These classes describe parts of a {py:class}`~ome_zarr_io.FilesetReport`.

```{eval-rst}
.. autoclass:: ome_zarr_io.report.LevelInfo
   :members:
   :undoc-members:
```

```{eval-rst}
.. autoclass:: ome_zarr_io.report.ChannelInfo
   :members:
   :undoc-members:
```

```{eval-rst}
.. autoclass:: ome_zarr_io.report.LabelInfo
   :members:
   :undoc-members:
```

## Schema validator

The lower-level validator that {py:func}`~ome_zarr_io.validate` is built on.

```{eval-rst}
.. autoclass:: ome_zarr_io.OMEZarrValidator
   :members:
```

# Models and helpers

## Axes

```{eval-rst}
.. autoclass:: ome_zarr_io.Axis
   :members:
   :undoc-members:
```

## Channel display metadata

{py:class}`~ome_zarr_io.schema_models.Omero`, {py:class}`~ome_zarr_io.schema_models.Channel` and
{py:class}`~ome_zarr_io.schema_models.Window` are legacy ways to describe channels. They are deprecated in favour
of the `channels=` argument of {py:class}`~ome_zarr_io.Writer`, and are importable from `ome_zarr_io.schema_models`
only.

```{eval-rst}
.. autoclass:: ome_zarr_io.schema_models.Omero
   :members:
   :undoc-members:
```

```{eval-rst}
.. autoclass:: ome_zarr_io.schema_models.Channel
   :members:
   :undoc-members:
```

```{eval-rst}
.. autoclass:: ome_zarr_io.schema_models.Window
   :members:
   :undoc-members:
```

## Helpers

```{eval-rst}
.. autofunction:: ome_zarr_io.random_colors
```

# Command line

The same validation is available as a command (also `python -m ome_zarr_io ...`):

```bash
ome-zarr-io validate example.ome.zarr                  # human-readable summary
ome-zarr-io validate example.ome.zarr --strict --quiet && echo ok
```

Exit status is `0` if valid, `1` if invalid, and `2` on a usage error or if the target cannot be read.
Remote URLs need `pip install "ome-zarr-io[remote]"`.

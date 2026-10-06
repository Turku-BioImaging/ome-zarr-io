# Validating a fileset

`validate()` checks a fileset (an image, a plate, a well or a bioformats2raw collection) against the OME-Zarr 0.5
schemas and returns a `FilesetReport`. It never
raises on malformed metadata; problems are collected in `report.errors`.

```python
from ome_zarr_io import validate

report = validate("example.ome.zarr")  # local path or URL

if report:  # same as report.is_valid
    print(report.spec_version, [a["name"] for a in report.axes])
    print([c.label for c in report.channels], [lb.name for lb in report.labels])
else:
    for issue in report.errors:
        print(issue.location, issue.path, issue.message)

print(report)               # human-readable summary
report.to_dict()            # JSON-serializable
report.raise_if_invalid()   # raises jsonschema.exceptions.ValidationError
```

Use `strict=True` for the stricter schemas.

A plate is validated all the way down: its own metadata, every well it lists, and every field of view and label image
in those wells. `issue.location` says where each problem is, for example `plate`, `A/1` (a well), `A/1/0` (a field) or
`A/1/0/labels/cells`. Besides the schemas, it checks that each well's `path`, `rowIndex` and `columnIndex` agree, that
listed wells and fields exist, that field acquisitions are defined by the plate, and that the OME-Zarr version is the
same throughout. Validating a well on its own checks its fields too.

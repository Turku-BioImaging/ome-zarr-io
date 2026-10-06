"""Sphinx configuration for the ome-zarr-io documentation."""

from importlib import metadata

project = "ome-zarr-io"
author = "Junel Solis, Turku BioImaging"
copyright = "Turku BioImaging"

try:
    release = metadata.version("ome-zarr-io")
except metadata.PackageNotFoundError:  # docs built without the package installed
    release = ""

extensions = [
    "myst_parser",
    "sphinx_copybutton",
]

# The pages are plain Markdown.
source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build"]
myst_heading_anchors = 3

html_theme = "furo"
html_title = "ome-zarr-io"
html_theme_options = {
    "source_repository": "https://github.com/Turku-BioImaging/ome-zarr-io",
    "source_branch": "main",
    "source_directory": "docs/",
}

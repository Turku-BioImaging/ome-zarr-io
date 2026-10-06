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
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # the docstrings are Google style
    "sphinx.ext.intersphinx",
]

# The pages are plain Markdown.
source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build"]
myst_heading_anchors = 3

html_theme = "furo"
html_title = "ome-zarr-io"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Typography and colours, matching the Turku BioImaging site (bioimaging.fi): brand blue
# #00aeef, Source Sans. The fonts are bundled in _static/fonts (see custom.css).
# On white, #00aeef is too light for text, so light mode uses a darker shade of the same
# hue (#00739f) for text and links; every text colour has at least 4.5:1 contrast.
_fonts = {
    "font-stack": '"Source Sans 3", "Source Sans Pro", system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    "font-stack--headings": "var(--font-stack)",
    "font-stack--monospace": '"JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace',
    "font-size--normal": "112.5%",  # Source Sans is small for its size
    "code-font-size": "0.85em",
}
html_theme_options = {
    "source_repository": "https://github.com/Turku-BioImaging/ome-zarr-io",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        **_fonts,
        "color-brand-primary": "#00739f",
        "color-brand-content": "#00739f",
        "color-brand-visited": "#4a6b82",
        "color-foreground-primary": "#222222",
        "color-foreground-secondary": "#54595f",
        "color-foreground-muted": "#666666",
        "color-background-primary": "#ffffff",
        "color-background-secondary": "#f1f1f1",
        "color-background-border": "#cccccc",
        "color-code-background": "#f6f6f6",
        "color-api-name": "#00739f",
        "color-api-pre-name": "#54595f",
    },
    "dark_css_variables": {
        **_fonts,
        "color-brand-primary": "#00aeef",
        "color-brand-content": "#00aeef",
        "color-brand-visited": "#008ab9",
        "color-foreground-primary": "#f2f2f2",
        "color-foreground-secondary": "#cccccc",
        "color-foreground-muted": "#8f8f8f",
        "color-background-primary": "#000000",
        "color-background-secondary": "#111111",
        "color-background-border": "#2a2a2a",
        "color-code-background": "#111111",
        "color-api-name": "#00aeef",
        "color-api-pre-name": "#cccccc",
    },
}
pygments_style = "default"
pygments_dark_style = "github-dark"

# -- API reference -----------------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autoclass_content = "both"  # class docstring plus __init__ docstring
napoleon_google_docstring = True
napoleon_numpy_docstring = False
# Single backticks in docstrings refer to Python objects, e.g. `Writer`.
default_role = "py:obj"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "dask": ("https://docs.dask.org/en/stable/", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable/", None),
}

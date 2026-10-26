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
templates_path = ["_templates"]
exclude_patterns = ["_build"]
myst_heading_anchors = 3

html_theme = "furo"
html_title = "ome-zarr-io"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Typography and colours. These are the design tokens of the Turku BioImaging website
# (tbi-website, site/src/styles/tokens.css): a cool slate palette with a blue accent,
# in a light and a dark theme, using the system font stack. The accent is a deep blue in
# light mode (#00698f, 5.8:1 on the page) and a light blue in dark mode (#3cc4f5, 9.3:1).
_fonts = {
    "font-stack": 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    "font-stack--headings": "var(--font-stack)",
    "font-stack--monospace": 'ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace',
    "font-size--normal": "106.25%",
    "code-font-size": "0.85em",
    # A little larger than furo's defaults (87.5%, 1rem) for easier scanning.
    "sidebar-item-font-size": "95%",
    "sidebar-item-line-height": "1.25rem",
    "sidebar-caption-font-size": "var(--font-size--small)",
    "sidebar-search-input-font-size": "95%",
}
html_theme_options = {
    "source_repository": "https://github.com/Turku-BioImaging/ome-zarr-io",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        **_fonts,
        "color-brand-primary": "#00698f",
        "color-brand-content": "#00698f",
        "color-brand-visited": "#3d6a80",
        "color-foreground-primary": "#161b20",
        "color-foreground-secondary": "#4b5763",
        "color-foreground-muted": "#4b5763",
        "color-background-primary": "#f7f9fa",
        "color-background-secondary": "#ffffff",
        "color-background-border": "#d3dbe1",
        "color-code-background": "#eef2f5",
        "color-api-name": "#00698f",
        "color-api-pre-name": "#4b5763",
        "ome-accent": "#00698f",
    },
    "dark_css_variables": {
        **_fonts,
        "color-brand-primary": "#3cc4f5",
        "color-brand-content": "#3cc4f5",
        "color-brand-visited": "#8bbfd6",
        "color-foreground-primary": "#e6ebef",
        "color-foreground-secondary": "#a3afba",
        "color-foreground-muted": "#a3afba",
        "color-background-primary": "#0e1216",
        "color-background-secondary": "#161c22",
        "color-background-border": "#2b343d",
        "color-code-background": "#1d252c",
        "color-api-name": "#3cc4f5",
        "color-api-pre-name": "#a3afba",
        "ome-accent": "#3cc4f5",
    },
}
pygments_style = "a11y-light"
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

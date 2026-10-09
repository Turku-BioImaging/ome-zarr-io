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

# Typography and colours. The brand blue #00aeef comes from the Turku BioImaging site
# (bioimaging.fi). It is only 2.5:1 on white, too light for text, so light mode uses a
# deeper shade of the same hue (#006186, 6.9:1 on white and 6.1:1 on the sidebar) for
# text and links. Dark mode uses the exact brand blue (8.3:1 on black). The decorative
# accent (heading underline, text selection) is --ome-accent: orange in light mode
# (#d9480f, 4.3:1 on white) and the brand blue in dark mode. The body font is Inter,
# chosen for readability on screen; fonts are bundled in _static/fonts (see custom.css).
_fonts = {
    "font-stack": 'Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    "font-stack--headings": "var(--font-stack)",
    "font-stack--monospace": '"JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace',
    "font-size--normal": "106.25%",
    "code-font-size": "0.8em",
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
        "color-brand-primary": "#006186",
        "color-brand-content": "#006186",
        "color-brand-visited": "#4a6b82",
        "color-foreground-primary": "#222222",
        "color-foreground-secondary": "#54595f",
        "color-foreground-muted": "#595959",
        "color-background-primary": "#ffffff",
        "color-background-secondary": "#f1f1f1",
        "color-background-border": "#cccccc",
        "color-code-background": "#f6f6f6",
        "color-api-name": "#006186",
        "color-api-pre-name": "#54595f",
        "ome-accent": "#d9480f",
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
        "ome-accent": "#00aeef",
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

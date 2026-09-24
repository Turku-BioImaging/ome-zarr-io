"""OME-Zarr package for writing, reading, and validating OME-Zarr 0.5 multiscale images."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ome-zarr-io")
except PackageNotFoundError:  # not installed, e.g. running from a source checkout
    __version__ = "0.0.0+unknown"
__author__ = "Junel Solis, Turku BioImaging"
__email__ = "junel.solis@abo.fi"

from .image import OmeZarrImage
from .validator import OMEZarrValidator
from .reader import Reader, PhysicalSize
from .schema_models import Axis, Channel, Omero, Window, create_axes

__all__ = [
    "OmeZarrImage",
    "OMEZarrValidator",
    "Reader",
    "PhysicalSize",
    "Axis",
    "Channel",
    "Window",
    "Omero",
    "create_axes",
]

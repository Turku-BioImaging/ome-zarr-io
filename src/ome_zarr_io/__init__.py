"""OME-Zarr package for reading and valid OME-Zarr 0.5 multiscale images."""

import importlib.metadata

__version__ = importlib.metadata.version("ome-zarr-io")
__author__ = "Junel Solis, Turku BioImaging"
__email__ = "junel.solis@abo.fi"

from mypy.typeshed.stdlib import importlib


from .image import OmeZarrImage
from .downscaler import Downscaler
from .validator import OMEZarrValidator
from .reader import Reader, PhysicalSize
from .schema_models import (
    OMEZarrImageMetadata,
    OMEMetadata,
    Multiscale,
    Dataset,
    Axis,
    ScaleTransformation,
    TranslationTransformation,
    Channel,
    Window,
    Omero,
    create_axes,  # Unified function for creating axes
    create_scale_transformation,
    validate_tczyx_axis_ordering,
    VALID_TIME_UNITS,
    VALID_SPACE_UNITS,
)

__all__ = [
    "OmeZarrImage",
    "Downscaler",
    "OMEZarrValidator",
    "Reader",
    "PhysicalSize",
    "OMEZarrImageMetadata",
    "OMEMetadata",
    "Multiscale",
    "Dataset",
    "Axis",
    "ScaleTransformation",
    "TranslationTransformation",
    "Channel",
    "Window",
    "Omero",
    "create_axes",  # Unified function for creating axes
    "create_scale_transformation",
    "validate_tczyx_axis_ordering",
    "VALID_TIME_UNITS",
    "VALID_SPACE_UNITS",
]

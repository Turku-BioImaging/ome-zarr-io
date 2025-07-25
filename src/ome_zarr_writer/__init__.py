"""OME-Zarr Writer package for writing valid OME-Zarr 0.5 multiscale images."""

__version__ = "0.1.0"
__author__ = "Junel Solis, Turku BioImaging"
__email__ = "junel.solis@abo.fi"

from .image import OmeZarrImage
from .downscaler import Downscaler
from .validator import OMEZarrValidator
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

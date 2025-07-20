"""OME-Zarr Writer package for writing valid OME-Zarr 0.5 multiscale images."""

__version__ = "0.1.0"
__author__ = "Junel Solis, Turku BioImaging"
__email__ = "junel.solis@abo.fi"

from .image import OMEZarrImage
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
    create_2d_axes,
    create_3d_axes,
    create_cyx_axes,
    create_czyx_axes,
    create_tyx_axes,
    create_tcyx_axes,
    create_scale_transformation
)

__all__ = [
    "OMEZarrImage", 
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
    "create_2d_axes",
    "create_3d_axes",
    "create_cyx_axes",
    "create_czyx_axes", 
    "create_tyx_axes",
    "create_tcyx_axes",
    "create_scale_transformation"
]

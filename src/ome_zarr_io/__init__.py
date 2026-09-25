"""OME-Zarr package for writing, reading, and validating OME-Zarr 0.5 multiscale images."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ome-zarr-io")
except PackageNotFoundError:  # not installed, e.g. running from a source checkout
    __version__ = "0.0.0+unknown"
__author__ = "Junel Solis, Turku BioImaging"
__email__ = "junel.solis@abo.fi"

from .writer import Writer
from .validator import OMEZarrValidator
from .reader import Reader, PhysicalSize
from .report import FilesetReport, ValidationIssue, validate
from .channels import random_colors
from .schema_models import Axis

__all__ = [
    "Writer",
    "OMEZarrValidator",
    "Reader",
    "PhysicalSize",
    "validate",
    "FilesetReport",
    "ValidationIssue",
    "Axis",
    "random_colors",
]


def __getattr__(name: str):
    if name == "Omero":
        import warnings

        from .schema_models import Omero

        warnings.warn(
            "ome_zarr_io.Omero is deprecated; pass channels=... to Writer instead "
            "(Omero remains importable from ome_zarr_io.schema_models).",
            DeprecationWarning,
            stacklevel=2,
        )
        return Omero
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""
Python dataclasses implementing the OME-Zarr 0.5 image.schema.

These dataclasses provide a type-safe way to construct and validate
OME-Zarr metadata that conforms to the official image.schema specification.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum


class AxisType(Enum):
    """Enumeration for axis types."""
    CHANNEL = "channel"
    TIME = "time"
    SPACE = "space"


class TransformationType(Enum):
    """Enumeration for coordinate transformation types."""
    SCALE = "scale"
    TRANSLATION = "translation"


@dataclass
class Window:
    """OMERO window configuration for channel display."""
    start: float
    min: float
    end: float
    max: float


@dataclass
class Channel:
    """OMERO channel configuration."""
    window: Optional[Window] = None
    label: Optional[str] = None
    family: Optional[str] = None
    color: Optional[str] = None
    active: Optional[bool] = None


@dataclass
class Omero:
    """OMERO metadata for image display configuration."""
    channels: List[Channel]


@dataclass
class Axis:
    """Axis definition with name and type."""
    name: str
    type: Optional[str] = None
    unit: Optional[str] = None

    def __post_init__(self):
        """Validate axis type."""
        if self.type is not None:
            # For space axes, type is required
            if self.type == "space" and self.unit is None:
                # Unit is typically required for space axes but not enforced here
                pass
            # Validate known types
            if self.type in ["channel", "time", "space"]:
                # These are the standard types
                pass


@dataclass
class CoordinateTransformation:
    """Base class for coordinate transformations."""
    type: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        raise NotImplementedError


@dataclass
class ScaleTransformation(CoordinateTransformation):
    """Scale coordinate transformation."""
    scale: List[float]
    type: str = field(default="scale", init=False)

    def __post_init__(self):
        """Validate scale transformation."""
        if len(self.scale) < 2:
            raise ValueError("Scale array must have at least 2 elements")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": self.type,
            "scale": self.scale
        }


@dataclass
class TranslationTransformation(CoordinateTransformation):
    """Translation coordinate transformation."""
    translation: List[float]
    type: str = field(default="translation", init=False)

    def __post_init__(self):
        """Validate translation transformation."""
        if len(self.translation) < 2:
            raise ValueError("Translation array must have at least 2 elements")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": self.type,
            "translation": self.translation
        }


@dataclass
class Dataset:
    """Dataset definition with path and coordinate transformations."""
    path: str
    coordinateTransformations: List[Union[ScaleTransformation, TranslationTransformation]]

    def __post_init__(self):
        """Validate dataset."""
        if not self.coordinateTransformations:
            raise ValueError("At least one coordinate transformation is required")
        
        # Check that at least one scale transformation exists
        scale_count = sum(1 for t in self.coordinateTransformations if isinstance(t, ScaleTransformation))
        if scale_count == 0:
            raise ValueError("At least one scale transformation is required")
        if scale_count > 1:
            raise ValueError("At most one scale transformation is allowed")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "path": self.path,
            "coordinateTransformations": [t.to_dict() for t in self.coordinateTransformations]
        }


@dataclass
class Multiscale:
    """Multiscale definition containing datasets and axes."""
    datasets: List[Dataset]
    axes: List[Axis]
    name: Optional[str] = None
    coordinateTransformations: Optional[List[Union[ScaleTransformation, TranslationTransformation]]] = None

    def __post_init__(self):
        """Validate multiscale."""
        if not self.datasets:
            raise ValueError("At least one dataset is required")
        
        # Validate axes
        if len(self.axes) < 2 or len(self.axes) > 5:
            raise ValueError("Axes must have between 2 and 5 items")
        
        # Check for space axes (should have 2-3 space axes)
        space_axes = [axis for axis in self.axes if axis.type == "space"]
        if len(space_axes) < 2 or len(space_axes) > 3:
            raise ValueError("Must have between 2 and 3 space axes")
        
        # Check axis name uniqueness
        axis_names = [axis.name for axis in self.axes]
        if len(axis_names) != len(set(axis_names)):
            raise ValueError("Axis names must be unique")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "datasets": [d.to_dict() for d in self.datasets],
            "axes": [
                {
                    "name": axis.name,
                    **({} if axis.type is None else {"type": axis.type}),
                    **({} if axis.unit is None else {"unit": axis.unit})
                }
                for axis in self.axes
            ]
        }
        
        if self.name is not None:
            result["name"] = self.name
        
        if self.coordinateTransformations is not None:
            result["coordinateTransformations"] = [t.to_dict() for t in self.coordinateTransformations]
        
        return result


@dataclass
class OMEMetadata:
    """OME metadata containing multiscales and version."""
    multiscales: List[Multiscale]
    version: str
    omero: Optional[Omero] = None

    def __post_init__(self):
        """Validate OME metadata."""
        if not self.multiscales:
            raise ValueError("At least one multiscale is required")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "multiscales": [m.to_dict() for m in self.multiscales],
            "version": self.version
        }
        
        if self.omero is not None:
            omero_dict = {"channels": []}
            for channel in self.omero.channels:
                channel_dict = {}
                if channel.window is not None:
                    channel_dict["window"] = {
                        "start": channel.window.start,
                        "min": channel.window.min,
                        "end": channel.window.end,
                        "max": channel.window.max
                    }
                if channel.label is not None:
                    channel_dict["label"] = channel.label
                if channel.family is not None:
                    channel_dict["family"] = channel.family
                if channel.color is not None:
                    channel_dict["color"] = channel.color
                if channel.active is not None:
                    channel_dict["active"] = channel.active
                omero_dict["channels"].append(channel_dict)
            result["omero"] = omero_dict
        
        return result


@dataclass
class OMEZarrImageMetadata:
    """Top-level OME-Zarr image metadata."""
    ome: OMEMetadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for zarr.attrs."""
        return {
            "ome": self.ome.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OMEZarrImageMetadata":
        """Create instance from dictionary data."""
        ome_data = data["ome"]
        
        # Parse multiscales
        multiscales = []
        for ms_data in ome_data["multiscales"]:
            # Parse datasets
            datasets = []
            for ds_data in ms_data["datasets"]:
                # Parse coordinate transformations
                coord_transforms = []
                for ct_data in ds_data["coordinateTransformations"]:
                    if ct_data["type"] == "scale":
                        coord_transforms.append(ScaleTransformation(scale=ct_data["scale"]))
                    elif ct_data["type"] == "translation":
                        coord_transforms.append(TranslationTransformation(translation=ct_data["translation"]))
                
                datasets.append(Dataset(
                    path=ds_data["path"],
                    coordinateTransformations=coord_transforms
                ))
            
            # Parse axes
            axes = []
            for axis_data in ms_data["axes"]:
                axes.append(Axis(
                    name=axis_data["name"],
                    type=axis_data.get("type"),
                    unit=axis_data.get("unit")
                ))
            
            # Parse coordinate transformations if present
            coord_transforms = None
            if "coordinateTransformations" in ms_data:
                coord_transforms = []
                for ct_data in ms_data["coordinateTransformations"]:
                    if ct_data["type"] == "scale":
                        coord_transforms.append(ScaleTransformation(scale=ct_data["scale"]))
                    elif ct_data["type"] == "translation":
                        coord_transforms.append(TranslationTransformation(translation=ct_data["translation"]))
            
            multiscales.append(Multiscale(
                datasets=datasets,
                axes=axes,
                name=ms_data.get("name"),
                coordinateTransformations=coord_transforms
            ))
        
        # Parse OMERO if present
        omero = None
        if "omero" in ome_data:
            omero_data = ome_data["omero"]
            channels = []
            for ch_data in omero_data["channels"]:
                window = None
                if "window" in ch_data:
                    w_data = ch_data["window"]
                    window = Window(
                        start=w_data["start"],
                        min=w_data["min"],
                        end=w_data["end"],
                        max=w_data["max"]
                    )
                
                channels.append(Channel(
                    window=window,
                    label=ch_data.get("label"),
                    family=ch_data.get("family"),
                    color=ch_data.get("color"),
                    active=ch_data.get("active")
                ))
            
            omero = Omero(channels=channels)
        
        ome_metadata = OMEMetadata(
            multiscales=multiscales,
            version=ome_data["version"],
            omero=omero
        )
        
        return cls(ome=ome_metadata)


# Convenience functions for creating common configurations

def create_2d_axes(pixel_size_x: float, pixel_size_y: float, 
                   unit: str = "micrometer") -> List[Axis]:
    """Create standard 2D spatial axes."""
    return [
        Axis(name="y", type="space", unit=unit),
        Axis(name="x", type="space", unit=unit)
    ]


def create_3d_axes(pixel_size_x: float, pixel_size_y: float, pixel_size_z: float,
                   unit: str = "micrometer") -> List[Axis]:
    """Create standard 3D spatial axes."""
    return [
        Axis(name="z", type="space", unit=unit),
        Axis(name="y", type="space", unit=unit),
        Axis(name="x", type="space", unit=unit)
    ]


def create_scale_transformation(scales: List[float]) -> ScaleTransformation:
    """Create a scale transformation."""
    return ScaleTransformation(scale=scales)


def create_translation_transformation(translation: List[float]) -> TranslationTransformation:
    """Create a translation transformation."""
    return TranslationTransformation(translation=translation)

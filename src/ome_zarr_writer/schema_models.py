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


# Valid units for space axes according to OME-Zarr specification
VALID_SPACE_UNITS = {
    "angstrom", "attometer", "centimeter", "decimeter", "exameter", 
    "femtometer", "foot", "gigameter", "hectometer", "inch", "kilometer", 
    "megameter", "meter", "micrometer", "mile", "millimeter", "nanometer", 
    "parsec", "petameter", "picometer", "terameter", "yard", "yoctometer", 
    "yottameter", "zeptometer", "zettameter"
}

# Valid units for time axes according to OME-Zarr specification
VALID_TIME_UNITS = {
    "attosecond", "centisecond", "day", "decisecond", "exasecond", 
    "femtosecond", "gigasecond", "hectosecond", "hour", "kilosecond", 
    "megasecond", "microsecond", "millisecond", "minute", "nanosecond", 
    "petasecond", "picosecond", "second", "terasecond", "yoctosecond", 
    "yottasecond", "zeptosecond", "zettasecond"
}


@dataclass
class Axis:
    """Axis definition with name and type."""
    name: str
    type: Optional[str] = None
    unit: Optional[str] = None

    def __post_init__(self):
        """Validate axis type and unit."""
        if self.type is not None:
            # Validate known types
            valid_types = ["channel", "time", "space"]
            if self.type not in valid_types:
                raise ValueError(f"Invalid axis type '{self.type}'. Must be one of: {', '.join(valid_types)}")
            
            # For space axes, validate unit if provided
            if self.type == "space" and self.unit is not None:
                if self.unit not in VALID_SPACE_UNITS:
                    raise ValueError(
                        f"Invalid unit '{self.unit}' for space axis. "
                        f"Valid units are: {', '.join(sorted(VALID_SPACE_UNITS))}"
                    )
            
            # For time axes, validate unit if provided
            if self.type == "time" and self.unit is not None:
                if self.unit not in VALID_TIME_UNITS:
                    raise ValueError(
                        f"Invalid unit '{self.unit}' for time axis. "
                        f"Valid units are: {', '.join(sorted(VALID_TIME_UNITS))}"
                    )


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


def validate_tczyx_axis_ordering(axes: List[Axis]) -> None:
    """Validate that axes follow the TCZYX dimension ordering.
    
    Input images always follow TCZYX dimension order where T, C, Z are optional.
    Valid combinations: YX, ZYX, CYX, CZYX, TYX, TZYX, TCYX, TCZYX
    
    Args:
        axes: List of axes to validate
        
    Raises:
        ValueError: If axes don't follow TCZYX ordering
    """
    if len(axes) < 2:
        raise ValueError("Must have at least 2 axes (Y, X)")
    
    # Extract axis names
    axis_names = [axis.name.lower() for axis in axes]
    
    # Check for unique axis names first
    if len(axis_names) != len(set(axis_names)):
        raise ValueError("Axis names must be unique")
    
    # Check that last two axes are always Y, X
    if axis_names[-2:] != ['y', 'x']:
        raise ValueError("Last two axes must be Y, X in that order")
    
    # Define the expected TCZYX ordering
    expected_order = ['t', 'c', 'z', 'y', 'x']
    
    # Check if all axis names are valid
    valid_names = set(expected_order)
    for name in axis_names:
        if name not in valid_names:
            raise ValueError(f"Invalid axis name '{name}'. Valid names are: {', '.join(expected_order)}")
    
    # Check ordering - each axis should appear in the correct relative position
    last_index = -1
    for axis_name in axis_names:
        current_index = expected_order.index(axis_name)
        if current_index <= last_index:
            raise ValueError(
                f"Axes must follow TCZYX ordering. Found '{axis_name}' after a later dimension. "
                f"Expected order: {' → '.join(expected_order)}, got: {' → '.join(axis_names)}"
            )
        last_index = current_index
    
    # Validate axis types
    type_mapping = {'t': 'time', 'c': 'channel', 'z': 'space', 'y': 'space', 'x': 'space'}
    for axis in axes:
        expected_type = type_mapping[axis.name.lower()]
        if axis.type != expected_type:
            raise ValueError(f"Axis '{axis.name}' should have type '{expected_type}', got '{axis.type}'")


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
        
        # Validate axes count (2-5 for YX to TCZYX)
        if len(self.axes) < 2 or len(self.axes) > 5:
            raise ValueError("Axes must have between 2 and 5 items (YX minimum, TCZYX maximum)")
        
        # Validate TCZYX ordering (includes uniqueness check)
        validate_tczyx_axis_ordering(self.axes)

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


# Convenience functions for creating common configurations with TCZYX ordering
# Input images always follow TCZYX dimension order where T, C, Z are optional
# Minimum dimensions: YX, Maximum dimensions: TCZYX

def create_axes(
    axes: str,
    x_size: float,
    y_size: float,
    z_size: Optional[float] = None,
    unit: str = "micrometer",
    time_unit: Optional[str] = None
) -> List[Axis]:
    """Create axes for images following TCZYX dimension ordering.
    
    Args:
        axes: String specifying which axes to include. Must follow TCZYX ordering.
              Valid combinations: "yx", "zyx", "cyx", "czyx", "tyx", "tzyx", "tcyx", "tczyx"
        x_size: Pixel size in X dimension (for documentation, not used in axis creation)
        y_size: Pixel size in Y dimension (for documentation, not used in axis creation)
        z_size: Pixel size in Z dimension (for documentation, required if 'z' in axes)
        unit: Unit for spatial dimensions (X, Y, Z)
        time_unit: Unit for time dimension (T), if present. Valid units include:
                  second, millisecond, microsecond, nanosecond, minute, hour, day, etc.
        
    Returns:
        List of Axis objects in TCZYX order
        
    Raises:
        ValueError: If axes string is invalid or z_size not provided when needed
        
    Examples:
        >>> # 2D spatial image
        >>> axes = create_axes("yx", 0.1, 0.1)
        
        >>> # Multichannel 3D image  
        >>> axes = create_axes("czyx", 0.1, 0.1, 0.3)
        
        >>> # Time-series multichannel 2D with time unit
        >>> axes = create_axes("tcyx", 0.2, 0.2, time_unit="second")
        
        >>> # Full 5D image with custom units
        >>> axes = create_axes("tczyx", 0.25, 0.25, 0.5, unit="nanometer", time_unit="millisecond")
    """
    # Normalize axes string
    axes = axes.lower().strip()
    
    # Valid axis combinations following TCZYX ordering
    valid_combinations = {
        "yx", "zyx", "cyx", "czyx", 
        "tyx", "tzyx", "tcyx", "tczyx"
    }
    
    if axes not in valid_combinations:
        raise ValueError(
            f"Invalid axes '{axes}'. Must be one of: {', '.join(sorted(valid_combinations))}. "
            f"Axes must follow TCZYX ordering where T, C, Z are optional."
        )
    
    # Check if z_size is required
    if 'z' in axes and z_size is None:
        raise ValueError(f"z_size is required when 'z' is in axes '{axes}'")
    
    # Build axis list in TCZYX order
    axis_list = []
    
    # Add dimensions in TCZYX order
    if 't' in axes:
        axis_list.append(Axis(name="t", type="time", unit=time_unit))
    
    if 'c' in axes:
        axis_list.append(Axis(name="c", type="channel"))
    
    if 'z' in axes:
        axis_list.append(Axis(name="z", type="space", unit=unit))
    
    # Y and X are always last and required
    axis_list.append(Axis(name="y", type="space", unit=unit))
    axis_list.append(Axis(name="x", type="space", unit=unit))
    
    # Validate the created axes follow TCZYX ordering
    validate_tczyx_axis_ordering(axis_list)
    
    return axis_list


# Convenience functions for backward compatibility
def create_yx_axes(pixel_size_x: float, pixel_size_y: float, 
                   unit: str = "micrometer") -> List[Axis]:
    """Create 2D spatial axes (Y, X order) - minimum required dimensions."""
    return create_axes("yx", pixel_size_x, pixel_size_y, unit=unit)


def create_zyx_axes(pixel_size_x: float, pixel_size_y: float, pixel_size_z: float,
                    unit: str = "micrometer") -> List[Axis]:
    """Create 3D spatial axes (Z, Y, X order) from TCZYX dimension order."""
    return create_axes("zyx", pixel_size_x, pixel_size_y, pixel_size_z, unit=unit)


def create_cyx_axes(pixel_size_x: float, pixel_size_y: float,
                    unit: str = "micrometer") -> List[Axis]:
    """Create axes for multichannel 2D images (C, Y, X) from TCZYX dimension order."""
    return create_axes("cyx", pixel_size_x, pixel_size_y, unit=unit)


def create_czyx_axes(pixel_size_x: float, pixel_size_y: float, pixel_size_z: float,
                     unit: str = "micrometer") -> List[Axis]:
    """Create axes for multichannel 3D images (C, Z, Y, X) from TCZYX dimension order."""
    return create_axes("czyx", pixel_size_x, pixel_size_y, pixel_size_z, unit=unit)


def create_tyx_axes(pixel_size_x: float, pixel_size_y: float,
                    unit: str = "micrometer") -> List[Axis]:
    """Create axes for time-series 2D images (T, Y, X) from TCZYX dimension order."""
    return create_axes("tyx", pixel_size_x, pixel_size_y, unit=unit)


def create_tzyx_axes(pixel_size_x: float, pixel_size_y: float, pixel_size_z: float,
                     unit: str = "micrometer") -> List[Axis]:
    """Create axes for time-series 3D images (T, Z, Y, X) from TCZYX dimension order."""
    return create_axes("tzyx", pixel_size_x, pixel_size_y, pixel_size_z, unit=unit)


def create_tcyx_axes(pixel_size_x: float, pixel_size_y: float,
                     unit: str = "micrometer") -> List[Axis]:
    """Create axes for time-series multichannel 2D images (T, C, Y, X) from TCZYX dimension order."""
    return create_axes("tcyx", pixel_size_x, pixel_size_y, unit=unit)


def create_tczyx_axes(pixel_size_x: float, pixel_size_y: float, pixel_size_z: float,
                      unit: str = "micrometer") -> List[Axis]:
    """Create axes for time-series multichannel 3D images (T, C, Z, Y, X) - maximum dimensions from TCZYX order."""
    return create_axes("tczyx", pixel_size_x, pixel_size_y, pixel_size_z, unit=unit)


def create_scale_transformation(scales: List[float]) -> ScaleTransformation:
    """Create a scale transformation.
    
    Args:
        scales: Scale factors for each dimension (same order as axes)
        
    Returns:
        ScaleTransformation object
    """
    return ScaleTransformation(scale=scales)


def create_translation_transformation(translation: List[float]) -> TranslationTransformation:
    """Create a translation transformation.
    
    Args:
        translation: Translation offsets for each dimension (same order as axes)
        
    Returns:
        TranslationTransformation object
    """
    return TranslationTransformation(translation=translation)


# Backward compatibility aliases
def create_2d_axes(pixel_size_x: float, pixel_size_y: float, 
                   unit: str = "micrometer") -> List[Axis]:
    """Backward compatibility alias for create_yx_axes."""
    return create_yx_axes(pixel_size_x, pixel_size_y, unit)


def create_3d_axes(pixel_size_x: float, pixel_size_y: float, pixel_size_z: float,
                   unit: str = "micrometer") -> List[Axis]:
    """Backward compatibility alias for create_zyx_axes."""
    return create_zyx_axes(pixel_size_x, pixel_size_y, pixel_size_z, unit)

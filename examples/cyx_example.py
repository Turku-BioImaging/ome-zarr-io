#!/usr/bin/env python3
"""
Example demonstrating CYX (Channel, Y, X) image handling with OME-Zarr schema dataclasses.

This example shows how to create metadata for multichannel images using the new
convenience functions for different dimensional arrangements.
"""

from ome_zarr_writer.schema_models import (
    OMEZarrImageMetadata,
    OMEMetadata,
    Multiscale,
    Dataset,
    Channel,
    Window,
    Omero,
    create_cyx_axes,
    create_czyx_axes,
    create_tyx_axes,
    create_tcyx_axes,
    create_scale_transformation,
)
import json


def create_cyx_metadata():
    """Create metadata for a CYX (Channel, Y, X) image."""
    print("Creating CYX (Channel, Y, X) image metadata...")

    # Image dimensions: (3 channels, 512 height, 512 width)
    pixel_size_x = 0.1  # micrometers per pixel
    pixel_size_y = 0.1  # micrometers per pixel

    # Create axes for CYX layout
    axes = create_cyx_axes(pixel_size_x, pixel_size_y, unit="micrometer")

    # Create coordinate transformation
    # Scale: [1.0 for channel (no scaling), pixel_size_y, pixel_size_x]
    scale_transform = create_scale_transformation([1.0, pixel_size_y, pixel_size_x])

    # Create dataset
    dataset = Dataset(path="0", coordinateTransformations=[scale_transform])

    # Create OMERO channel settings for RGB channels
    channels = [
        Channel(
            label="Red",
            color="FF0000",
            window=Window(start=0, min=0, end=255, max=255),
            active=True,
        ),
        Channel(
            label="Green",
            color="00FF00",
            window=Window(start=0, min=0, end=255, max=255),
            active=True,
        ),
        Channel(
            label="Blue",
            color="0000FF",
            window=Window(start=0, min=0, end=255, max=255),
            active=True,
        ),
    ]

    omero = Omero(channels=channels)

    # Create multiscale
    multiscale = Multiscale(name="RGB CYX Image", datasets=[dataset], axes=axes)

    # Create OME metadata with OMERO
    ome_metadata = OMEMetadata(multiscales=[multiscale], version="0.5", omero=omero)

    # Create top-level metadata
    metadata = OMEZarrImageMetadata(ome=ome_metadata)

    # Convert to dictionary
    attrs_dict = metadata.to_dict()
    print("Generated CYX metadata:")
    print(json.dumps(attrs_dict, indent=2))

    return metadata


def create_czyx_metadata():
    """Create metadata for a CZYX (Channel, Z, Y, X) image."""
    print("\nCreating CZYX (Channel, Z, Y, X) image metadata...")

    # Image dimensions: (2 channels, 20 slices, 256 height, 256 width)
    pixel_size_x = 0.1  # micrometers per pixel
    pixel_size_y = 0.1  # micrometers per pixel
    pixel_size_z = 0.3  # micrometers per slice

    # Create axes for CZYX layout
    axes = create_czyx_axes(pixel_size_x, pixel_size_y, pixel_size_z, unit="micrometer")

    # Create coordinate transformation
    # Scale: [1.0 for channel, pixel_size_z, pixel_size_y, pixel_size_x]
    scale_transform = create_scale_transformation(
        [1.0, pixel_size_z, pixel_size_y, pixel_size_x]
    )

    # Create dataset
    dataset = Dataset(path="0", coordinateTransformations=[scale_transform])

    # Create OMERO channel settings for fluorescence channels
    channels = [
        Channel(
            label="DAPI",
            color="0000FF",
            window=Window(start=100, min=0, end=4095, max=4095),
            active=True,
        ),
        Channel(
            label="GFP",
            color="00FF00",
            window=Window(start=200, min=0, end=3000, max=4095),
            active=True,
        ),
    ]

    omero = Omero(channels=channels)

    # Create multiscale
    multiscale = Multiscale(
        name="Fluorescence CZYX Image", datasets=[dataset], axes=axes
    )

    # Create OME metadata with OMERO
    ome_metadata = OMEMetadata(multiscales=[multiscale], version="0.5", omero=omero)

    # Create top-level metadata
    metadata = OMEZarrImageMetadata(ome=ome_metadata)

    # Convert to dictionary
    attrs_dict = metadata.to_dict()
    print("Generated CZYX metadata:")
    print(json.dumps(attrs_dict, indent=2))

    return metadata


def create_tyx_metadata():
    """Create metadata for a TYX (Time, Y, X) image."""
    print("\nCreating TYX (Time, Y, X) time-series image metadata...")

    # Image dimensions: (100 timepoints, 256 height, 256 width)
    pixel_size_x = 0.2  # micrometers per pixel
    pixel_size_y = 0.2  # micrometers per pixel

    # Create axes for TYX layout
    axes = create_tyx_axes(pixel_size_x, pixel_size_y, unit="micrometer")

    # Create coordinate transformation
    # Scale: [1.0 for time (no scaling), pixel_size_y, pixel_size_x]
    scale_transform = create_scale_transformation([1.0, pixel_size_y, pixel_size_x])

    # Create dataset
    dataset = Dataset(path="0", coordinateTransformations=[scale_transform])

    # Create multiscale
    multiscale = Multiscale(name="Time-series TYX Image", datasets=[dataset], axes=axes)

    # Create OME metadata
    ome_metadata = OMEMetadata(multiscales=[multiscale], version="0.5")

    # Create top-level metadata
    metadata = OMEZarrImageMetadata(ome=ome_metadata)

    # Convert to dictionary
    attrs_dict = metadata.to_dict()
    print("Generated TYX metadata:")
    print(json.dumps(attrs_dict, indent=2))

    return metadata


def create_tcyx_metadata():
    """Create metadata for a TCYX (Time, Channel, Y, X) image."""
    print(
        "\nCreating TCYX (Time, Channel, Y, X) time-series multichannel image metadata..."
    )

    # Image dimensions: (50 timepoints, 3 channels, 128 height, 128 width)
    pixel_size_x = 0.25  # micrometers per pixel
    pixel_size_y = 0.25  # micrometers per pixel

    # Create axes for TCYX layout
    axes = create_tcyx_axes(pixel_size_x, pixel_size_y, unit="micrometer")

    # Create coordinate transformation
    # Scale: [1.0 for time, 1.0 for channel, pixel_size_y, pixel_size_x]
    scale_transform = create_scale_transformation(
        [1.0, 1.0, pixel_size_y, pixel_size_x]
    )

    # Create dataset
    dataset = Dataset(path="0", coordinateTransformations=[scale_transform])

    # Create OMERO channel settings
    channels = [
        Channel(
            label="Phase",
            color="888888",
            window=Window(start=0, min=0, end=1000, max=4095),
            active=True,
        ),
        Channel(
            label="Calcium",
            color="00FFFF",
            window=Window(start=100, min=0, end=2000, max=4095),
            active=True,
        ),
        Channel(
            label="Membrane",
            color="FF00FF",
            window=Window(start=50, min=0, end=1500, max=4095),
            active=False,
        ),
    ]

    omero = Omero(channels=channels)

    # Create multiscale
    multiscale = Multiscale(
        name="Time-series Multichannel TCYX Image", datasets=[dataset], axes=axes
    )

    # Create OME metadata with OMERO
    ome_metadata = OMEMetadata(multiscales=[multiscale], version="0.5", omero=omero)

    # Create top-level metadata
    metadata = OMEZarrImageMetadata(ome=ome_metadata)

    # Convert to dictionary
    attrs_dict = metadata.to_dict()
    print("Generated TCYX metadata:")
    print(json.dumps(attrs_dict, indent=2))

    return metadata


def demonstrate_multiscale_cyx():
    """Create multiscale CYX metadata with multiple resolution levels."""
    print("\nCreating multiscale CYX image with pyramid...")

    # Create axes for CYX layout
    axes = create_cyx_axes(0.1, 0.1, unit="micrometer")

    # Create datasets for multiple resolutions
    datasets = []
    for level in range(3):
        scale_factor = 2**level
        scale_transform = create_scale_transformation(
            [
                1.0,  # No scaling for channel
                0.1 * scale_factor,  # Y pixel size
                0.1 * scale_factor,  # X pixel size
            ]
        )

        dataset = Dataset(path=str(level), coordinateTransformations=[scale_transform])
        datasets.append(dataset)

    # Create OMERO channels
    channels = [
        Channel(label="Red", color="FF0000", active=True),
        Channel(label="Green", color="00FF00", active=True),
        Channel(label="Blue", color="0000FF", active=False),
    ]
    omero = Omero(channels=channels)

    # Create multiscale
    multiscale = Multiscale(name="Multiscale CYX Pyramid", datasets=datasets, axes=axes)

    # Create OME metadata
    ome_metadata = OMEMetadata(multiscales=[multiscale], version="0.5", omero=omero)

    # Create top-level metadata
    metadata = OMEZarrImageMetadata(ome=ome_metadata)

    # Convert to dictionary
    attrs_dict = metadata.to_dict()
    print("Generated multiscale CYX metadata:")
    print(json.dumps(attrs_dict, indent=2))

    print(f"\nCreated CYX pyramid with {len(datasets)} resolution levels:")
    for i, dataset in enumerate(datasets):
        scale = dataset.coordinateTransformations[0].scale
        print(f"  Level {i}: scales = {scale} (channel, μm/pixel Y, μm/pixel X)")

    return metadata


def demonstrate_usage_patterns():
    """Show common usage patterns for different image types."""
    print("\n" + "=" * 60)
    print("USAGE PATTERNS FOR DIFFERENT IMAGE TYPES")
    print("=" * 60)

    print("\n1. Microscopy RGB image (CYX):")
    print("   - Use create_cyx_axes()")
    print("   - Scale: [1.0, pixel_size_y, pixel_size_x]")
    print("   - OMERO channels for RGB display")

    print("\n2. Fluorescence Z-stack (CZYX):")
    print("   - Use create_czyx_axes()")
    print("   - Scale: [1.0, pixel_size_z, pixel_size_y, pixel_size_x]")
    print("   - OMERO channels for fluorescence colors")

    print("\n3. Time-lapse movie (TYX):")
    print("   - Use create_tyx_axes()")
    print("   - Scale: [1.0, pixel_size_y, pixel_size_x]")
    print("   - Time axis typically has no physical unit")

    print("\n4. Time-lapse multichannel (TCYX):")
    print("   - Use create_tcyx_axes()")
    print("   - Scale: [1.0, 1.0, pixel_size_y, pixel_size_x]")
    print("   - Combines time-series with multichannel imaging")

    print("\n5. Multiscale pyramids:")
    print("   - Any of the above can be made multiscale")
    print("   - Create multiple datasets with different scale factors")
    print("   - Keep non-spatial dimensions (time, channel) at scale 1.0")


if __name__ == "__main__":
    print("OME-Zarr CYX and Multichannel Examples")
    print("=" * 50)

    # Run examples for different dimensional arrangements
    create_cyx_metadata()
    create_czyx_metadata()
    create_tyx_metadata()
    create_tcyx_metadata()
    demonstrate_multiscale_cyx()
    demonstrate_usage_patterns()

    print("\nAll CYX examples completed successfully!")

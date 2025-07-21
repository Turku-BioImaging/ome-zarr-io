#!/usr/bin/env python3
"""
Example demonstrating how to use the OME-Zarr schema dataclasses.

This example shows how to create valid OME-Zarr metadata using the dataclasses
that implement the image.schema specification.
"""

from ome_zarr_writer.schema_models import (
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
    create_scale_transformation,
)
import json


def create_simple_2d_metadata():
    """Create metadata for a simple 2D image."""
    print("Creating simple 2D image metadata...")

    # Create axes for 2D image (Y, X)
    axes = create_2d_axes(0.1, 0.1, unit="micrometer")

    # Create coordinate transformations for a single resolution
    scale_transform = create_scale_transformation([0.1, 0.1])  # 0.1 μm/pixel

    # Create dataset
    dataset = Dataset(path="0", coordinateTransformations=[scale_transform])

    # Create multiscale
    multiscale = Multiscale(name="2D Image", datasets=[dataset], axes=axes)

    # Create OME metadata
    ome_metadata = OMEMetadata(multiscales=[multiscale], version="0.5")

    # Create top-level metadata
    metadata = OMEZarrImageMetadata(ome=ome_metadata)

    # Convert to dictionary for zarr attrs
    attrs_dict = metadata.to_dict()
    print("Generated metadata:")
    print(json.dumps(attrs_dict, indent=2))

    return metadata


def create_multiscale_3d_metadata():
    """Create metadata for a 3D image with multiple resolution levels."""
    print("\nCreating multiscale 3D image metadata...")

    # Create axes for 3D image (Z, Y, X)
    axes = create_3d_axes(0.1, 0.1, 0.2, unit="micrometer")

    # Create datasets for multiple resolutions
    datasets = []
    for level in range(3):
        scale_factor = 2**level
        scale_transform = ScaleTransformation(
            scale=[
                0.2 * scale_factor,  # Z
                0.1 * scale_factor,  # Y
                0.1 * scale_factor,  # X
            ]
        )

        dataset = Dataset(path=str(level), coordinateTransformations=[scale_transform])
        datasets.append(dataset)

    # Create multiscale
    multiscale = Multiscale(name="3D Multiscale Image", datasets=datasets, axes=axes)

    # Create OME metadata
    ome_metadata = OMEMetadata(multiscales=[multiscale], version="0.5")

    # Create top-level metadata
    metadata = OMEZarrImageMetadata(ome=ome_metadata)

    # Convert to dictionary
    attrs_dict = metadata.to_dict()
    print("Generated metadata:")
    print(json.dumps(attrs_dict, indent=2))

    return metadata


def create_multichannel_metadata_with_omero():
    """Create metadata for a multichannel image with OMERO display settings."""
    print("\nCreating multichannel image metadata with OMERO settings...")

    # Create axes for multichannel 2D image (C, Y, X)
    axes = [
        Axis(name="c", type="channel"),
        Axis(name="y", type="space", unit="micrometer"),
        Axis(name="x", type="space", unit="micrometer"),
    ]

    # Create coordinate transformations
    scale_transform = ScaleTransformation(
        scale=[1.0, 0.1, 0.1]
    )  # No scaling for channel, 0.1 μm/pixel for spatial

    # Create dataset
    dataset = Dataset(path="0", coordinateTransformations=[scale_transform])

    # Create OMERO channel settings
    channels = [
        Channel(
            label="DAPI",
            color="0000FF",
            window=Window(start=0, min=0, end=4095, max=4095),
            active=True,
        ),
        Channel(
            label="GFP",
            color="00FF00",
            window=Window(start=0, min=0, end=4095, max=4095),
            active=True,
        ),
        Channel(
            label="RFP",
            color="FF0000",
            window=Window(start=0, min=0, end=4095, max=4095),
            active=False,
        ),
    ]

    omero = Omero(channels=channels)

    # Create multiscale
    multiscale = Multiscale(name="Multichannel Image", datasets=[dataset], axes=axes)

    # Create OME metadata with OMERO
    ome_metadata = OMEMetadata(multiscales=[multiscale], version="0.5", omero=omero)

    # Create top-level metadata
    metadata = OMEZarrImageMetadata(ome=ome_metadata)

    # Convert to dictionary
    attrs_dict = metadata.to_dict()
    print("Generated metadata:")
    print(json.dumps(attrs_dict, indent=2))

    return metadata


def create_with_translation():
    """Create metadata with both scale and translation transformations."""
    print("\nCreating metadata with scale and translation transformations...")

    # Create axes
    axes = create_2d_axes(0.1, 0.1)

    # Create transformations with both scale and translation
    scale_transform = ScaleTransformation(scale=[0.1, 0.1])
    translation_transform = TranslationTransformation(translation=[100.0, 50.0])

    # Create dataset
    dataset = Dataset(
        path="0", coordinateTransformations=[scale_transform, translation_transform]
    )

    # Create multiscale
    multiscale = Multiscale(name="Translated Image", datasets=[dataset], axes=axes)

    # Create OME metadata
    ome_metadata = OMEMetadata(multiscales=[multiscale], version="0.5")

    # Create top-level metadata
    metadata = OMEZarrImageMetadata(ome=ome_metadata)

    # Convert to dictionary
    attrs_dict = metadata.to_dict()
    print("Generated metadata:")
    print(json.dumps(attrs_dict, indent=2))

    return metadata


def demonstrate_validation():
    """Demonstrate validation features of the dataclasses."""
    print("\nDemonstrating validation...")

    try:
        # This should fail - no datasets
        Multiscale(datasets=[], axes=create_2d_axes(0.1, 0.1))
    except ValueError as e:
        print(f"✓ Validation caught empty datasets: {e}")

    try:
        # This should fail - only one space axis
        axes = [Axis(name="x", type="space", unit="micrometer")]
        dataset = Dataset(
            path="0", coordinateTransformations=[ScaleTransformation(scale=[0.1])]
        )
        Multiscale(datasets=[dataset], axes=axes)
    except ValueError as e:
        print(f"✓ Validation caught insufficient space axes: {e}")

    try:
        # This should fail - no scale transformation
        axes = create_2d_axes(0.1, 0.1)
        dataset = Dataset(
            path="0",
            coordinateTransformations=[
                TranslationTransformation(translation=[0.0, 0.0])
            ],
        )
        Multiscale(datasets=[dataset], axes=axes)
    except ValueError as e:
        print(f"✓ Validation caught missing scale transformation: {e}")

    print("All validations working correctly!")


def demonstrate_roundtrip():
    """Demonstrate converting to dict and back to dataclasses."""
    print("\nDemonstrating roundtrip conversion...")

    # Create original metadata
    original = create_simple_2d_metadata()

    # Convert to dict
    data_dict = original.to_dict()

    # Convert back to dataclasses
    reconstructed = OMEZarrImageMetadata.from_dict(data_dict)

    # Verify they're equivalent
    reconstructed_dict = reconstructed.to_dict()

    if data_dict == reconstructed_dict:
        print("✓ Roundtrip conversion successful!")
    else:
        print("✗ Roundtrip conversion failed!")
        print("Original:", json.dumps(data_dict, indent=2))
        print("Reconstructed:", json.dumps(reconstructed_dict, indent=2))


if __name__ == "__main__":
    print("OME-Zarr Schema Dataclasses Example")
    print("=" * 40)

    # Run examples
    create_simple_2d_metadata()
    create_multiscale_3d_metadata()
    create_multichannel_metadata_with_omero()
    create_with_translation()

    # Demonstrate validation
    demonstrate_validation()

    # Demonstrate roundtrip
    demonstrate_roundtrip()

    print("\nExample completed successfully!")

#!/usr/bin/env python3
"""
Comprehensive example demonstrating TCZYX dimension ordering in OME-Zarr schema dataclasses.

This example shows how to create metadata for all valid TCZYX dimension combinations:
- YX (minimum): 2D spatial image
- ZYX: 3D spatial image
- CYX: Multichannel 2D image
- CZYX: Multichannel 3D image
- TYX: Time-series 2D image
- TZYX: Time-series 3D image
- TCYX: Time-series multichannel 2D image
- TCZYX (maximum): Time-series multichannel 3D image

All dimensions follow the strict TCZYX ordering where T, C, Z are optional.
"""

from ome_zarr_writer.schema_models import (
    OMEZarrImageMetadata,
    OMEMetadata,
    Multiscale,
    Dataset,
    Channel,
    Omero,
    create_yx_axes,
    create_zyx_axes,
    create_cyx_axes,
    create_czyx_axes,
    create_tyx_axes,
    create_tzyx_axes,
    create_tcyx_axes,
    create_tczyx_axes,
    create_scale_transformation,
    validate_tczyx_axis_ordering,
)


def create_basic_metadata(axes, scales, name, channels=None):
    """Helper function to create basic OME-Zarr metadata."""
    # Create scale transformation
    scale_transform = create_scale_transformation(scales)

    # Create dataset
    dataset = Dataset(path="0", coordinateTransformations=[scale_transform])

    # Create multiscale
    multiscale = Multiscale(datasets=[dataset], axes=axes, name=name)

    # Create OME metadata
    ome_kwargs = {"multiscales": [multiscale], "version": "0.5"}
    if channels:
        ome_kwargs["omero"] = Omero(channels=channels)

    ome_metadata = OMEMetadata(**ome_kwargs)

    # Create top-level metadata
    return OMEZarrImageMetadata(ome=ome_metadata)


def demonstrate_yx_dimensions():
    """YX: Minimum 2D spatial image (e.g., single grayscale image)."""
    print("1. YX Dimensions - 2D Spatial Image")
    print("=" * 50)

    # Image shape: (512, 512)
    axes = create_yx_axes(0.1, 0.1, unit="micrometer")
    scales = [0.1, 0.1]  # Y, X pixel sizes

    metadata = create_basic_metadata(axes, scales, "Grayscale 2D Image")

    print(f"Axes: {[ax.name for ax in axes]}")
    print(f"Scale: {scales}")
    print(f"Dimensions represent: Y={axes[0].name}, X={axes[1].name}")
    print("Valid TCZYX ordering: ✓")

    # Validate ordering
    validate_tczyx_axis_ordering(axes)
    print("Validation passed ✓\n")

    return metadata


def demonstrate_zyx_dimensions():
    """ZYX: 3D spatial image (e.g., confocal Z-stack)."""
    print("2. ZYX Dimensions - 3D Spatial Image")
    print("=" * 50)

    # Image shape: (20, 256, 256)
    axes = create_zyx_axes(0.1, 0.1, 0.3, unit="micrometer")
    scales = [0.3, 0.1, 0.1]  # Z, Y, X pixel sizes

    metadata = create_basic_metadata(axes, scales, "Confocal Z-stack")

    print(f"Axes: {[ax.name for ax in axes]}")
    print(f"Scale: {scales}")
    print(f"Dimensions represent: Z={axes[0].name}, Y={axes[1].name}, X={axes[2].name}")
    print("Valid TCZYX ordering: ✓")

    validate_tczyx_axis_ordering(axes)
    print("Validation passed ✓\n")

    return metadata


def demonstrate_cyx_dimensions():
    """CYX: Multichannel 2D image (e.g., RGB or fluorescence)."""
    print("3. CYX Dimensions - Multichannel 2D Image")
    print("=" * 50)

    # Image shape: (3, 512, 512)
    axes = create_cyx_axes(0.1, 0.1, unit="micrometer")
    scales = [1.0, 0.1, 0.1]  # C (no scaling), Y, X pixel sizes

    # Create OMERO channels for RGB
    channels = [
        Channel(label="Red", color="FF0000", active=True),
        Channel(label="Green", color="00FF00", active=True),
        Channel(label="Blue", color="0000FF", active=True),
    ]

    metadata = create_basic_metadata(axes, scales, "RGB Image", channels)

    print(f"Axes: {[ax.name for ax in axes]}")
    print(f"Scale: {scales}")
    print(f"Dimensions represent: C={axes[0].name}, Y={axes[1].name}, X={axes[2].name}")
    print(f"Channel labels: {[ch.label for ch in channels]}")
    print("Valid TCZYX ordering: ✓")

    validate_tczyx_axis_ordering(axes)
    print("Validation passed ✓\n")

    return metadata


def demonstrate_czyx_dimensions():
    """CZYX: Multichannel 3D image (e.g., fluorescence Z-stack)."""
    print("4. CZYX Dimensions - Multichannel 3D Image")
    print("=" * 50)

    # Image shape: (2, 15, 256, 256)
    axes = create_czyx_axes(0.1, 0.1, 0.25, unit="micrometer")
    scales = [1.0, 0.25, 0.1, 0.1]  # C (no scaling), Z, Y, X pixel sizes

    # Create OMERO channels for fluorescence
    channels = [
        Channel(label="DAPI", color="0000FF", active=True),
        Channel(label="GFP", color="00FF00", active=True),
    ]

    metadata = create_basic_metadata(axes, scales, "Fluorescence Z-stack", channels)

    print(f"Axes: {[ax.name for ax in axes]}")
    print(f"Scale: {scales}")
    print(
        f"Dimensions represent: C={axes[0].name}, Z={axes[1].name}, Y={axes[2].name}, X={axes[3].name}"
    )
    print(f"Channel labels: {[ch.label for ch in channels]}")
    print("Valid TCZYX ordering: ✓")

    validate_tczyx_axis_ordering(axes)
    print("Validation passed ✓\n")

    return metadata


def demonstrate_tyx_dimensions():
    """TYX: Time-series 2D image (e.g., live cell imaging)."""
    print("5. TYX Dimensions - Time-series 2D Image")
    print("=" * 50)

    # Image shape: (100, 256, 256)
    axes = create_tyx_axes(0.15, 0.15, unit="micrometer")
    scales = [1.0, 0.15, 0.15]  # T (no scaling), Y, X pixel sizes

    metadata = create_basic_metadata(axes, scales, "Live Cell Time-lapse")

    print(f"Axes: {[ax.name for ax in axes]}")
    print(f"Scale: {scales}")
    print(f"Dimensions represent: T={axes[0].name}, Y={axes[1].name}, X={axes[2].name}")
    print("Valid TCZYX ordering: ✓")

    validate_tczyx_axis_ordering(axes)
    print("Validation passed ✓\n")

    return metadata


def demonstrate_tzyx_dimensions():
    """TZYX: Time-series 3D image (e.g., 4D live imaging)."""
    print("6. TZYX Dimensions - Time-series 3D Image")
    print("=" * 50)

    # Image shape: (50, 10, 128, 128)
    axes = create_tzyx_axes(0.2, 0.2, 0.5, unit="micrometer")
    scales = [1.0, 0.5, 0.2, 0.2]  # T (no scaling), Z, Y, X pixel sizes

    metadata = create_basic_metadata(axes, scales, "4D Live Imaging")

    print(f"Axes: {[ax.name for ax in axes]}")
    print(f"Scale: {scales}")
    print(
        f"Dimensions represent: T={axes[0].name}, Z={axes[1].name}, Y={axes[2].name}, X={axes[3].name}"
    )
    print("Valid TCZYX ordering: ✓")

    validate_tczyx_axis_ordering(axes)
    print("Validation passed ✓\n")

    return metadata


def demonstrate_tcyx_dimensions():
    """TCYX: Time-series multichannel 2D image (e.g., live fluorescence)."""
    print("7. TCYX Dimensions - Time-series Multichannel 2D Image")
    print("=" * 50)

    # Image shape: (50, 2, 128, 128)
    axes = create_tcyx_axes(0.2, 0.2, unit="micrometer")
    scales = [1.0, 1.0, 0.2, 0.2]  # T (no scaling), C (no scaling), Y, X pixel sizes

    # Create OMERO channels
    channels = [
        Channel(label="Calcium", color="00FFFF", active=True),
        Channel(label="Membrane", color="FF00FF", active=True),
    ]

    metadata = create_basic_metadata(
        axes, scales, "Live Fluorescence Time-lapse", channels
    )

    print(f"Axes: {[ax.name for ax in axes]}")
    print(f"Scale: {scales}")
    print(
        f"Dimensions represent: T={axes[0].name}, C={axes[1].name}, Y={axes[2].name}, X={axes[3].name}"
    )
    print(f"Channel labels: {[ch.label for ch in channels]}")
    print("Valid TCZYX ordering: ✓")

    validate_tczyx_axis_ordering(axes)
    print("Validation passed ✓\n")

    return metadata


def demonstrate_tczyx_dimensions():
    """TCZYX: Maximum dimensions - time-series multichannel 3D image."""
    print("8. TCZYX Dimensions - Time-series Multichannel 3D Image (Maximum)")
    print("=" * 70)

    # Image shape: (20, 3, 8, 64, 64)
    axes = create_tczyx_axes(0.25, 0.25, 0.5, unit="micrometer")
    scales = [1.0, 1.0, 0.5, 0.25, 0.25]  # T, C (no scaling), Z, Y, X pixel sizes

    # Create OMERO channels for 3-channel fluorescence
    channels = [
        Channel(label="DAPI", color="0000FF", active=True),
        Channel(label="GFP", color="00FF00", active=True),
        Channel(label="RFP", color="FF0000", active=False),
    ]

    metadata = create_basic_metadata(axes, scales, "5D Live Cell Imaging", channels)

    print(f"Axes: {[ax.name for ax in axes]}")
    print(f"Scale: {scales}")
    print(
        f"Dimensions represent: T={axes[0].name}, C={axes[1].name}, Z={axes[2].name}, Y={axes[3].name}, X={axes[4].name}"
    )
    print(f"Channel labels: {[ch.label for ch in channels]}")
    print("Valid TCZYX ordering: ✓")
    print("This represents the maximum possible dimensions in TCZYX ordering!")

    validate_tczyx_axis_ordering(axes)
    print("Validation passed ✓\n")

    return metadata


def demonstrate_invalid_orderings():
    """Show examples of invalid orderings that will be rejected."""
    print("Invalid Dimension Orderings (Will be Rejected)")
    print("=" * 60)

    invalid_cases = [
        {"name": "XY instead of YX", "axes": ["x", "y"], "types": ["space", "space"]},
        {
            "name": "CZT instead of TCZ",
            "axes": ["c", "z", "t", "y", "x"],
            "types": ["channel", "space", "time", "space", "space"],
        },
        {
            "name": "ZC instead of CZ",
            "axes": ["z", "c", "y", "x"],
            "types": ["space", "channel", "space", "space"],
        },
        {
            "name": "Invalid axis name 'W'",
            "axes": ["w", "y", "x"],
            "types": ["space", "space", "space"],
        },
    ]

    for i, case in enumerate(invalid_cases, 1):
        print(f"{i}. {case['name']}")
        try:
            from ome_zarr_writer.schema_models import Axis

            axes = [
                Axis(
                    name=name,
                    type=type_name,
                    unit="micrometer" if type_name == "space" else None,
                )
                for name, type_name in zip(case["axes"], case["types"])
            ]
            validate_tczyx_axis_ordering(axes)
            print("   ✗ Should have failed but didn't!")
        except ValueError as e:
            print(f"   ✓ Correctly rejected: {str(e)[:50]}...")
        print()


def main():
    """Run all TCZYX dimension demonstrations."""
    print("OME-Zarr TCZYX Dimension Ordering Examples")
    print("=" * 80)
    print(
        "Input images always follow TCZYX dimension order where T, C, Z are optional."
    )
    print("Valid combinations: YX, ZYX, CYX, CZYX, TYX, TZYX, TCYX, TCZYX")
    print("=" * 80)
    print()

    # Demonstrate all valid orderings
    examples = [
        demonstrate_yx_dimensions,
        demonstrate_zyx_dimensions,
        demonstrate_cyx_dimensions,
        demonstrate_czyx_dimensions,
        demonstrate_tyx_dimensions,
        demonstrate_tzyx_dimensions,
        demonstrate_tcyx_dimensions,
        demonstrate_tczyx_dimensions,
    ]

    metadata_results = []
    for example_func in examples:
        result = example_func()
        metadata_results.append(result)

    # Show invalid orderings
    demonstrate_invalid_orderings()

    print("Summary")
    print("=" * 20)
    print(
        f"✓ Successfully created metadata for {len(metadata_results)} valid TCZYX dimension combinations"
    )
    print("✓ All dimension orderings follow the strict TCZYX rule")
    print("✓ Invalid orderings are properly rejected")
    print(
        "\nThe library ensures that all input images follow the TCZYX dimension order,"
    )
    print("providing consistent and predictable metadata structure for OME-Zarr files.")


if __name__ == "__main__":
    main()

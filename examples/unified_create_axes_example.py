#!/usr/bin/env python3
"""
Example demonstrating the unified create_axes function for TCZYX dimension ordering.

This example shows how to use the single create_axes() function instead of
8 separate convenience functions, makin    scales = [1.0, 1.0, 0.3, 0.1, 0.1]  # Time, Channel (no scaling), Z, Y, X
    create_basic_metadata(axes, scales, "5D Live Imaging", channels)  # Create but don't store
    print(f"  Axes: {[ax.name for ax in axes]}")
    print(f"  Channels: {[ch.label for ch in channels]}")
    print("  This is the maximum possible dimensions in TCZYX!") API much cleaner and easier to use.
"""

from ome_zarr_writer.schema_models import (
    OMEZarrImageMetadata,
    OMEMetadata,
    Multiscale,
    Dataset,
    Channel,
    Omero,
    create_axes,  # Single unified function!
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


def demonstrate_unified_api():
    """Demonstrate all TCZYX combinations using the unified create_axes function."""

    print("Unified create_axes() Function Examples")
    print("=" * 80)
    print("Single function replaces 8 separate convenience functions!")
    print("Valid combinations: yx, zyx, cyx, czyx, tyx, tzyx, tcyx, tczyx")
    print("=" * 80)
    print()

    examples = [
        {
            "name": "2D Spatial Image (YX)",
            "axes_call": 'create_axes("yx", 0.1, 0.1)',
            "image_shape": "(512, 512)",
            "use_case": "Grayscale 2D image",
        },
        {
            "name": "3D Spatial Image (ZYX)",
            "axes_call": 'create_axes("zyx", 0.1, 0.1, 0.3)',
            "image_shape": "(20, 256, 256)",
            "use_case": "Confocal Z-stack",
        },
        {
            "name": "Multichannel 2D Image (CYX)",
            "axes_call": 'create_axes("cyx", 0.1, 0.1)',
            "image_shape": "(3, 512, 512)",
            "use_case": "RGB or fluorescence image",
        },
        {
            "name": "Multichannel 3D Image (CZYX)",
            "axes_call": 'create_axes("czyx", 0.1, 0.1, 0.25)',
            "image_shape": "(2, 15, 256, 256)",
            "use_case": "Fluorescence Z-stack",
        },
        {
            "name": "Time-series 2D Image (TYX)",
            "axes_call": 'create_axes("tyx", 0.15, 0.15)',
            "image_shape": "(100, 256, 256)",
            "use_case": "Live cell time-lapse",
        },
        {
            "name": "Time-series 3D Image (TZYX)",
            "axes_call": 'create_axes("tzyx", 0.2, 0.2, 0.5)',
            "image_shape": "(50, 10, 128, 128)",
            "use_case": "4D live imaging",
        },
        {
            "name": "Time-series Multichannel 2D (TCYX)",
            "axes_call": 'create_axes("tcyx", 0.2, 0.2)',
            "image_shape": "(50, 2, 128, 128)",
            "use_case": "Live fluorescence time-lapse",
        },
        {
            "name": "Maximum Dimensions (TCZYX)",
            "axes_call": 'create_axes("tczyx", 0.25, 0.25, 0.5)',
            "image_shape": "(20, 3, 8, 64, 64)",
            "use_case": "5D live cell imaging",
        },
    ]

    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
        print("-" * 50)
        print(f"Function call: {example['axes_call']}")
        print(f"Image shape:   {example['image_shape']}")
        print(f"Use case:      {example['use_case']}")

        # Execute the function call
        if "z" in example["axes_call"]:
            # Extract parameters for Z-containing calls
            if "0.3" in example["axes_call"]:
                axes = create_axes(example["axes_call"].split('"')[1], 0.1, 0.1, 0.3)
            elif "0.25" in example["axes_call"]:
                axes = create_axes(example["axes_call"].split('"')[1], 0.1, 0.1, 0.25)
            elif "0.5" in example["axes_call"]:
                if "tczyx" in example["axes_call"]:
                    axes = create_axes("tczyx", 0.25, 0.25, 0.5)
                else:
                    axes = create_axes("tzyx", 0.2, 0.2, 0.5)
        else:
            axes_type = example["axes_call"].split('"')[1]
            if "0.15" in example["axes_call"]:
                axes = create_axes(axes_type, 0.15, 0.15)
            elif "0.2" in example["axes_call"]:
                axes = create_axes(axes_type, 0.2, 0.2)
            else:
                axes = create_axes(axes_type, 0.1, 0.1)

        axis_names = [ax.name for ax in axes]
        axis_types = [ax.type for ax in axes]

        print(f"Result:        {len(axes)} axes: {axis_names}")
        print(f"Types:         {axis_types}")
        print("Validation:    ✓ Passes TCZYX ordering")

        # Validate
        validate_tczyx_axis_ordering(axes)
        print()


def demonstrate_error_handling():
    """Show error handling for invalid inputs."""
    print("Error Handling Examples")
    print("=" * 40)

    error_cases = [
        {
            "description": "Invalid axis combination",
            "call": 'create_axes("xy", 0.1, 0.1)',
            "expected": "Wrong order (should be YX)",
        },
        {
            "description": "Missing Z size",
            "call": 'create_axes("zyx", 0.1, 0.1)',
            "expected": "z_size required for Z axis",
        },
        {
            "description": "Invalid axis names",
            "call": 'create_axes("abc", 0.1, 0.1)',
            "expected": "Invalid axis names",
        },
    ]

    for i, case in enumerate(error_cases, 1):
        print(f"{i}. {case['description']}")
        print(f"   Call: {case['call']}")
        print(f"   Expected: {case['expected']}")

        try:
            if case["call"] == 'create_axes("xy", 0.1, 0.1)':
                create_axes("xy", 0.1, 0.1)
            elif case["call"] == 'create_axes("zyx", 0.1, 0.1)':
                create_axes("zyx", 0.1, 0.1)
            elif case["call"] == 'create_axes("abc", 0.1, 0.1)':
                create_axes("abc", 0.1, 0.1)
            print("   ✗ Should have failed!")
        except ValueError as e:
            print(f"   ✓ Correctly rejected: {str(e)[:50]}...")
        print()


def demonstrate_practical_usage():
    """Show practical usage examples."""
    print("Practical Usage Examples")
    print("=" * 40)

    print("1. Creating metadata for common scenarios:")
    print()

    # RGB image
    print("RGB Image:")
    axes = create_axes("cyx", 0.1, 0.1, unit="micrometer")
    channels = [
        Channel(label="Red", color="FF0000", active=True),
        Channel(label="Green", color="00FF00", active=True),
        Channel(label="Blue", color="0000FF", active=True),
    ]
    scales = [1.0, 0.1, 0.1]  # Channel (no scaling), Y, X
    create_basic_metadata(axes, scales, "RGB Image", channels)  # Create but don't store
    print(f"  Axes: {[ax.name for ax in axes]}")
    print(f"  Channels: {[ch.label for ch in channels]}")
    print()

    # Time-lapse fluorescence
    print("Time-lapse Fluorescence:")
    axes = create_axes("tcyx", 0.2, 0.2, unit="micrometer")
    channels = [
        Channel(label="DAPI", color="0000FF", active=True),
        Channel(label="GFP", color="00FF00", active=True),
    ]
    scales = [1.0, 1.0, 0.2, 0.2]  # Time, Channel (no scaling), Y, X
    create_basic_metadata(
        axes, scales, "Live Cell Imaging", channels
    )  # Create but don't store
    print(f"  Axes: {[ax.name for ax in axes]}")
    print(f"  Channels: {[ch.label for ch in channels]}")
    print()

    # Full 5D dataset
    print("5D Live Cell Dataset (Maximum):")
    axes = create_axes("tczyx", 0.1, 0.1, 0.3, unit="micrometer")
    channels = [
        Channel(label="Brightfield", color="808080", active=False),
        Channel(label="Calcium", color="00FFFF", active=True),
        Channel(label="Membrane", color="FF00FF", active=True),
    ]
    scales = [1.0, 1.0, 0.3, 0.1, 0.1]  # Time, Channel (no scaling), Z, Y, X
    create_basic_metadata(
        axes, scales, "5D Live Imaging", channels
    )  # Create but don't store
    print(f"  Axes: {[ax.name for ax in axes]}")
    print(f"  Channels: {[ch.label for ch in channels]}")
    print("  This is the maximum possible dimensions in TCZYX!")


def demonstrate_comparison():
    """Compare old vs new API."""
    print("\nAPI Comparison: Old vs New")
    print("=" * 50)

    print("OLD WAY (8 separate functions):")
    print("  create_yx_axes(0.1, 0.1)")
    print("  create_zyx_axes(0.1, 0.1, 0.3)")
    print("  create_cyx_axes(0.1, 0.1)")
    print("  create_czyx_axes(0.1, 0.1, 0.3)")
    print("  create_tyx_axes(0.1, 0.1)")
    print("  create_tzyx_axes(0.1, 0.1, 0.3)")
    print("  create_tcyx_axes(0.1, 0.1)")
    print("  create_tczyx_axes(0.1, 0.1, 0.3)")
    print()

    print("NEW WAY (1 unified function):")
    print('  create_axes("yx", 0.1, 0.1)')
    print('  create_axes("zyx", 0.1, 0.1, 0.3)')
    print('  create_axes("cyx", 0.1, 0.1)')
    print('  create_axes("czyx", 0.1, 0.1, 0.3)')
    print('  create_axes("tyx", 0.1, 0.1)')
    print('  create_axes("tzyx", 0.1, 0.1, 0.3)')
    print('  create_axes("tcyx", 0.1, 0.1)')
    print('  create_axes("tczyx", 0.1, 0.1, 0.3)')
    print()

    print("BENEFITS:")
    print("  ✓ Single function to remember")
    print("  ✓ Clear, descriptive axis specification")
    print("  ✓ Case-insensitive and flexible")
    print("  ✓ Better error messages")
    print("  ✓ Backward compatibility maintained")


def main():
    """Run all demonstrations."""
    print("OME-Zarr Unified create_axes() Function")
    print("=" * 80)
    print("Simplified API for TCZYX dimension ordering")
    print("=" * 80)
    print()

    demonstrate_unified_api()
    demonstrate_error_handling()
    demonstrate_practical_usage()
    demonstrate_comparison()

    print("\nSUMMARY")
    print("=" * 20)
    print("✓ Single create_axes() function replaces 8 separate functions")
    print("✓ Cleaner, more intuitive API")
    print("✓ Maintains strict TCZYX ordering")
    print("✓ Comprehensive error handling")
    print("✓ Full backward compatibility")
    print("\nThe unified API makes OME-Zarr metadata creation much simpler!")


if __name__ == "__main__":
    main()

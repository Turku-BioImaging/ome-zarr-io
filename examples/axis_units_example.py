#!/usr/bin/env python3
"""
Example demonstrating the new axis_units parameter in OmeZarrImage.

This example shows how to:
1. Use a list of Axis objects to specify units
2. Use a dictionary to specify units (simplified approach)
3. Create different types of images with proper axis specifications
"""

import numpy as np
from pathlib import Path
from ome_zarr_writer import OmeZarrImage
from ome_zarr_writer.schema_models import Axis


def example_with_axis_list():
    """Example using a list of Axis objects."""
    print("=== Example 1: Using a list of Axis objects ===")
    
    # Create sample CZYX image data
    channels, z_slices, height, width = 3, 10, 256, 256
    image = np.random.randint(0, 255, size=(channels, z_slices, height, width), dtype=np.uint8)
    
    print(f"Image shape: {image.shape} (C={channels}, Z={z_slices}, Y={height}, X={width})")
    
    # Define output path
    output_path = Path("axis_list_example.zarr")
    
    # Remove existing file if it exists
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
    
    # Create dimension names list
    dims = ["c", "z", "y", "x"]
    
    # Define axis units as a list of Axis objects
    axis_units = [
        Axis(name="c", type="channel"),  # Channels don't have units
        Axis(name="z", type="space", unit="micrometer"),  # Z-axis in micrometers
        Axis(name="y", type="space", unit="micrometer"),  # Y-axis in micrometers
        Axis(name="x", type="space", unit="micrometer")   # X-axis in micrometers
    ]
    
    print("Axis units defined as list:")
    for i, axis in enumerate(axis_units):
        print(f"  {dims[i]}: {axis.type} axis with unit '{axis.unit}'")
    
    # Create OME-Zarr image
    ome_zarr_image = OmeZarrImage(
        path=output_path,
        image=image,
        dims=dims,
        axis_units=axis_units,
        overwrite=True
    )
    
    print(f"✅ Created OME-Zarr with axis list: {output_path}")
    print(f"Stored axes: {[(ax.name, ax.type, ax.unit) for ax in ome_zarr_image.axes]}")


def example_with_axis_dictionary():
    """Example using a dictionary for axis units."""
    print("\n=== Example 2: Using a dictionary for axis units ===")
    
    # Create sample TCZYX image data (time-series)
    time_points, channels, z_slices, height, width = 5, 2, 8, 128, 128
    image = np.random.randint(0, 255, size=(time_points, channels, z_slices, height, width), dtype=np.uint8)
    
    print(f"Image shape: {image.shape} (T={time_points}, C={channels}, Z={z_slices}, Y={height}, X={width})")
    
    # Define output path
    output_path = Path("axis_dict_example.zarr")
    
    # Remove existing file if it exists
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
    
    # Create dimension names list
    dims = ["t", "c", "z", "y", "x"]
    
    # Define axis units as a dictionary (simpler approach)
    axis_units = {
        "unit": "nanometer",     # Unit for spatial dimensions (z, y, x)
        "time_unit": "second"    # Unit for time dimension
    }
    
    print("Axis units defined as dictionary:")
    print(f"  Spatial unit: {axis_units['unit']}")
    print(f"  Time unit: {axis_units['time_unit']}")
    
    # Create OME-Zarr image
    ome_zarr_image = OmeZarrImage(
        path=output_path,
        image=image,
        dims=dims,
        axis_units=axis_units,
        overwrite=True
    )
    
    print(f"✅ Created OME-Zarr with axis dictionary: {output_path}")
    print(f"Stored axes: {[(ax.name, ax.type, ax.unit) for ax in ome_zarr_image.axes]}")


def example_with_minimal_dictionary():
    """Example using minimal dictionary (defaults)."""
    print("\n=== Example 3: Using minimal dictionary (defaults) ===")
    
    # Create simple 2D image
    height, width = 512, 512
    image = np.random.randint(0, 255, size=(height, width), dtype=np.uint8)
    
    print(f"Image shape: {image.shape} (Y={height}, X={width})")
    
    # Define output path
    output_path = Path("axis_minimal_example.zarr")
    
    # Remove existing file if it exists
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
    
    # Create dimension names list
    dims = ["y", "x"]
    
    # Define axis units as an empty dictionary (will use defaults)
    axis_units = {}  # Will default to micrometer for spatial dimensions
    
    print("Using empty dictionary (defaults):")
    print("  Default spatial unit: micrometer")
    
    # Create OME-Zarr image
    ome_zarr_image = OmeZarrImage(
        path=output_path,
        image=image,
        dims=dims,
        axis_units=axis_units,
        overwrite=True
    )
    
    print(f"✅ Created OME-Zarr with default units: {output_path}")
    print(f"Stored axes: {[(ax.name, ax.type, ax.unit) for ax in ome_zarr_image.axes]}")


def example_with_custom_units():
    """Example with custom units in different scales."""
    print("\n=== Example 4: Using custom units in different scales ===")
    
    # Create 3D image (ZYX)
    z_slices, height, width = 20, 64, 64
    image = np.random.randint(0, 255, size=(z_slices, height, width), dtype=np.uint8)
    
    print(f"Image shape: {image.shape} (Z={z_slices}, Y={height}, X={width})")
    
    # Define output path
    output_path = Path("axis_custom_example.zarr")
    
    # Remove existing file if it exists
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
    
    # Create dimension names list
    dims = ["z", "y", "x"]
    
    # Create axis units with different scales for each dimension
    axis_units = [
        Axis(name="z", type="space", unit="millimeter"),  # Z-axis in millimeters (larger scale)
        Axis(name="y", type="space", unit="micrometer"),  # Y-axis in micrometers
        Axis(name="x", type="space", unit="nanometer")    # X-axis in nanometers (smaller scale)
    ]
    
    print("Custom axis units with different scales:")
    for i, axis in enumerate(axis_units):
        print(f"  {dims[i]}: {axis.type} axis with unit '{axis.unit}'")
    
    # Create OME-Zarr image
    ome_zarr_image = OmeZarrImage(
        path=output_path,
        image=image,
        dims=dims,
        axis_units=axis_units,
        overwrite=True
    )
    
    print(f"✅ Created OME-Zarr with custom units: {output_path}")
    print(f"Stored axes: {[(ax.name, ax.type, ax.unit) for ax in ome_zarr_image.axes]}")


def main():
    """Run all examples."""
    print("OME-Zarr Writer - axis_units Parameter Examples")
    print("=" * 50)
    
    try:
        example_with_axis_list()
        example_with_axis_dictionary()
        example_with_minimal_dictionary()
        example_with_custom_units()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        print("\nThe axis_units parameter provides flexible ways to specify units:")
        print("1. List of Axis objects - Full control over each axis")
        print("2. Dictionary - Convenient for common cases")
        print("3. Empty dictionary - Uses sensible defaults")
        print("4. Mixed units - Different scales for different dimensions")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()

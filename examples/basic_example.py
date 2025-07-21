#!/usr/bin/env python3
"""
Basic example of using ome-zarr-writer to create an OME-Zarr file.

This example demonstrates how to:
1. Create sample CZYX image data (Channel, Z, Y, X)
2. Initialize the OME-Zarr image with proper axis units
3. Write a complete OME-Zarr file with multiscale pyramids
"""

import numpy as np
from pathlib import Path
from ome_zarr_writer import OmeZarrImage


def main():
    """Create a basic OME-Zarr file with CZYX dimensions."""
    print("Creating sample OME-Zarr file with CZYX format...")

    # Create sample CZYX image data
    # Shape: (channels, z_slices, height, width)
    # CZYX is the standard format for OME-Zarr multichannel 3D images
    channels, z_slices, height, width = 3, 10, 512, 512
    image = np.random.randint(
        0, 255, size=(channels, z_slices, height, width), dtype=np.uint8
    )

    print(
        f"Image shape: {image.shape} (C={channels}, Z={z_slices}, Y={height}, X={width})"
    )

    # Define output path
    output_path = Path("example-czyx.zarr")

    # Define dimension names for CZYX layout
    dims = ["c", "z", "y", "x"]

    # Define axis units - spatial dimensions get micrometers, channel has no unit
    axis_units = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}

    # For more advanced usage with specific pixel sizes and coordinate transformations:
    # from ome_zarr_writer.schema_models import ScaleTransformation
    # axis_units = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}
    # scale_transformations = {"z": 0.25, "y": 0.1, "x": 0.1}  # ZYX pixel sizes

    # Initialize OME-Zarr image with multiscale support
    ome_zarr_image = OmeZarrImage(
        path=output_path,
        image=image,
        dims=dims,
        axis_units=axis_units,
        downscale_levels=3,  # Create 3 additional downscale levels
        overwrite=True,
    )

    print(f"Successfully initialized OME-Zarr image: {output_path}")
    print(f"Dimensions: {dims}")
    print(f"Downscale levels: {ome_zarr_image.downscale_levels}")
    print("Pixel size: 1.0 unit (default)")

    # Write the OME-Zarr file
    print("\nWriting OME-Zarr file...")
    ome_zarr_image.write()
    print(f"✓ OME-Zarr file written successfully to: {output_path}")
    
    # Verify the output
    if output_path.exists():
        print(f"✓ File exists at: {output_path}")
        print(f"✓ File size: {sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / (1024*1024):.2f} MB")
    else:
        print("✗ Error: File was not created")


if __name__ == "__main__":
    main()

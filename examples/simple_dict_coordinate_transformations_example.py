#!/usr/bin/env python3
"""
Simple example demonstrating the new dictionary-based scale transformations.

This example shows how easy it is to specify pixel sizes without needing to import
and create transformation objects.
"""

import numpy as np
from pathlib import Path
from ome_zarr_writer import OmeZarrImage


def main():
    """Demonstrate simple scale transformations using dictionary format."""
    print("Creating OME-Zarr with dictionary-based scale transformations...")

    # Create sample image data - a simple 3D image (ZYX)
    z_slices, height, width = 10, 256, 256
    image = np.random.randint(0, 255, size=(z_slices, height, width), dtype=np.uint8)

    print(f"Image shape: {image.shape} (Z={z_slices}, Y={height}, X={width})")

    # Define output path
    output_path = Path("simple_dict_example.zarr")

    # Easy way to specify pixel sizes - no imports needed!
    scale_transformations = {
        "z": 0.25,  # 0.25 micrometers per z-slice
        "y": 0.1,   # 0.1 micrometers per pixel in Y
        "x": 0.1    # 0.1 micrometers per pixel in X
    }

    print(f"Pixel sizes: {scale_transformations}")

    # Create and write the OME-Zarr image
    ome_zarr_image = OmeZarrImage(
        path=output_path,
        image=image,
        dims=["z", "y", "x"],
        axis_units={"z": "micrometer", "y": "micrometer", "x": "micrometer"},
        scale_transformations=scale_transformations,
        downscale_levels=3,
        overwrite=True,
    )

    ome_zarr_image.write()
    print(f"\n✅ Done! OME-Zarr saved to: {output_path}")
    print("\nNote: No need to import ScaleTransformation anymore!")


if __name__ == "__main__":
    main()

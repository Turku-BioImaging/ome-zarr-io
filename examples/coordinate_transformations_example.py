#!/usr/bin/env python3
"""
Example demonstrating scale transformations with multiscale OME-Zarr images.

This example shows how to:
1. Define scale transformations using a simple dictionary format
2. Automatically calculate scale transformations for downscale levels
3. Use physical pixel/voxel sizes for proper spatial metadata
"""

import numpy as np
from pathlib import Path
from ome_zarr_writer import OmeZarrImage


def main():
    """Demonstrate scale transformations with multiscale OME-Zarr."""
    print("Creating multiscale OME-Zarr with scale transformations...")

    # Create sample CZYX image data
    channels, z_slices, height, width = 2, 5, 256, 256
    image = np.random.randint(
        0, 255, size=(channels, z_slices, height, width), dtype=np.uint8
    )

    print(
        f"Original image shape: {image.shape} (C={channels}, Z={z_slices}, Y={height}, X={width})"
    )

    # Define output path
    output_path = Path("coordinate_transforms_example.zarr")

    # Remove existing file if it exists
    if output_path.exists():
        import shutil

        shutil.rmtree(output_path)

    # Create dimension names list for CZYX
    dims = ["c", "z", "y", "x"]

    # Define axis units using dictionary approach
    axis_units = {
        "z": "micrometer",
        "y": "micrometer", 
        "x": "micrometer"
    }

    # Define scale transformations for the original image
    # These represent the pixel/voxel sizes for the original resolution
    scale_transformations = {
        "z": 0.25,  # 0.25 μm z spacing 
        "y": 0.1,   # 0.1 μm y pixel size
        "x": 0.1    # 0.1 μm x pixel size
    }

    print("Scale transformations dictionary:")
    for dim, scale in scale_transformations.items():
        print(f"  {dim}: {scale} μm")

    # Test with multiple downscale levels
    downscale_levels = 3
    downscale_factor = 2

    try:
        # Initialize OME-Zarr image with coordinate transformations
        ome_zarr_image = OmeZarrImage(
            path=output_path,
            image=image,
            dims=dims,
            axis_units=axis_units,
            scale_transformations=scale_transformations,
            downscale_levels=downscale_levels,
            downscale_factor=downscale_factor,
            overwrite=True,
        )

        print(f"\nSuccessfully initialized OME-Zarr image: {output_path}")
        print(f"Dimensions: {dims}")
        print(f"Downscale levels: {downscale_levels}")
        print(f"Downscale factor: {downscale_factor}")

        # Import here just for analysis
        from ome_zarr_writer.schema_models import ScaleTransformation

        # Create and display the scale transformations for each level
        print("\nCalculating scale transformations for each level...")
        level_transformations = (
            ome_zarr_image._create_coordinate_transformations_for_levels()
        )

        print(f"Number of transformation sets: {len(level_transformations)}")

        for level, transformations in enumerate(level_transformations):
            scale_factor_actual = downscale_factor**level
            print(f"\nLevel {level} (scale factor: {scale_factor_actual}x):")

            for i, transform in enumerate(transformations):
                if isinstance(transform, ScaleTransformation):
                    print(f"  {i+1}. Scale: {transform.scale}")

        # Create and display the downscaled arrays
        print("\nCreating multiscale pyramid...")
        downscaled_arrays = ome_zarr_image._create_downscaled_arrays()

        print(f"Number of resolution levels: {len(downscaled_arrays)}")
        for i, arr in enumerate(downscaled_arrays):
            scale_factor_actual = downscale_factor**i
            print(f"  Level {i}: {arr.shape} (scale factor: {scale_factor_actual}x)")

        print(
            "\n✅ Multiscale pyramid with scale transformations created successfully!"
        )
        print(
            "Note: Scale transformations adjust spatial dimensions (Y, X) for each level."
        )
        print("      Translation transformations remain constant across levels.")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()

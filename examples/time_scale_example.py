#!/usr/bin/env python3
"""
Example demonstrating scale transformations with time axis.

This example shows how to include temporal scale information 
(frame intervals) along with spatial pixel sizes.
"""

import numpy as np
from pathlib import Path
from ome_zarr_writer import OmeZarrImage


def main():
    """Demonstrate time axis scale transformations."""
    print("Creating OME-Zarr with time and spatial scale transformations...")

    # Create a 4D time-lapse image (T, Z, Y, X)
    image = np.random.randint(0, 255, size=(10, 8, 128, 128), dtype=np.uint16)
    dims = ["t", "z", "y", "x"]

    # Define axis units
    axis_units = {
        "t": "second",
        "z": "micrometer",
        "y": "micrometer", 
        "x": "micrometer"
    }

    # Define scale transformations for all dimensions
    # Units are determined by axis_units above
    scale_transformations = {
        "t": 0.5,    # 0.5 second frame interval
        "z": 0.25,   # 0.25 μm z-step size
        "y": 0.065,  # 0.065 μm pixel size in Y
        "x": 0.065   # 0.065 μm pixel size in X
    }

    print(f"Image shape: {image.shape} (T={image.shape[0]}, Z={image.shape[1]}, Y={image.shape[2]}, X={image.shape[3]})")
    print("Scale transformations:")
    for dim, scale in scale_transformations.items():
        unit = "second" if dim == "t" else "μm"
        print(f"  {dim}: {scale} {unit}")

    # Create OME-Zarr with multiscale and write
    output_path = Path("time_scale_example.zarr")
    
    ome_zarr_image = OmeZarrImage(
        path=output_path,
        image=image,
        dims=dims,
        axis_units=axis_units,
        scale_transformations=scale_transformations,
        downscale_levels=2,  # Create 2 additional downscale levels
        downscale_factor=2,
        overwrite=True,
    )
    
    ome_zarr_image.write()

    print(f"✅ Successfully wrote OME-Zarr to: {output_path}")
    print(f"   - 3 resolution levels (original + 2 downscaled)")
    print(f"   - Shape: {image.shape}")
    print(f"   - Dimensions: {dims}")
    print(f"   - Time points: {image.shape[0]} (every 0.5 seconds)")
    print(f"   - Z slices: {image.shape[1]} (0.25 μm spacing)")
    print(f"   - Pixel size: 0.065 × 0.065 μm")

    print("\n✅ Done! Time-lapse OME-Zarr with proper temporal and spatial metadata created.")
    print("\nNote: The time scale transformation represents frame intervals,")
    print("      while spatial transformations represent pixel/voxel sizes.")


if __name__ == "__main__":
    main()

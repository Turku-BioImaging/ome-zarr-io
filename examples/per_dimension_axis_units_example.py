#!/usr/bin/env python3
"""
Example demonstrating per-dimension axis units specification.

This example shows how to specify units for each dimension individually,
providing maximum flexibility and clarity.
"""

import numpy as np
from pathlib import Path
from ome_zarr_writer import OmeZarrImage


def main():
    """Demonstrate per-dimension axis units specification."""
    print("🧬 Per-Dimension Axis Units Example")
    print("=" * 50)

    # Example 1: Basic CZYX with per-dimension units
    print("\n1. Basic CZYX with per-dimension units")
    image_czyx = np.random.randint(0, 255, size=(3, 20, 256, 256), dtype=np.uint16)
    dims_czyx = ["c", "z", "y", "x"]
    
    # Specify units for each spatial dimension
    axis_units_czyx = {
        "z": "micrometer",     # Z-step size in micrometers
        "y": "micrometer",     # Y pixel size in micrometers  
        "x": "micrometer"      # X pixel size in micrometers
        # Note: 'c' (channel) dimension doesn't need a unit - handled automatically
    }
    
    scale_transformations_czyx = {
        "z": 0.5,   # 0.5 μm z-step
        "y": 0.1,   # 0.1 μm pixel size
        "x": 0.1    # 0.1 μm pixel size
    }
    
    ome_zarr_czyx = OmeZarrImage(
        path="per_dimension_czyx.zarr",
        image=image_czyx,
        dims=dims_czyx,
        axis_units=axis_units_czyx,
        scale_transformations=scale_transformations_czyx,
        overwrite=True
    )
    
    ome_zarr_czyx.write()
    print(f"   ✅ Created per_dimension_czyx.zarr")
    print(f"   - Shape: {image_czyx.shape} (C={image_czyx.shape[0]}, Z={image_czyx.shape[1]}, Y={image_czyx.shape[2]}, X={image_czyx.shape[3]})")
    print(f"   - Axes: {[(ax.name, ax.type, ax.unit) for ax in ome_zarr_czyx.axes]}")

    # Example 2: Time-lapse with mixed units
    print("\n2. Time-lapse with mixed spatial units")
    image_tzyx = np.random.randint(0, 255, size=(10, 8, 128, 128), dtype=np.uint16)
    dims_tzyx = ["t", "z", "y", "x"]
    
    # Different units for different dimensions
    axis_units_tzyx = {
        "t": "second",         # Time in seconds
        "z": "nanometer",      # High-resolution Z in nanometers  
        "y": "micrometer",     # Y in micrometers
        "x": "micrometer"      # X in micrometers
    }
    
    scale_transformations_tzyx = {
        "t": 0.1,      # 100 ms frame interval
        "z": 50.0,     # 50 nm z-step
        "y": 0.032,    # 32 nm pixel size
        "x": 0.032     # 32 nm pixel size
    }
    
    ome_zarr_tzyx = OmeZarrImage(
        path="per_dimension_tzyx.zarr",
        image=image_tzyx,
        dims=dims_tzyx,
        axis_units=axis_units_tzyx,
        scale_transformations=scale_transformations_tzyx,
        downscale_levels=2,
        overwrite=True
    )
    
    ome_zarr_tzyx.write()
    print(f"   ✅ Created per_dimension_tzyx.zarr")
    print(f"   - Shape: {image_tzyx.shape} (T={image_tzyx.shape[0]}, Z={image_tzyx.shape[1]}, Y={image_tzyx.shape[2]}, X={image_tzyx.shape[3]})")
    print(f"   - Axes: {[(ax.name, ax.type, ax.unit) for ax in ome_zarr_tzyx.axes]}")

    # Example 3: Full 5D with per-dimension units
    print("\n3. Full 5D (TCZYX) with per-dimension units")
    image_tczyx = np.random.randint(0, 255, size=(5, 2, 10, 64, 64), dtype=np.uint8)
    dims_tczyx = ["t", "c", "z", "y", "x"]
    
    axis_units_tczyx = {
        "t": "millisecond",    # High temporal resolution
        "z": "micrometer",     # Z in micrometers
        "y": "micrometer",     # Y in micrometers
        "x": "micrometer"      # X in micrometers
        # 'c' (channel) handled automatically
    }
    
    scale_transformations_tczyx = {
        "t": 50,       # 50 ms frame interval
        "z": 0.2,      # 0.2 μm z-step
        "y": 0.065,    # 65 nm pixel size
        "x": 0.065     # 65 nm pixel size
    }
    
    ome_zarr_tczyx = OmeZarrImage(
        path="per_dimension_tczyx.zarr",
        image=image_tczyx,
        dims=dims_tczyx,
        axis_units=axis_units_tczyx,
        scale_transformations=scale_transformations_tczyx,
        downscale_levels=1,
        overwrite=True
    )
    
    ome_zarr_tczyx.write()
    print(f"   ✅ Created per_dimension_tczyx.zarr")
    print(f"   - Shape: {image_tczyx.shape} (T={image_tczyx.shape[0]}, C={image_tczyx.shape[1]}, Z={image_tczyx.shape[2]}, Y={image_tczyx.shape[3]}, X={image_tczyx.shape[4]})")
    print(f"   - Axes: {[(ax.name, ax.type, ax.unit) for ax in ome_zarr_tczyx.axes]}")

    print("\n" + "=" * 50)
    print("✅ All examples completed successfully!")
    print("\nKey benefits of per-dimension axis units:")
    print("• Explicit and clear - each dimension has its own unit")
    print("• Flexible - mix different units for different dimensions")
    print("• Automatic channel handling - 'c' dimension needs no unit")
    print("• Error prevention - missing units for required dimensions cause clear errors")


if __name__ == "__main__":
    main()

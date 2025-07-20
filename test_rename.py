#!/usr/bin/env python3
"""Test script to verify the class rename worked correctly."""

from ome_zarr_writer import OmeZarrImage
import numpy as np

print("✅ Import successful!")
print(f"Class name: {OmeZarrImage.__name__}")

# Test basic functionality
image = np.random.randint(0, 255, size=(2, 3, 32, 32), dtype=np.uint8)
ome_zarr = OmeZarrImage('test.zarr', image, ['c', 'z', 'y', 'x'], downscale_levels=1, overwrite=True)
arrays = ome_zarr._create_downscaled_arrays()
print(f"Created {len(arrays)} downscale levels")
print("✅ Functionality working correctly!")

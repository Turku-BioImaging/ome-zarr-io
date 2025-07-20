#!/usr/bin/env python3
"""Test the warning and validation system."""

import sys
import os
import warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from ome_zarr_writer.image import OmeZarrImage

print("Testing validation and warnings...")

# Test 1: Too many levels
print("\n1. Testing too many levels (should warn and adjust):")
small_image = np.random.randint(0, 255, size=(1, 1, 8, 8), dtype=np.uint8)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    
    ome_zarr = OmeZarrImage(
        path="test.zarr",
        image=small_image,
        dims=["c", "z", "y", "x"],
        downscale_levels=10,  # Too many for 8x8 image
        downscale_factor=2,
        overwrite=True
    )
    
    if w:
        print(f"Warning: {w[0].message}")
    
    print("Original levels requested: 10")
    print(f"Adjusted levels: {ome_zarr.downscale_levels}")

# Test 2: Invalid downscale factor
print("\n2. Testing invalid downscale_factor (should raise error):")
try:
    ome_zarr = OmeZarrImage(
        path="test.zarr",
        image=small_image,
        dims=["c", "z", "y", "x"],
        downscale_levels=2,
        downscale_factor=1,  # Invalid
        overwrite=True
    )
except ValueError as e:
    print(f"Expected error: {e}")

# Test 3: Different factors
print("\n3. Testing different downscale factors:")
image = np.random.randint(0, 255, size=(1, 1, 64, 64), dtype=np.uint8)

for factor in [2, 3, 4]:
    ome_zarr = OmeZarrImage(
        path="test.zarr",
        image=image,
        dims=["c", "z", "y", "x"],
        downscale_levels=3,
        downscale_factor=factor,
        overwrite=True
    )
    
    arrays = ome_zarr._create_downscaled_arrays()
    print(f"Factor {factor}: Created {len(arrays)} levels")
    for i, arr in enumerate(arrays):
        if i > 0:
            scale_y = arrays[i-1].shape[-2] / arr.shape[-2]
            scale_x = arrays[i-1].shape[-1] / arr.shape[-1]
            print(f"  Level {i}: {arr.shape} (scale: {scale_y:.1f}x{scale_x:.1f})")
        else:
            print(f"  Level {i}: {arr.shape} (original)")

print("\n✅ All validation tests completed!")

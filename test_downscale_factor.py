#!/usr/bin/env python3
"""
Test script to verify the configurable downscale_factor functionality.
"""

import numpy as np
import warnings
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from ome_zarr_writer import OmeZarrImage


def test_downscale_factor():
    """Test different downscale factors."""
    print("Testing configurable downscale_factor...")
    
    # Create a test CZYX image
    channels, z_slices, height, width = 2, 3, 128, 128
    test_image = np.random.randint(0, 255, size=(channels, z_slices, height, width), dtype=np.uint8)
    
    print(f"Original image shape: {test_image.shape}")
    
    # Test different downscale factors
    for factor in [2, 3, 4]:
        print(f"\nTesting with downscale_factor={factor}:")
        
        ome_zarr = OmeZarrImage(
            path="test.zarr",
            image=test_image,
            dims=["c", "z", "y", "x"],
            downscale_levels=3,
            downscale_factor=factor,
            overwrite=True
        )
        
        arrays = ome_zarr._create_downscaled_arrays()
        
        print(f"  Created {len(arrays)} levels:")
        for i, arr in enumerate(arrays):
            if i > 0:
                prev_shape = arrays[i-1].shape
                curr_shape = arr.shape
                actual_factor_y = prev_shape[-2] // curr_shape[-2]
                actual_factor_x = prev_shape[-1] // curr_shape[-1]
                print(f"    Level {i}: {curr_shape} (factors: Y={actual_factor_y}, X={actual_factor_x})")
            else:
                print(f"    Level {i}: {arr.shape} (original)")


def test_validation_warnings():
    """Test validation and warning system."""
    print("\nTesting validation and warnings...")
    
    # Create a small test image that will trigger warnings
    small_image = np.random.randint(0, 255, size=(1, 1, 16, 16), dtype=np.uint8)
    
    print(f"Small image shape: {small_image.shape}")
    
    # Test 1: Too many levels with default factor
    print("\n1. Testing too many levels with default factor (should warn):")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ome_zarr = OmeZarrImage(
            path="test.zarr",
            image=small_image,
            dims=["c", "z", "y", "x"],
            downscale_levels=10,  # Too many for 16x16 image
            overwrite=True
        )
        if w:
            print(f"    Warning: {w[0].message}")
        print(f"    Adjusted downscale_levels: {ome_zarr.downscale_levels}")
    
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
        print(f"    Expected error: {e}")
    
    # Test 3: High factor with many levels (should warn and suggest smaller factor)
    print("\n3. Testing high factor with many levels (should suggest smaller factor):")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ome_zarr = OmeZarrImage(
            path="test.zarr",
            image=small_image,
            dims=["c", "z", "y", "x"],
            downscale_levels=5,
            downscale_factor=4,  # Too high for this many levels
            overwrite=True
        )
        if w:
            print(f"    Warning: {w[0].message}")
        print(f"    Adjusted downscale_levels: {ome_zarr.downscale_levels}")


def test_actual_downscaling():
    """Test that the actual downscaling works with different factors."""
    print("\nTesting actual downscaling with different factors...")
    
    image = np.random.randint(0, 255, size=(1, 1, 64, 64), dtype=np.uint8)
    
    for factor in [2, 4]:
        print(f"\nFactor {factor}:")
        ome_zarr = OmeZarrImage(
            path="test.zarr",
            image=image,
            dims=["c", "z", "y", "x"],
            downscale_levels=2,
            downscale_factor=factor,
            overwrite=True
        )
        
        arrays = ome_zarr._create_downscaled_arrays()
        
        for i, arr in enumerate(arrays):
            print(f"    Level {i}: {arr.shape}")
            if i > 0:
                # Verify that data is computed correctly
                sample = arr[0, 0, :2, :2].compute()
                print(f"        Sample data range: {sample.min():.1f}-{sample.max():.1f}")


if __name__ == "__main__":
    test_downscale_factor()
    test_validation_warnings()
    test_actual_downscaling()
    print("\n✅ All tests completed!")

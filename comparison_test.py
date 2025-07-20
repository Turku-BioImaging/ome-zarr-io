#!/usr/bin/env python3
"""
Comparison between the old coarsening method and the new Gaussian filtering + rescaling method.
"""

import numpy as np
import dask.array as da
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from ome_zarr_writer import OMEZarrImage


def create_test_pattern():
    """Create a test pattern that will show aliasing artifacts."""
    size = 128
    # Create a high-frequency pattern that will show aliasing when downscaled
    y, x = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    
    # Create a pattern with both low and high frequency components
    pattern = (
        100 * np.sin(2 * np.pi * x / 8) * np.sin(2 * np.pi * y / 8) +  # Low frequency
        50 * np.sin(2 * np.pi * x / 2) * np.sin(2 * np.pi * y / 2) +   # High frequency
        128
    )
    
    return np.clip(pattern, 0, 255).astype(np.uint8)


def old_coarsening_method(image_array, downscale_levels):
    """Old method using da.coarsen for comparison."""
    arrays = [image_array]
    current_array = image_array
    
    for level in range(1, downscale_levels + 1):
        if current_array.shape[-2] < 2 or current_array.shape[-1] < 2:
            break
            
        downscale_factors = [1] * current_array.ndim
        downscale_factors[-2] = 2  # Y axis
        downscale_factors[-1] = 2  # X axis
        
        try:
            current_array = da.coarsen(
                np.mean,
                current_array,
                {i: factor for i, factor in enumerate(downscale_factors)},
                trim_excess=True
            )
            arrays.append(current_array)
        except ValueError:
            break
    
    return arrays


def compare_methods():
    """Compare the old and new downscaling methods."""
    print("Comparing downscaling methods...")
    
    # Create test pattern
    test_pattern = create_test_pattern()
    print(f"Created test pattern with shape: {test_pattern.shape}")
    
    # Add channel and z dimensions to make it CZYX
    test_image = test_pattern[np.newaxis, np.newaxis, :, :]  # Shape: (1, 1, 128, 128)
    
    # Test with old method (coarsening)
    print("\nOld method (coarsening):")
    old_arrays = old_coarsening_method(da.from_array(test_image), 2)
    for i, arr in enumerate(old_arrays):
        print(f"  Level {i}: {arr.shape}")
        # Show some statistics to see the difference
        data = arr[0, 0].compute()
        print(f"    Mean: {data.mean():.1f}, Std: {data.std():.1f}, Range: {data.min():.1f}-{data.max():.1f}")
    
    # Test with new method (Gaussian + rescaling)
    print("\nNew method (Gaussian + rescaling):")
    ome_zarr = OMEZarrImage(
        path="test.zarr",
        image=test_image,
        dims=["c", "z", "y", "x"],
        downscale_levels=2,
        overwrite=True
    )
    new_arrays = ome_zarr._create_downscaled_arrays()
    for i, arr in enumerate(new_arrays):
        print(f"  Level {i}: {arr.shape}")
        # Show some statistics to see the difference
        data = arr[0, 0].compute()
        print(f"    Mean: {data.mean():.1f}, Std: {data.std():.1f}, Range: {data.min():.1f}-{data.max():.1f}")
    
    # Test timing (rough comparison)
    import time
    
    print("\nTiming comparison (rough):")
    
    # Time old method
    start = time.time()
    old_result = old_coarsening_method(da.from_array(test_image), 3)
    _ = [arr.compute() for arr in old_result]  # Force computation
    old_time = time.time() - start
    print(f"Old method: {old_time:.3f} seconds")
    
    # Time new method
    start = time.time()
    new_result = ome_zarr._create_downscaled_arrays()
    _ = [arr.compute() for arr in new_result]  # Force computation
    new_time = time.time() - start
    print(f"New method: {new_time:.3f} seconds")
    
    print(f"\nSpeedup factor: {old_time/new_time:.2f}x {'(new is faster)' if new_time < old_time else '(old is faster)'}")


if __name__ == "__main__":
    compare_methods()
    print("\n✅ Comparison completed!")

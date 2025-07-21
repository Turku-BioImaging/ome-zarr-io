#!/usr/bin/env python3
"""Test script to verify float downscale_factor functionality."""

import numpy as np
import sys
import os

# Add src to path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ome_zarr_writer.image import OmeZarrImage

def test_float_downscale_factor():
    """Test that float downscale_factor values work correctly."""
    print("Testing float downscale_factor...")
    
    # Create a small test image
    test_image = np.random.rand(2, 2, 50, 50).astype(np.float32)
    
    # Test with float downscale_factor = 1.5
    print("Testing downscale_factor=1.5...")
    try:
        writer = OmeZarrImage(
            path='/tmp/test_float_factor_1_5.zarr',
            image=test_image,
            dims=['c', 'z', 'y', 'x'],
            axis_units={'z': 'micrometer', 'y': 'micrometer', 'x': 'micrometer'},
            downscale_factor=1.5,
            downscale_levels=2
        )
        print('✅ Float downscale_factor 1.5 accepted')
    except Exception as e:
        print(f'❌ Error with factor=1.5: {e}')
        return False
    
    # Test with float downscale_factor = 2.5
    print("Testing downscale_factor=2.5...")
    try:
        writer = OmeZarrImage(
            path='/tmp/test_float_factor_2_5.zarr',
            image=test_image,
            dims=['c', 'z', 'y', 'x'],
            axis_units={'z': 'micrometer', 'y': 'micrometer', 'x': 'micrometer'},
            downscale_factor=2.5,
            downscale_levels=2
        )
        print('✅ Float downscale_factor 2.5 accepted')
    except Exception as e:
        print(f'❌ Error with factor=2.5: {e}')
        return False

    # Test validation: factor must be > 1.0
    print("Testing validation for factor=1.0...")
    try:
        writer = OmeZarrImage(
            path='/tmp/test_invalid_factor.zarr',
            image=test_image,
            dims=['c', 'z', 'y', 'x'],
            axis_units={'z': 'micrometer', 'y': 'micrometer', 'x': 'micrometer'},
            downscale_factor=1.0
        )
        print('❌ Should have failed with factor=1.0')
        return False
    except ValueError as e:
        print(f'✅ Correctly rejected factor=1.0: {e}')
    
    # Test validation: factor must be > 1.0 
    print("Testing validation for factor=0.5...")
    try:
        writer = OmeZarrImage(
            path='/tmp/test_invalid_factor2.zarr',
            image=test_image,
            dims=['c', 'z', 'y', 'x'],
            axis_units={'z': 'micrometer', 'y': 'micrometer', 'x': 'micrometer'},
            downscale_factor=0.5
        )
        print('❌ Should have failed with factor=0.5')
        return False
    except ValueError as e:
        print(f'✅ Correctly rejected factor=0.5: {e}')
    
    print('✅ All float downscale_factor tests passed!')
    return True

if __name__ == "__main__":
    success = test_float_downscale_factor()
    sys.exit(0 if success else 1)

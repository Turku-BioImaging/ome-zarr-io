#!/usr/bin/env python3
"""Simple test to debug the downscale_factor implementation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import numpy as np
    print("✅ NumPy imported")
    
    from ome_zarr_writer.image import OmeZarrImage
    print("✅ OmeZarrImage imported")
    
    # Test basic initialization
    image = np.random.randint(0, 255, size=(1, 1, 32, 32), dtype=np.uint8)
    print(f"✅ Created test image: {image.shape}")
    
    # Test with default factor
    ome_zarr = OmeZarrImage(
        path="test.zarr",
        image=image,
        dims=["c", "z", "y", "x"],
        downscale_levels=2,
        overwrite=True
    )
    print(f"✅ Created OmeZarrImage with default factor: {ome_zarr.downscale_factor}")
    
    # Test with custom factor
    ome_zarr2 = OmeZarrImage(
        path="test.zarr",
        image=image,
        dims=["c", "z", "y", "x"],
        downscale_levels=2,
        downscale_factor=3,
        overwrite=True
    )
    print(f"✅ Created OmeZarrImage with custom factor: {ome_zarr2.downscale_factor}")
    
    # Test downscaling
    arrays = ome_zarr2._create_downscaled_arrays()
    print(f"✅ Created {len(arrays)} downscaled arrays")
    for i, arr in enumerate(arrays):
        print(f"    Level {i}: {arr.shape}")
    
    print("✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

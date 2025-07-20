#!/usr/bin/env python3
"""
Basic example of using ome-zarr-writer to create an OME-Zarr file.

This example demonstrates how to:
1. Create sample CZYX image data (Channel, Z, Y, X)
2. Initialize the OME-Zarr image
3. Create proper metadata using the schema dataclasses
"""

import numpy as np
from pathlib import Path
from ome_zarr_writer import OmeZarrImage, create_axes


def main():
    """Create a basic OME-Zarr file with CZYX dimensions."""
    print("Creating sample OME-Zarr file with CZYX format...")
    
    # Create sample CZYX image data
    # Shape: (channels, z_slices, height, width)
    # CZYX is the standard format for OME-Zarr multichannel 3D images
    channels, z_slices, height, width = 3, 10, 512, 512
    image = np.random.randint(0, 255, size=(channels, z_slices, height, width), dtype=np.uint8)
    
    print(f"Image shape: {image.shape} (C={channels}, Z={z_slices}, Y={height}, X={width})")
    
    # Define output path
    output_path = Path("example-czyx.zarr")
    
    # Remove existing file if it exists
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
    
    # Create axes for CZYX layout
    # Pixel sizes: X=0.1μm, Y=0.1μm, Z=0.25μm
    axes = create_axes("czyx", 0.1, 0.1, 0.25, unit="micrometer")
    
    # Create dimension names list
    dims = ["c", "z", "y", "x"]
    
    try:
        # Initialize OME-Zarr image
        ome_zarr_image = OmeZarrImage(
            path=output_path,
            image=image,
            dims=dims,
            overwrite=True
        )
        
        print(f"Successfully initialized OME-Zarr image: {output_path}")
        print(f"Dimensions: {dims}")
        print(f"Axes created: {[ax.name for ax in axes]}")
        print("Pixel sizes: X=0.1μm, Y=0.1μm, Z=0.25μm")
        
        # The ome_zarr_image object is now ready for further processing
        print(f"OME-Zarr image object created successfully for {ome_zarr_image.path}")
        
    except NotImplementedError:
        print("Note: This is a skeleton implementation.")
        print("The actual writing functionality will be implemented in future versions.")
        print(f"OME-Zarr image initialized for: {output_path}")
        print(f"Image shape: {image.shape} (CZYX format)")


if __name__ == "__main__":
    main()

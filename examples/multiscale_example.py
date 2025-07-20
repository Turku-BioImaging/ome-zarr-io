#!/usr/bin/env python3
"""
Example demonstrating the downscaling functionality with CZYX images.
"""

import numpy as np
from pathlib import Path
from ome_zarr_writer import OmeZarrImage, create_axes


def main():
    """Demonstrate multiscale OME-Zarr creation with downscaling."""
    print("Creating multiscale OME-Zarr file with CZYX format...")
    
    # Create sample CZYX image data
    # Shape: (channels, z_slices, height, width)
    # CZYX is the standard format for OME-Zarr multichannel 3D images
    channels, z_slices, height, width = 3, 10, 512, 512
    image = np.random.randint(0, 255, size=(channels, z_slices, height, width), dtype=np.uint8)
    
    print(f"Original image shape: {image.shape} (C={channels}, Z={z_slices}, Y={height}, X={width})")
    
    # Define output path
    output_path = Path("multiscale_example.zarr")
    
    # Remove existing file if it exists
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
    
    # Create axes for CZYX layout
    # Pixel sizes: X=0.1μm, Y=0.1μm, Z=0.25μm
    axes = create_axes("czyx", 0.1, 0.1, 0.25, unit="micrometer")
    print("Pixel sizes defined: X=0.1μm, Y=0.1μm, Z=0.25μm")
    print(f"Axes created: {[ax.name for ax in axes]}")
    
    # Create dimension names list
    dims = ["c", "z", "y", "x"]
    
    # Test different downscale levels and factor
    downscale_levels = 4
    downscale_factor = 2  # Can be 2, 3, 4, etc. - higher values create more aggressive downscaling
    
    try:
        # Initialize OME-Zarr image with downscaling
        ome_zarr_image = OmeZarrImage(
            path=output_path,
            image=image,
            dims=dims,
            downscale_levels=downscale_levels,
            downscale_factor=downscale_factor,
            overwrite=True
        )
        
        print(f"\nSuccessfully initialized OME-Zarr image: {output_path}")
        print(f"Dimensions: {dims}")
        print(f"Downscale levels: {downscale_levels}")
        print(f"Downscale factor: {downscale_factor}")
        
        # Create and display the downscaled arrays
        print("\nCreating multiscale pyramid...")
        downscaled_arrays = ome_zarr_image._create_downscaled_arrays()
        
        print(f"Number of resolution levels: {len(downscaled_arrays)}")
        for i, arr in enumerate(downscaled_arrays):
            scale_factor_actual = downscale_factor ** i
            print(f"  Level {i}: {arr.shape} (scale factor: {scale_factor_actual}x)")
        
        print("\n✅ Multiscale pyramid created successfully!")
        print("Note: Only Y and X dimensions are downscaled, preserving C and Z dimensions.")
        
    except NotImplementedError as e:
        print(f"Note: {e}")
        print(f"Multiscale structure prepared for: {output_path}")


if __name__ == "__main__":
    main()

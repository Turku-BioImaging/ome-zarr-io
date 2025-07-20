#!/usr/bin/env python3
"""
Basic example of using ome-zarr-writer to create an OME-Zarr file.

This example demonstrates how to:
1. Create sample image data
2. Initialize the OME-Zarr writer
3. Write the image with metadata
"""

import numpy as np
from pathlib import Path
from ome_zarr_writer import OMEZarrWriter


def main():
    """Create a basic OME-Zarr file."""
    print("Creating sample OME-Zarr file...")
    
    # Create sample RGB image data
    height, width, channels = 512, 512, 3
    image = np.random.randint(0, 255, size=(height, width, channels), dtype=np.uint8)
    
    # Define output path
    output_path = Path("example_output.zarr")
    
    # Remove existing file if it exists
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
    
    # Initialize writer
    writer = OMEZarrWriter(output_path, overwrite=True)
    
    try:
        # Write image with metadata
        writer.write_image(
            image=image,
            pixel_size=(0.5, 0.5),  # 0.5 micrometers per pixel
            units=["micrometer", "micrometer"],
            channel_names=["Red", "Green", "Blue"]
        )
        print(f"Successfully created OME-Zarr file: {output_path}")
        
    except NotImplementedError:
        print("Note: This is a skeleton implementation.")
        print("The actual writing functionality will be implemented in future versions.")
        print(f"Writer initialized for: {output_path}")


if __name__ == "__main__":
    main()

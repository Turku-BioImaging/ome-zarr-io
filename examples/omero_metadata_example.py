#!/usr/bin/env python3
"""
Example demonstrating OMERO metadata functionality in OME-Zarr Writer.

This example shows how to create an OME-Zarr image with OMERO metadata
for channel display configuration.
"""

import numpy as np
from pathlib import Path
import sys
sys.path.append('../src')
from ome_zarr_writer.image import OmeZarrImage
from ome_zarr_writer.schema_models import (
    Omero,
    Channel,
    Window,
)


def create_ome_zarr_with_omero_metadata():
    """Create an OME-Zarr with OMERO metadata for channel visualization."""
    
    # Create a 3-channel RGB-like image (C, Y, X)
    height, width = 256, 256
    channels = 3
    
    # Generate synthetic multichannel data
    image = np.zeros((channels, height, width), dtype=np.uint16)
    
    # Red channel - gradual intensity
    image[0, :, :] = np.linspace(0, 65535, height * width).reshape(height, width)
    
    # Green channel - circular pattern
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
    image[1, :, :] = (65535 * np.exp(-distance / 50)).astype(np.uint16)
    
    # Blue channel - checkerboard pattern
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    image[2, :, :] = ((xx // 32 + yy // 32) % 2) * 65535
    
    # Create OMERO metadata for channel display configuration
    channels_metadata = []
    
    # Red channel configuration
    red_window = Window(start=1000.0, min=0.0, end=50000.0, max=65535.0)
    red_channel = Channel(
        window=red_window,
        label="Red Fluorescence",
        color="FF0000",  # Red color
        active=True,
        family="linear"
    )
    channels_metadata.append(red_channel)
    
    # Green channel configuration
    green_window = Window(start=500.0, min=0.0, end=40000.0, max=65535.0)
    green_channel = Channel(
        window=green_window,
        label="Green Fluorescence", 
        color="00FF00",  # Green color
        active=True,
        family="linear"
    )
    channels_metadata.append(green_channel)
    
    # Blue channel configuration
    blue_window = Window(start=0.0, min=0.0, end=65535.0, max=65535.0)
    blue_channel = Channel(
        window=blue_window,
        label="Blue Fluorescence",
        color="0000FF",  # Blue color
        active=False,    # Start with this channel inactive
        family="linear"
    )
    channels_metadata.append(blue_channel)
    
    # Create OMERO metadata object
    omero_metadata = Omero(channels=channels_metadata)
    
    # Output path
    output_path = Path("multichannel_with_omero.ome.zarr")
    
    # Create OME-Zarr image with OMERO metadata
    ome_zarr = OmeZarrImage(
        path=output_path,
        image=image,
        dims=["c", "y", "x"],
        axis_units={
            "c": None,  # Channel dimension doesn't need units
            "y": "micrometer", 
            "x": "micrometer"
        },
        scale_transformations={
            "y": 0.25,  # 0.25 micrometers per pixel
            "x": 0.25   # 0.25 micrometers per pixel
        },
        downscale_levels=3,     # Create 3 additional downscale levels
        downscale_factor=2,     # Each level is 2x smaller
        omero_metadata=omero_metadata,  # Include OMERO metadata
        overwrite=True
    )
    
    # Write the OME-Zarr
    ome_zarr.write()
    
    print(f"✅ Created OME-Zarr with OMERO metadata: {output_path}")
    print(f"   - Image shape: {image.shape}")
    print(f"   - Channels: {[ch.label for ch in channels_metadata]}")
    print(f"   - Active channels: {[ch.label for ch in channels_metadata if ch.active]}")
    print(f"   - Channel colors: {[ch.color for ch in channels_metadata]}")
    
    # Display metadata information
    print("\n📊 Channel Display Configuration:")
    for i, channel in enumerate(channels_metadata):
        status = "Active" if channel.active else "Inactive"
        print(f"   Channel {i+1} ({channel.label}):")
        print(f"     - Status: {status}")
        print(f"     - Color: #{channel.color}")
        print(f"     - Display range: {channel.window.start} - {channel.window.end}")
        print(f"     - Data range: {channel.window.min} - {channel.window.max}")
    
    return output_path


def inspect_omero_metadata(zarr_path: Path):
    """Inspect the OMERO metadata in an existing OME-Zarr file."""
    import zarr
    
    # Open the zarr file
    root = zarr.open(str(zarr_path), mode='r')
    
    # Extract OMERO metadata
    if 'ome' in root.attrs and 'omero' in root.attrs['ome']:
        omero_attrs = root.attrs['ome']['omero']
        
        print(f"\n🔍 Inspecting OMERO metadata in {zarr_path}")
        print(f"   Number of channels: {len(omero_attrs['channels'])}")
        
        for i, channel in enumerate(omero_attrs['channels']):
            print(f"\n   Channel {i+1}:")
            print(f"     - Label: {channel.get('label', 'N/A')}")
            print(f"     - Color: #{channel.get('color', 'N/A')}")
            print(f"     - Active: {channel.get('active', 'N/A')}")
            if 'window' in channel:
                window = channel['window']
                print(f"     - Display window: {window['start']} - {window['end']}")
                print(f"     - Data range: {window['min']} - {window['max']}")
    else:
        print(f"❌ No OMERO metadata found in {zarr_path}")


if __name__ == "__main__":
    print("🧬 OME-Zarr Writer with OMERO Metadata Example")
    print("=" * 50)
    
    # Create OME-Zarr with OMERO metadata
    output_path = create_ome_zarr_with_omero_metadata()
    
    # Inspect the created metadata
    inspect_omero_metadata(output_path)
    
    print(f"\n🎯 Example complete! Check the created file: {output_path}")
    print("   You can open this file in compatible viewers like napari or")
    print("   other OME-Zarr visualization tools to see the channel display settings.")

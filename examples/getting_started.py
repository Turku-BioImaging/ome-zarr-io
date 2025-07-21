#!/usr/bin/env python3
"""
Getting Started with OME-Zarr Writer

This example provides the essential patterns for new users, demonstrating:
1. Basic 2D image writing
2. 3D multichannel confocal data
3. 4D time-lapse data
4. Basic compression and chunking
5. Integration with existing workflows

Run this example to understand the core concepts before exploring advanced features.
"""

import numpy as np
import tempfile
from pathlib import Path
from ome_zarr_writer import OmeZarrImage
from zarr.codecs import BloscCodec


def example_1_simple_2d_image():
    """Example 1: Simple 2D image - the minimal case."""
    print("📸 Example 1: Simple 2D Image")
    print("-" * 40)
    
    # Create sample 2D image data
    image = np.random.randint(0, 255, size=(512, 512), dtype=np.uint8)
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "simple_2d.zarr"
        
        # Create OME-Zarr image
        writer = OmeZarrImage(
            path=output_path,
            image=image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )
        
        # Write with default settings
        writer.write()
        
        print(f"✅ Created: {output_path.name}")
        print(f"   Shape: {image.shape}")
        print(f"   Dimensions: {dims}")
        print(f"   Data type: {image.dtype}")


def example_2_multichannel_3d():
    """Example 2: Multichannel 3D confocal data - most common use case."""
    print("\n🔬 Example 2: Multichannel 3D Confocal Data")
    print("-" * 50)
    
    # Create sample multichannel 3D confocal image data
    channels, z_slices, height, width = 3, 32, 512, 512
    image = np.random.randint(0, 4095, size=(channels, z_slices, height, width), dtype=np.uint16)
    dims = ["c", "z", "y", "x"]
    
    # Define axis units and scale transformations
    axis_units = {
        "z": "micrometer",
        "y": "micrometer", 
        "x": "micrometer"
        # Note: 'c' (channel) dimension doesn't need a unit
    }
    
    scale_transformations = {
        "z": 0.325,  # 0.325 μm z-step size
        "y": 0.15,   # 0.15 μm pixel size in Y  
        "x": 0.15    # 0.15 μm pixel size in X
    }
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "confocal_3d.zarr"
        
        writer = OmeZarrImage(
            path=output_path,
            image=image,
            dims=dims,
            axis_units=axis_units,
            scale_transformations=scale_transformations,
            downscale_levels=3,  # Create 3 additional downscale levels
            overwrite=True
        )
        
        # Write with basic compression
        writer.write(compressors=BloscCodec(cname="zstd", clevel=5))
        
        print(f"✅ Created: {output_path.name}")
        print(f"   Shape: {image.shape} (C={channels}, Z={z_slices}, Y={height}, X={width})")
        print(f"   Voxel size: {scale_transformations['x']:.3f} × {scale_transformations['y']:.3f} × {scale_transformations['z']:.3f} μm")
        print("   Downscale levels: 4 total (original + 3 downscaled)")
        print("   Compression: Blosc with Zstd")


def example_3_time_lapse_4d():
    """Example 3: 4D time-lapse data with temporal information."""
    print("\n⏱️  Example 3: 4D Time-lapse Data")
    print("-" * 40)
    
    # Create sample 4D time-lapse data (T, Z, Y, X)
    timepoints, z_slices, height, width = 10, 8, 256, 256
    image = np.random.randint(0, 255, size=(timepoints, z_slices, height, width), dtype=np.uint8)
    dims = ["t", "z", "y", "x"]
    
    # Define axis units for both temporal and spatial dimensions
    axis_units = {
        "t": "second",       # Time dimension in seconds
        "z": "micrometer",   # Z dimension in micrometers
        "y": "micrometer",   # Y dimension in micrometers  
        "x": "micrometer"    # X dimension in micrometers
    }
    
    scale_transformations = {
        "t": 30.0,   # 30 second intervals
        "z": 0.5,    # 0.5 μm z-step
        "y": 0.2,    # 0.2 μm pixel size
        "x": 0.2     # 0.2 μm pixel size
    }
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "timelapse_4d.zarr"
        
        writer = OmeZarrImage(
            path=output_path,
            image=image,
            dims=dims,
            axis_units=axis_units,
            scale_transformations=scale_transformations,
            downscale_levels=2,
            overwrite=True
        )
        
        # Write with custom chunking for time-lapse data
        writer.write(
            compressors=BloscCodec(cname="lz4", clevel=3),  # Fast compression for time-lapse
            chunks=(1, 8, 128, 128)  # Chunk by single timepoint
        )
        
        print(f"✅ Created: {output_path.name}")
        print(f"   Shape: {image.shape} (T={timepoints}, Z={z_slices}, Y={height}, X={width})")
        print(f"   Time interval: {scale_transformations['t']:.1f} seconds")
        print(f"   Total duration: {(timepoints-1) * scale_transformations['t']:.0f} seconds")
        print(f"   Spatial resolution: {scale_transformations['x']:.1f} μm/pixel")
        print("   Chunking: Optimized for time-lapse access")


def example_4_integration_workflow():
    """Example 4: Integration with existing analysis workflows."""
    print("\n🔗 Example 4: Integration with Existing Workflow")
    print("-" * 55)
    
    # Simulate loading data from another source (microscope, analysis pipeline, etc.)
    def simulate_data_loading():
        """Simulate loading data from your existing pipeline."""
        # This could be your actual data loading function
        data = np.random.randint(0, 65535, size=(2, 16, 256, 256), dtype=np.uint16)
        metadata = {
            "pixel_size_um": 0.1,
            "z_step_um": 0.25,
            "channels": ["DAPI", "GFP"],
            "acquisition_date": "2025-01-20",
        }
        return data, metadata
    
    # Load your existing data
    image_data, metadata = simulate_data_loading()
    
    # Convert to OME-Zarr format
    dims = ["c", "z", "y", "x"]
    axis_units = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}
    scale_transformations = {
        "z": metadata["z_step_um"],
        "y": metadata["pixel_size_um"],
        "x": metadata["pixel_size_um"]
    }
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "integrated_data.zarr"
        
        writer = OmeZarrImage(
            path=output_path,
            image=image_data,
            dims=dims,
            axis_units=axis_units,
            scale_transformations=scale_transformations,
            downscale_levels=2,
            overwrite=True
        )
        
        writer.write()
        
        print(f"✅ Created: {output_path.name}")
        print(f"   Original data shape: {image_data.shape}")
        print(f"   Channels: {metadata['channels']}")
        print(f"   Pixel size: {metadata['pixel_size_um']} μm")
        print(f"   Z-step: {metadata['z_step_um']} μm")
        print(f"   Acquired: {metadata['acquisition_date']}")
        print("   💡 This data is now ready for cloud-based analysis!")


def main():
    """Run all getting started examples."""
    print("🚀 OME-Zarr Writer - Getting Started Examples")
    print("=" * 60)
    print("These examples demonstrate the essential patterns for new users.")
    print("Run advanced_features.py for compression, multiscale, and optimization examples.")
    print()
    
    example_1_simple_2d_image()
    example_2_multichannel_3d()
    example_3_time_lapse_4d()
    example_4_integration_workflow()
    
    print("\n" + "=" * 60)
    print("🎯 Key Takeaways:")
    print("   • Use dims=['y', 'x'] for 2D, ['c', 'z', 'y', 'x'] for 3D multichannel")
    print("   • Always specify axis_units for spatial dimensions")
    print("   • Use scale_transformations to define pixel/voxel sizes")
    print("   • downscale_levels creates pyramidal multiscale representations")
    print("   • Basic compression with BloscCodec works well for most cases")
    print()
    print("Next steps:")
    print("   📚 Read the documentation: https://github.com/Turku-BioImaging/ome-zarr-writer")
    print("   🔧 Try advanced_features.py for optimization and special use cases")
    print("   📐 Try dimension_examples.py for complex dimensional arrangements")
    print("   ✅ All examples completed successfully!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Metadata and Schema Examples for OME-Zarr Writer

This example demonstrates advanced metadata capabilities:
1. Schema dataclasses usage
2. OMERO metadata integration
3. Custom coordinate transformations
4. Channel metadata and visualization
5. Advanced metadata validation

For users who need full control over OME-Zarr metadata structure.
"""

import numpy as np
import tempfile
from pathlib import Path
from ome_zarr_writer import OmeZarrImage


def example_1_basic_metadata():
    """Example 1: Basic metadata configuration."""
    print("📋 Example 1: Basic Metadata Configuration")
    print("-" * 50)
    
    # Create sample multichannel data
    image = np.random.randint(0, 4095, size=(3, 256, 256), dtype=np.uint16)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "basic_metadata.zarr"
        
        # Basic metadata through constructor parameters
        writer = OmeZarrImage(
            path=output_path,
            image=image,
            dims=["c", "y", "x"],
            axis_units={"y": "micrometer", "x": "micrometer"},
            scale_transformations={"y": 0.1, "x": 0.1},
            overwrite=True,
        )
        
        writer.write()
        
        print("   ✅ Created OME-Zarr with basic metadata")
        print("      Channels: 3")
        print("      Pixel size: 0.1 μm")
        print("      Metadata: OME-Zarr 0.5 compliant")


def example_2_multichannel_metadata():
    """Example 2: Multichannel metadata with descriptive information."""
    print("\n🎨 Example 2: Multichannel Metadata")
    print("-" * 45)
    
    # Create sample multichannel fluorescence data
    channels, height, width = 4, 512, 512
    image = np.random.randint(0, 4095, size=(channels, height, width), dtype=np.uint16)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "multichannel_metadata.zarr"
        
        # Import OMERO metadata classes
        from ome_zarr_writer.schema_models import Omero, Channel, Window
        
        # Channel information with OMERO metadata
        channel_info = [
            {"name": "DAPI", "color": "0000FF", "description": "Nuclear stain"},
            {"name": "GFP", "color": "00FF00", "description": "Protein of interest"},
            {"name": "RFP", "color": "FF0000", "description": "Cell membrane marker"},
            {"name": "Cy5", "color": "FF00FF", "description": "Secondary marker"},
        ]
        
        # Create OMERO channel metadata
        omero_channels = []
        for i, info in enumerate(channel_info):
            # Create window for display intensity range
            window = Window(start=0.0, min=0.0, end=4095.0, max=4095.0)
            
            # Create channel with metadata
            channel = Channel(
                window=window,
                label=info["name"],
                color=info["color"],
                active=True
            )
            omero_channels.append(channel)
        
        # Create OMERO metadata object
        omero_metadata = Omero(channels=omero_channels)
        
        writer = OmeZarrImage(
            path=output_path,
            image=image,
            dims=["c", "y", "x"],
            axis_units={"y": "micrometer", "x": "micrometer"},
            scale_transformations={"y": 0.064, "x": 0.064},
            omero_metadata=omero_metadata,
            overwrite=True,
        )
        
        writer.write()
        
        print("   ✅ Created multichannel OME-Zarr with OMERO metadata")
        print("      Channel configuration:")
        for info in channel_info:
            print(f"         {info['name']}: {info['description']} (color: #{info['color']})")
        print("      Compatible with OME-Zarr viewers")
        print("      ✅ OMERO metadata written to zarr attributes")
        
        # Verify OMERO metadata was written correctly
        import zarr
        root = zarr.open(str(output_path), mode='r')
        attrs = dict(root.attrs)
        if 'ome' in attrs and isinstance(attrs['ome'], dict) and 'omero' in attrs['ome']:
            omero_attrs = attrs['ome']['omero']
            if isinstance(omero_attrs, dict) and 'channels' in omero_attrs:
                channels = omero_attrs['channels']
                if isinstance(channels, list):
                    print(f"      📊 Verified: {len(channels)} channels in OMERO metadata")
                else:
                    print("      ⚠️  Warning: OMERO channels data is not a list")
            else:
                print("      ⚠️  Warning: OMERO channels not found")
        else:
            print("      ⚠️  Warning: OMERO metadata not found in zarr attributes")


def example_3_coordinate_transformations():
    """Example 3: Custom coordinate transformations."""
    print("\n🗺️  Example 3: Custom Coordinate Transformations")
    print("-" * 55)
    
    # Create sample Z-stack data
    image = np.random.randint(0, 255, size=(2, 32, 256, 256), dtype=np.uint8)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "transformations_example.zarr"
        
        # Example of custom coordinate transformation (rotation + scaling)
        # This would be useful for data that's been rotated or has non-standard orientation
        
        # For this example, we'll use the simple scale transformation
        # but show how you could extend it for more complex cases
        
        writer = OmeZarrImage(
            path=output_path,
            image=image,
            dims=["c", "z", "y", "x"],
            axis_units={"z": "micrometer", "y": "micrometer", "x": "micrometer"},
            scale_transformations={
                "z": 0.5,    # 500 nm Z-step
                "y": 0.1,    # 100 nm pixel
                "x": 0.1,    # 100 nm pixel
            },
            downscale_levels=3,
            overwrite=True,
        )
        
        writer.write()
        
        print("   ✅ Created OME-Zarr with coordinate transformations")
        print("      Z resolution: 500 nm/slice")
        print("      XY resolution: 100 nm/pixel")
        print("      Transformations applied to all pyramid levels")
        print("      💡 Can be extended for rotation, translation, etc.")


def example_4_multichannel_configurations():
    """Example 4: Different multichannel configurations."""
    print("\n🌈 Example 4: Multichannel Configurations")
    print("-" * 50)
    
    multichannel_configs = [
        {
            "name": "Fluorescence microscopy",
            "dims": ["c", "z", "y", "x"],
            "shape": (4, 16, 256, 256),
            "channels": ["DAPI", "FITC", "TRITC", "Cy5"],
            "description": "Standard fluorescence channels"
        },
        {
            "name": "Bright-field series",
            "dims": ["c", "y", "x"],
            "shape": (3, 512, 512),
            "channels": ["Red", "Green", "Blue"],
            "description": "RGB color channels"
        },
        {
            "name": "Multiplex imaging",
            "dims": ["c", "y", "x"],
            "shape": (8, 1024, 1024),
            "channels": [f"Marker_{i+1}" for i in range(8)],
            "description": "High-content multiplex assay"
        }
    ]
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Import OMERO metadata classes
        from ome_zarr_writer.schema_models import Omero, Channel, Window
        
        for config in multichannel_configs:
            image = np.random.randint(0, 4095, size=config["shape"], dtype=np.uint16)
            output_path = Path(tmp_dir) / f"{config['name'].lower().replace(' ', '_')}.zarr"
            
            # Set up axis configuration
            axis_units = {}
            scale_transformations = {}
            for dim in config["dims"]:
                if dim in ["z", "y", "x"]:
                    axis_units[dim] = "micrometer"
                    scale_transformations[dim] = 0.1  # 100 nm resolution
            
            # Create OMERO metadata for multichannel data
            omero_metadata = None
            if "c" in config["dims"]:
                # Create OMERO channel metadata
                omero_channels = []
                for i, channel_name in enumerate(config["channels"]):
                    # Create window for display intensity range
                    window = Window(start=0.0, min=0.0, end=4095.0, max=4095.0)
                    
                    # Define default colors for common channel types
                    default_colors = {
                        "DAPI": "0000FF", "FITC": "00FF00", "TRITC": "FF0000", "Cy5": "FF00FF",
                        "Red": "FF0000", "Green": "00FF00", "Blue": "0000FF",
                    }
                    color = default_colors.get(channel_name, f"{(i*60)%256:02X}{(i*120)%256:02X}{(i*180)%256:02X}")
                    
                    # Create channel with metadata
                    channel = Channel(
                        window=window,
                        label=channel_name,
                        color=color,
                        active=True
                    )
                    omero_channels.append(channel)
                
                # Create OMERO metadata object
                omero_metadata = Omero(channels=omero_channels)
            
            writer = OmeZarrImage(
                path=output_path,
                image=image,
                dims=config["dims"],
                axis_units=axis_units,
                scale_transformations=scale_transformations,
                omero_metadata=omero_metadata,
                overwrite=True,
            )
            
            writer.write()
            
            print(f"   ✅ {config['name']}:")
            print(f"      Description: {config['description']}")
            print(f"      Channels ({len(config['channels'])}): {', '.join(config['channels'])}")
            print(f"      Dimensions: {config['dims']}")
            if omero_metadata:
                print(f"      ✅ OMERO metadata written with {len(config['channels'])} channel definitions")
            else:
                print("      ℹ️  No channel dimension - OMERO metadata not applicable")
            print()


def example_5_validation_examples():
    """Example 5: Metadata validation examples."""
    print("✅ Example 5: Metadata Validation")
    print("-" * 40)
    
    print("   Valid configurations:")
    
    # Valid spatial units
    valid_spatial_units = ["angstrom", "nanometer", "micrometer", "millimeter", "meter"]
    print(f"      Spatial units: {', '.join(valid_spatial_units)}")
    
    # Valid temporal units  
    valid_temporal_units = ["attosecond", "nanosecond", "millisecond", "second", "minute", "hour", "day"]
    print(f"      Temporal units: {', '.join(valid_temporal_units)}")
    
    # Valid dimension orders
    valid_orders = [
        ["y", "x"],
        ["z", "y", "x"],
        ["c", "y", "x"],
        ["t", "y", "x"],
        ["c", "z", "y", "x"],
        ["t", "z", "y", "x"],
        ["t", "c", "y", "x"],
        ["t", "c", "z", "y", "x"]
    ]
    print("      Valid dimension orders:")
    for order in valid_orders:
        print(f"         {order}")
    
    print()
    print("   💡 The OME-Zarr writer automatically validates:")
    print("      • Dimension ordering follows TCZYX convention")
    print("      • Axis units are from approved vocabularies")
    print("      • Scale transformations match axis dimensions")
    print("      • Metadata structure follows OME-Zarr 0.5 schema")
    
    # Example of creating a validated configuration
    with tempfile.TemporaryDirectory() as tmp_dir:
        image = np.random.randint(0, 255, size=(2, 8, 128, 128), dtype=np.uint8)
        output_path = Path(tmp_dir) / "validated_example.zarr"
        
        writer = OmeZarrImage(
            path=output_path,
            image=image,
            dims=["c", "z", "y", "x"],
            axis_units={"z": "micrometer", "y": "micrometer", "x": "micrometer"},
            scale_transformations={"z": 0.2, "y": 0.1, "x": 0.1},
            overwrite=True,
        )
        
        writer.write()
        
        print(f"   ✅ Created validated example: {output_path.name}")
        print("      All metadata passed validation checks")


def main():
    """Run all metadata and schema examples."""
    print("🚀 OME-Zarr Writer - Metadata and Schema Examples")
    print("=" * 65)
    print("These examples demonstrate advanced metadata capabilities and schema usage.")
    print("Focus on understanding metadata structure and validation requirements.")
    print()
    
    example_1_basic_metadata()
    example_2_multichannel_metadata()
    example_3_coordinate_transformations()
    example_4_multichannel_configurations()
    example_5_validation_examples()
    
    print("\n" + "=" * 65)
    print("🎯 Metadata Best Practices:")
    print("   • Use schema dataclasses for precise control")
    print("   • Create OMERO metadata objects to write channel information")
    print("   • Define proper coordinate transformations")
    print("   • Validate all metadata against OME-Zarr schema")
    print("   • Include descriptive channel names and colors in OMERO metadata")
    print()
    print("Key Benefits:")
    print("   📊 Schema validation ensures compatibility")
    print("   🎨 OMERO metadata written to zarr enables better visualization")
    print("   🗺️  Coordinate transformations preserve spatial information")
    print("   🔍 Rich metadata enables better analysis workflows")
    print()
    print("✅ All metadata examples completed successfully!")


if __name__ == "__main__":
    main()

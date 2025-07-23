#!/usr/bin/env python3
"""
Dimension and Axis Examples for OME-Zarr Writer

This example demonstrates working with different dimensional arrangements:
1. All supported dimension orders (2D to 5D)
2. Axis units and scale transformations
3. Time-lapse data with temporal units
4. Per-dimension axis configuration
5. Coordinate transformations

Focus on understanding dimension ordering and axis configuration.
"""

import numpy as np
import tempfile
from pathlib import Path
from ome_zarr_writer import OmeZarrImage


def example_1_dimension_orders():
    """Example 1: All supported dimension orders from 2D to 5D."""
    print("📐 Example 1: Supported Dimension Orders")
    print("-" * 50)

    dimension_examples = [
        # (dims, shape, description)
        (["y", "x"], (512, 512), "2D image"),
        (["z", "y", "x"], (32, 512, 512), "3D volume"),
        (["c", "y", "x"], (3, 512, 512), "2D multichannel"),
        (["t", "y", "x"], (10, 512, 512), "2D time-lapse"),
        (["c", "z", "y", "x"], (3, 32, 512, 512), "3D multichannel"),
        (["t", "z", "y", "x"], (10, 32, 512, 512), "3D time-lapse"),
        (["t", "c", "y", "x"], (10, 3, 512, 512), "2D multichannel time-lapse"),
        (
            ["t", "c", "z", "y", "x"],
            (5, 2, 16, 256, 256),
            "5D: full time-lapse multichannel 3D",
        ),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        for dims, shape, description in dimension_examples:
            # Create sample data with appropriate shape
            image = np.random.randint(0, 255, size=shape, dtype=np.uint8)

            # Define axis units for spatial dimensions
            axis_units = {}
            if "z" in dims:
                axis_units["z"] = "micrometer"
            if "y" in dims:
                axis_units["y"] = "micrometer"
            if "x" in dims:
                axis_units["x"] = "micrometer"
            if "t" in dims:
                axis_units["t"] = "second"

            output_path = Path(tmp_dir) / f"{'_'.join(dims)}.zarr"

            writer = OmeZarrImage(
                path=output_path,
                image=image,
                dims=dims,
                axis_units=axis_units,
                overwrite=True,
            )

            writer.write()

            print(f"   ✅ {description}")
            print(f"      Dimensions: {dims}")
            print(f"      Shape: {shape}")
            print(f"      File: {output_path.name}")
            print()


def example_2_axis_units_comprehensive():
    """Example 2: Comprehensive axis units examples."""
    print("📏 Example 2: Comprehensive Axis Units")
    print("-" * 45)

    # Spatial units examples
    spatial_examples = [
        ("nanometer", 50, "High-resolution EM data"),
        ("micrometer", 0.1, "Standard fluorescence microscopy"),
        ("millimeter", 0.5, "Low-magnification imaging"),
        ("meter", 0.001, "Large-scale imaging"),
    ]

    print("   Spatial units examples:")
    with tempfile.TemporaryDirectory() as tmp_dir:
        for unit, pixel_size, description in spatial_examples:
            image = np.random.randint(0, 255, size=(128, 128), dtype=np.uint8)
            dims = ["y", "x"]
            axis_units = {"y": unit, "x": unit}
            scale_transformations = {"y": pixel_size, "x": pixel_size}

            output_path = Path(tmp_dir) / f"spatial_{unit}.zarr"

            writer = OmeZarrImage(
                path=output_path,
                image=image,
                dims=dims,
                axis_units=axis_units,
                scale_transformations=scale_transformations,
                overwrite=True,
            )

            writer.write()

            print(f"      • {unit}: {pixel_size} {unit}/pixel ({description})")

    print()

    # Temporal units examples
    temporal_examples = [
        ("second", 1.0, "Real-time imaging"),
        ("millisecond", 100, "Fast dynamics"),
        ("minute", 5, "Slow processes"),
        ("hour", 0.5, "Long-term studies"),
    ]

    print("   Temporal units examples:")
    with tempfile.TemporaryDirectory() as tmp_dir:
        for unit, interval, description in temporal_examples:
            image = np.random.randint(0, 255, size=(10, 128, 128), dtype=np.uint8)
            dims = ["t", "y", "x"]
            axis_units = {"t": unit, "y": "micrometer", "x": "micrometer"}
            scale_transformations = {"t": interval, "y": 0.1, "x": 0.1}

            output_path = Path(tmp_dir) / f"temporal_{unit}.zarr"

            writer = OmeZarrImage(
                path=output_path,
                image=image,
                dims=dims,
                axis_units=axis_units,
                scale_transformations=scale_transformations,
                overwrite=True,
            )

            writer.write()

            print(f"      • {unit}: {interval} {unit}/frame ({description})")


def example_3_scale_transformations():
    """Example 3: Scale transformations for different imaging modalities."""
    print("\n🔬 Example 3: Scale Transformations for Different Modalities")
    print("-" * 65)

    modality_examples = [
        {
            "name": "Confocal Microscopy",
            "dims": ["c", "z", "y", "x"],
            "shape": (3, 32, 512, 512),
            "axis_units": {"z": "micrometer", "y": "micrometer", "x": "micrometer"},
            "scales": {"z": 0.2, "y": 0.065, "x": 0.065},
            "description": "Typical confocal with 63x objective",
        },
        {
            "name": "Light Sheet Microscopy",
            "dims": ["t", "c", "z", "y", "x"],
            "shape": (20, 2, 64, 256, 256),
            "axis_units": {
                "t": "second",
                "z": "micrometer",
                "y": "micrometer",
                "x": "micrometer",
            },
            "scales": {"t": 30, "z": 1.0, "y": 0.4, "x": 0.4},
            "description": "Fast 3D time-lapse imaging",
        },
        {
            "name": "Super-resolution",
            "dims": ["c", "y", "x"],
            "shape": (2, 1024, 1024),
            "axis_units": {"y": "nanometer", "x": "nanometer"},
            "scales": {"y": 20, "x": 20},
            "description": "STORM/PALM super-resolution",
        },
        {
            "name": "Slide Scanner",
            "dims": ["c", "y", "x"],
            "shape": (3, 2048, 2048),
            "axis_units": {"y": "micrometer", "x": "micrometer"},
            "scales": {"y": 0.25, "x": 0.25},
            "description": "Whole slide imaging",
        },
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        for config in modality_examples:
            image = np.random.randint(0, 4095, size=config["shape"], dtype=np.uint16)
            output_path = (
                Path(tmp_dir) / f"{config['name'].lower().replace(' ', '_')}.zarr"
            )

            writer = OmeZarrImage(
                path=output_path,
                image=image,
                dims=config["dims"],
                axis_units=config["axis_units"],
                scale_transformations=config["scales"],
                downscale_levels=2,
                overwrite=True,
            )

            writer.write()

            print(f"   ✅ {config['name']}:")
            print(f"      Description: {config['description']}")
            print(f"      Dimensions: {config['dims']}")
            print(f"      Resolution: {config['scales']}")
            print()


def example_4_time_series_configurations():
    """Example 4: Different time-series imaging configurations."""
    print("⏱️  Example 4: Time-series Imaging Configurations")
    print("-" * 55)

    time_series_configs = [
        {
            "name": "Fast calcium imaging",
            "dims": ["t", "y", "x"],
            "shape": (1000, 128, 128),
            "time_unit": "millisecond",
            "time_interval": 50,  # 50 ms = 20 Hz
            "spatial_res": 0.5,
            "description": "High-speed calcium indicator imaging",
        },
        {
            "name": "Cell division tracking",
            "dims": ["t", "c", "z", "y", "x"],
            "shape": (48, 2, 16, 256, 256),
            "time_unit": "minute",
            "time_interval": 15,  # Every 15 minutes
            "spatial_res": 0.1,
            "description": "Long-term cell division tracking",
        },
        {
            "name": "Development time-lapse",
            "dims": ["t", "c", "y", "x"],
            "shape": (72, 3, 512, 512),
            "time_unit": "hour",
            "time_interval": 2,  # Every 2 hours
            "spatial_res": 2.0,
            "description": "Embryonic development over 6 days",
        },
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        for config in time_series_configs:
            image = np.random.randint(0, 255, size=config["shape"], dtype=np.uint8)

            # Set up axis units
            axis_units = {"t": config["time_unit"]}
            scale_transformations = {"t": config["time_interval"]}

            # Add spatial dimensions
            for dim in ["z", "y", "x"]:
                if dim in config["dims"]:
                    axis_units[dim] = "micrometer"
                    scale_transformations[dim] = config["spatial_res"]

            output_path = (
                Path(tmp_dir) / f"{config['name'].lower().replace(' ', '_')}.zarr"
            )

            writer = OmeZarrImage(
                path=output_path,
                image=image,
                dims=config["dims"],
                axis_units=axis_units,
                scale_transformations=scale_transformations,
                overwrite=True,
            )

            writer.write()

            # Calculate total duration
            total_time = (config["shape"][0] - 1) * config["time_interval"]

            print(f"   ✅ {config['name']}:")
            print(f"      Description: {config['description']}")
            print(f"      Time points: {config['shape'][0]}")
            print(f"      Interval: {config['time_interval']} {config['time_unit']}")
            print(f"      Total duration: {total_time} {config['time_unit']}")
            print(f"      Spatial resolution: {config['spatial_res']} μm/pixel")
            print()


def example_5_coordinate_system_examples():
    """Example 5: Different coordinate system examples."""
    print("🗺️  Example 5: Coordinate System Examples")
    print("-" * 45)

    # Example with anisotropic voxels (common in microscopy)
    print("   Anisotropic voxel example (confocal microscopy):")
    with tempfile.TemporaryDirectory() as tmp_dir:
        image = np.random.randint(0, 4095, size=(2, 64, 512, 512), dtype=np.uint16)
        dims = ["c", "z", "y", "x"]
        axis_units = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}

        # Typical confocal: much larger Z step than XY pixel size
        scale_transformations = {
            "z": 0.3,  # 300 nm Z-step
            "y": 0.064,  # 64 nm pixel
            "x": 0.064,  # 64 nm pixel
        }

        output_path = Path(tmp_dir) / "anisotropic_confocal.zarr"

        writer = OmeZarrImage(
            path=output_path,
            image=image,
            dims=dims,
            axis_units=axis_units,
            scale_transformations=scale_transformations,
            downscale_levels=3,
            overwrite=True,
        )

        writer.write()

        print(f"      X resolution: {scale_transformations['x']*1000:.0f} nm/pixel")
        print(f"      Y resolution: {scale_transformations['y']*1000:.0f} nm/pixel")
        print(f"      Z resolution: {scale_transformations['z']*1000:.0f} nm/slice")
        print(
            f"      Anisotropy ratio (Z:XY): {scale_transformations['z']/scale_transformations['x']:.1f}:1"
        )

    print()

    # Example with isotropic voxels
    print("   Isotropic voxel example (light sheet microscopy):")
    with tempfile.TemporaryDirectory() as tmp_dir:
        image = np.random.randint(0, 255, size=(16, 128, 128, 128), dtype=np.uint8)
        dims = ["t", "z", "y", "x"]
        axis_units = {
            "t": "second",
            "z": "micrometer",
            "y": "micrometer",
            "x": "micrometer",
        }

        # Light sheet: isotropic spatial resolution
        scale_transformations = {
            "t": 60,  # 1 minute intervals
            "z": 0.5,  # 500 nm
            "y": 0.5,  # 500 nm
            "x": 0.5,  # 500 nm
        }

        output_path = Path(tmp_dir) / "isotropic_lightsheet.zarr"

        writer = OmeZarrImage(
            path=output_path,
            image=image,
            dims=dims,
            axis_units=axis_units,
            scale_transformations=scale_transformations,
            overwrite=True,
        )

        writer.write()

        print(f"      Spatial resolution: {scale_transformations['x']} μm (isotropic)")
        print(f"      Temporal resolution: {scale_transformations['t']} seconds")
        print(
            f"      Total duration: {(image.shape[0]-1) * scale_transformations['t']/60:.1f} minutes"
        )


def main():
    """Run all dimension and axis examples."""
    print("🚀 OME-Zarr Writer - Dimension and Axis Examples")
    print("=" * 65)
    print(
        "These examples demonstrate different dimensional arrangements and axis configuration."
    )
    print(
        "Focus on understanding how to specify dimensions, units, and scale transformations."
    )
    print()

    example_1_dimension_orders()
    example_2_axis_units_comprehensive()
    example_3_scale_transformations()
    example_4_time_series_configurations()
    example_5_coordinate_system_examples()

    print("=" * 65)
    print("🎯 Dimension Guidelines:")
    print("   • Always use standard order: [t, c, z, y, x]")
    print("   • Spatial dimensions (z, y, x) require units")
    print("   • Time dimension (t) requires temporal units")
    print("   • Channel dimension (c) doesn't need units")
    print("   • Use scale_transformations to define real-world sizes")
    print()
    print("Supported spatial units:")
    print("   📏 angstrom, nanometer, micrometer, millimeter, meter, etc.")
    print()
    print("Supported temporal units:")
    print("   ⏱️  attosecond, nanosecond, millisecond, second, minute, hour, day")
    print()
    print("✅ All dimension examples completed successfully!")


if __name__ == "__main__":
    main()

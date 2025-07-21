#!/usr/bin/env python3
"""Complete demonstration of the OME-Zarr writer with write() method functionality."""

import numpy as np
import zarr
from pathlib import Path
import tempfile
from ome_zarr_writer.image import OmeZarrImage


def main():
    """Demonstrate complete OME-Zarr writing functionality."""
    print("🧬 OME-Zarr Writer - Complete Example")
    print("=" * 50)

    # Example 1: Simple 2D image
    print("\n📸 Example 1: Simple 2D Image")
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create sample image
        image_2d = np.random.randint(0, 255, size=(128, 128), dtype=np.uint8)
        dims = ["y", "x"]
        axis_units = {"y": "micrometer", "x": "micrometer"}
        output_path = Path(tmp_dir) / "simple_2d.zarr"

        # Create and write OME-Zarr
        writer = OmeZarrImage(
            path=output_path,
            image=image_2d,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )
        writer.write()

        # Verify the output
        group = zarr.open_group(str(output_path), mode="r")
        print(f"✅ Created: {output_path.name}")
        print(f"   - Levels: {list(group.keys())}")
        print(f"   - Shape: {group['0'].shape}")
        print(f"   - OME version: {group.attrs['ome']['version']}")

    # Example 2: Multiscale 4D image with coordinate transformations
    print("\n📊 Example 2: Multiscale 4D Image with Transformations")
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create sample 4D image (channels, z, y, x)
        image_4d = np.random.randint(0, 255, size=(3, 10, 128, 128), dtype=np.uint16)
        dims = ["c", "z", "y", "x"]
        axis_units = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}

        # Define scale transformations (0.1 μm pixel size)
        scale_transforms = {
            "z": 1.0,    # 1 μm z spacing
            "y": 0.1,    # 0.1 μm y pixel size
            "x": 0.1     # 0.1 μm x pixel size
        }

        output_path = Path(tmp_dir) / "multiscale_4d.zarr"

        # Create multiscale OME-Zarr with 3 downscale levels
        writer = OmeZarrImage(
            path=output_path,
            image=image_4d,
            dims=dims,
            axis_units=axis_units,
            scale_transformations=scale_transforms,
            downscale_levels=3,
            downscale_factor=2,
            overwrite=True,
        )
        writer.write()

        # Verify the output
        group = zarr.open_group(str(output_path), mode="r")
        print(f"✅ Created: {output_path.name}")
        print(f"   - Levels: {list(group.keys())}")
        for level in ["0", "1", "2", "3"]:
            if level in group:
                print(f"   - Level {level}: {group[level].shape}")

        # Check coordinate transformations
        ome_meta = group.attrs["ome"]
        datasets = ome_meta["multiscales"][0]["datasets"]
        print(f"   - Coordinate transformations per level:")
        for i, dataset in enumerate(datasets):
            transforms = dataset["coordinateTransformations"]
            print(f"     Level {i}: {len(transforms)} transforms")

    # Example 3: Time-lapse with custom units
    print("\n⏰ Example 3: Time-lapse with Custom Units")
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create time-lapse image (time, y, x)
        image_time = np.random.randint(0, 255, size=(5, 64, 64), dtype=np.uint8)
        dims = ["t", "y", "x"]
        axis_units = {"t": "second", "y": "micrometer", "x": "micrometer"}

        output_path = Path(tmp_dir) / "timelapse.zarr"

        writer = OmeZarrImage(
            path=output_path,
            image=image_time,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )
        writer.write()

        # Verify axes
        group = zarr.open_group(str(output_path), mode="r")
        ome_meta = group.attrs["ome"]
        axes = ome_meta["multiscales"][0]["axes"]

        print(f"✅ Created: {output_path.name}")
        print(f"   - Shape: {group['0'].shape}")
        print(f"   - Axes:")
        for axis in axes:
            print(
                f"     {axis['name']}: {axis['type']} ({axis.get('unit', 'no unit')})"
            )

    # Example 4: Using List of Axis objects directly
    print("\n🎯 Example 4: Using List of Axis Objects")
    with tempfile.TemporaryDirectory() as tmp_dir:
        from ome_zarr_writer.schema_models import Axis

        # Create 3D image
        image_3d = np.random.randint(0, 255, size=(20, 64, 64), dtype=np.uint8)
        dims = ["z", "y", "x"]

        # Define axes explicitly
        axis_list = [
            Axis(name="z", type="space", unit="micrometer"),
            Axis(name="y", type="space", unit="nanometer"),  # Different unit for Y
            Axis(name="x", type="space", unit="nanometer"),  # Different unit for X
        ]

        output_path = Path(tmp_dir) / "custom_axes.zarr"

        writer = OmeZarrImage(
            path=output_path,
            image=image_3d,
            dims=dims,
            axis_units=axis_list,
            downscale_levels=2,
            overwrite=True,
        )
        writer.write()

        # Verify custom axes
        group = zarr.open_group(str(output_path), mode="r")
        ome_meta = group.attrs["ome"]
        axes = ome_meta["multiscales"][0]["axes"]

        print(f"✅ Created: {output_path.name}")
        print(f"   - Shape: {group['0'].shape}")
        print(f"   - Custom axes:")
        for axis in axes:
            print(
                f"     {axis['name']}: {axis['type']} ({axis.get('unit', 'no unit')})"
            )

    print("\n🎉 All examples completed successfully!")
    print("\n📚 Key Features Demonstrated:")
    print("   ✓ Simple 2D image writing")
    print("   ✓ Multiscale pyramid generation")
    print("   ✓ Coordinate transformations")
    print("   ✓ Time-lapse data with custom time units")
    print("   ✓ Custom axis specifications")
    print("   ✓ OME-Zarr 0.5 compliant metadata")
    print("   ✓ Zarr v3 format support")


if __name__ == "__main__":
    main()

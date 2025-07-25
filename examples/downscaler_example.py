"""Example demonstrating the use of the separate Downscaler class."""

import numpy as np
import dask.array as da
from ome_zarr_writer import OmeZarrImage, Downscaler
from ome_zarr_writer.schema_models import ScaleTransformation


def main():
    # Create a sample 2D image
    image = np.random.randint(0, 255, size=(200, 200), dtype=np.uint8)
    image_da = da.from_array(image, chunks="auto")

    print("=== Demonstrating the new Downscaler class ===")
    print(f"Original image shape: {image.shape}")

    # Example 1: Using Downscaler independently
    print("\n1. Using Downscaler class independently:")
    
    # Create a downscaler instance
    downscaler = Downscaler(
        downscale_factor=2.0,
        downscale_method="gaussian",
        downscale_levels=3
    )

    # Get downscaled arrays
    downscaled_arrays = downscaler.create_downscaled_arrays(image_da)
    print(f"   Created {len(downscaled_arrays)} resolution levels")
    for i, arr in enumerate(downscaled_arrays):
        print(f"   Level {i}: {arr.shape}")

    # Example 2: Create coordinate transformations
    print("\n2. Creating coordinate transformations:")
    
    # Define original coordinate transformations
    original_transforms = [ScaleTransformation(scale=[0.1, 0.1])]  # 0.1 µm/pixel
    
    # Create transformations for all levels
    level_transforms = downscaler.create_coordinate_transformations_for_levels(
        original_transforms,
        image.shape,
        len(downscaled_arrays)
    )
    
    print(f"   Created transformations for {len(level_transforms)} levels")
    for i, transforms in enumerate(level_transforms):
        scale = transforms[0].scale
        print(f"   Level {i}: scale = {scale} (pixel size: {scale[0]:.2f} x {scale[1]:.2f} µm)")

    # Example 3: Using with OmeZarrImage (integrated approach)
    print("\n3. Using with OmeZarrImage (same as before):")
    
    path = "/tmp/example_multiscale.zarr"
    dims = ["y", "x"]
    axis_units = {"y": "micrometer", "x": "micrometer"}
    scale_transformations = {"y": 0.1, "x": 0.1}

    # Create OmeZarrImage (which internally uses the Downscaler)
    writer = OmeZarrImage(
        path=path,
        image=image,
        dims=dims,
        axis_units=axis_units,
        downscale_method="gaussian",
        downscale_levels=3,
        downscale_factor=2.0,
        scale_transformations=scale_transformations,
    )

    # The downscaler is accessible via the writer
    print(f"   OmeZarrImage.downscaler: {type(writer.downscaler).__name__}")
    print(f"   Downscale method: {writer.downscale_method}")
    print(f"   Downscale levels: {writer.downscale_levels}")
    print(f"   Downscale factor: {writer.downscale_factor}")

    # Write the OME-Zarr file
    writer.write()
    print(f"   ✅ Successfully wrote OME-Zarr to: {path}")

    print("\n=== Benefits of the refactored design ===")
    print("✓ Separation of concerns: downscaling logic is isolated")
    print("✓ Reusability: Downscaler can be used independently")
    print("✓ Testability: Downscaler can be tested in isolation")
    print("✓ Maintainability: Easier to extend or modify downscaling")
    print("✓ Backward compatibility: OmeZarrImage API remains the same")


if __name__ == "__main__":
    main()

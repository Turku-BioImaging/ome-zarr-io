#!/usr/bin/env python3
"""
Basic example of using ome-zarr-writer to create an OME-Zarr file.

This example demonstrates how to:
1. Create sample CZYX image data (Channel, Z, Y, X)
2. Initialize the OME-Zarr image with proper axis units
3. Write a complete OME-Zarr file with multiscale pyramids
4. Configure chunking, sharding, and compression options
"""

import numpy as np
from pathlib import Path
from ome_zarr_writer import OmeZarrImage


def create_basic_example():
    """Create a basic OME-Zarr file with default settings."""
    print("=" * 60)
    print("Creating basic OME-Zarr file with default settings...")
    
    # Create sample CZYX image data
    channels, z_slices, height, width = 3, 10, 512, 512
    image = np.random.randint(
        0, 255, size=(channels, z_slices, height, width), dtype=np.uint8
    )

    print(f"Image shape: {image.shape} (C={channels}, Z={z_slices}, Y={height}, X={width})")

    # Basic configuration
    output_path = Path("example-basic.zarr")
    dims = ["c", "z", "y", "x"]
    axis_units = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}

    ome_zarr_image = OmeZarrImage(
        path=output_path,
        image=image,
        dims=dims,
        axis_units=axis_units,
        downscale_levels=3,
        overwrite=True,
    )

    print(f"Writing basic example to: {output_path}")
    ome_zarr_image.write()
    print("✓ Basic example completed")


def create_chunked_example():
    """Create an OME-Zarr file with custom chunking strategy."""
    print("=" * 60)
    print("Creating OME-Zarr file with custom chunking...")
    
    channels, z_slices, height, width = 3, 20, 1024, 1024
    image = np.random.randint(
        0, 65535, size=(channels, z_slices, height, width), dtype=np.uint16
    )

    output_path = Path("example-chunked.zarr")
    dims = ["c", "z", "y", "x"]
    axis_units = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}

    # Custom chunking strategy - optimize for different access patterns
    # Chunk size should balance memory usage and I/O performance
    chunk_shape = (1, 4, 256, 256)  # (C, Z, Y, X) - good for z-stack access
    
    ome_zarr_image = OmeZarrImage(
        path=output_path,
        image=image,
        dims=dims,
        axis_units=axis_units,
        downscale_levels=3,
        chunk_shape=chunk_shape,
        overwrite=True,
    )

    print(f"Image shape: {image.shape}")
    print(f"Chunk shape: {chunk_shape}")
    print(f"Writing chunked example to: {output_path}")
    ome_zarr_image.write()
    print("✓ Chunked example completed")


def create_compressed_example():
    """Create an OME-Zarr file with compression."""
    print("=" * 60)
    print("Creating OME-Zarr file with compression...")
    
    channels, z_slices, height, width = 3, 15, 512, 512
    image = np.random.randint(
        0, 255, size=(channels, z_slices, height, width), dtype=np.uint8
    )

    output_path = Path("example-compressed.zarr")
    dims = ["c", "z", "y", "x"]
    axis_units = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}

    # Configure compression - blosc is fast and effective for scientific data
    compression_config = {
        "compressor": "blosc",
        "compression_level": 5,  # Balance between speed and compression ratio
        "shuffle": True,  # Improve compression for scientific data
    }
    
    ome_zarr_image = OmeZarrImage(
        path=output_path,
        image=image,
        dims=dims,
        axis_units=axis_units,
        downscale_levels=3,
        chunk_shape=(1, 5, 128, 128),  # Smaller chunks work better with compression
        compression=compression_config,
        overwrite=True,
    )

    print(f"Image shape: {image.shape}")
    print(f"Compression: {compression_config}")
    print(f"Writing compressed example to: {output_path}")
    ome_zarr_image.write()
    print("✓ Compressed example completed")


def create_sharded_example():
    """Create an OME-Zarr file with sharding for better performance."""
    print("=" * 60)
    print("Creating OME-Zarr file with sharding...")
    
    channels, z_slices, height, width = 2, 25, 2048, 2048
    image = np.random.randint(
        0, 4095, size=(channels, z_slices, height, width), dtype=np.uint16
    )

    output_path = Path("example-sharded.zarr")
    dims = ["c", "z", "y", "x"]
    axis_units = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}

    # Sharding configuration - groups chunks into larger files for better I/O
    shard_config = {
        "shard_shape": (1, 5, 512, 512),  # Size of each shard
        "index_location": "start",  # Where to store the index
    }
    
    chunk_shape = (1, 5, 256, 256)  # Should be smaller than or equal to shard_shape
    
    ome_zarr_image = OmeZarrImage(
        path=output_path,
        image=image,
        dims=dims,
        axis_units=axis_units,
        downscale_levels=2,
        chunk_shape=chunk_shape,
        sharding=shard_config,
        compression={"compressor": "zstd", "compression_level": 3},
        overwrite=True,
    )

    print(f"Image shape: {image.shape}")
    print(f"Chunk shape: {chunk_shape}")
    print(f"Shard config: {shard_config}")
    print(f"Writing sharded example to: {output_path}")
    ome_zarr_image.write()
    print("✓ Sharded example completed")


def create_optimized_example():
    """Create an OME-Zarr file with optimized settings for large datasets."""
    print("=" * 60)
    print("Creating optimized OME-Zarr file for large datasets...")
    
    channels, z_slices, height, width = 4, 50, 4096, 4096
    # Create a more realistic pattern instead of pure random
    image = np.zeros((channels, z_slices, height, width), dtype=np.uint16)
    for c in range(channels):
        for z in range(z_slices):
            # Create some structure to make compression more effective
            y, x = np.ogrid[:height, :width]
            pattern = np.sin(y * 0.01 + c) * np.cos(x * 0.01 + z) * 2000 + 2000
            noise = np.random.normal(0, 100, (height, width))
            image[c, z] = np.clip(pattern + noise, 0, 65535).astype(np.uint16)

    output_path = Path("example-optimized.zarr")
    dims = ["c", "z", "y", "x"]
    axis_units = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}

    # Optimized configuration for large datasets
    chunk_shape = (1, 1, 1024, 1024)  # Large chunks for better I/O with large data
    shard_config = {
        "shard_shape": (1, 10, 2048, 2048),
        "index_location": "start",
    }
    compression_config = {
        "compressor": "zstd",  # Good compression ratio and speed
        "compression_level": 3,  # Fast compression
        "shuffle": True,
    }
    
    ome_zarr_image = OmeZarrImage(
        path=output_path,
        image=image,
        dims=dims,
        axis_units=axis_units,
        downscale_levels=4,  # More levels for large images
        chunk_shape=chunk_shape,
        sharding=shard_config,
        compression=compression_config,
        overwrite=True,
    )

    print(f"Image shape: {image.shape}")
    print(f"Chunk shape: {chunk_shape}")
    print(f"Shard config: {shard_config}")
    print(f"Compression: {compression_config}")
    print(f"Writing optimized example to: {output_path}")
    ome_zarr_image.write()
    print("✓ Optimized example completed")


def print_file_sizes():
    """Print file sizes for comparison."""
    print("=" * 60)
    print("File size comparison:")
    
    examples = [
        "example-basic.zarr",
        "example-chunked.zarr", 
        "example-compressed.zarr",
        "example-sharded.zarr",
        "example-optimized.zarr"
    ]
    
    for example in examples:
        path = Path(example)
        if path.exists():
            size_mb = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / (1024*1024)
            print(f"  {example:25} {size_mb:8.2f} MB")
        else:
            print(f"  {example:25} {'Not found':>8}")


def main():
    """Create multiple OME-Zarr examples demonstrating different configurations."""
    print("OME-Zarr Writer Examples: Chunking, Sharding, and Compression")
    print("=" * 60)
    
    # Create different examples
    create_basic_example()
    create_chunked_example()
    create_compressed_example()
    create_sharded_example()
    create_optimized_example()
    
    # Compare file sizes
    print_file_sizes()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("\nConfiguration guidelines:")
    print("- Chunking: Balance memory usage and I/O patterns")
    print("- Compression: Use blosc/zstd for scientific data")
    print("- Sharding: Group chunks for better file system performance")
    print("- For large datasets: Use larger chunks and sharding")


if __name__ == "__main__":
    main()

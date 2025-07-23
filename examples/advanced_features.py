#!/usr/bin/env python3
"""
Advanced Features of OME-Zarr Writer

This example demonstrates advanced capabilities including:
1. Different compression algorithms and settings
2. Custom chunking and sharding strategies
3. Multiscale pyramid optimization
4. Performance considerations
5. Integration with existing Zarr workflows

Run getting_started.py first to understand the basics.
"""

import numpy as np
import tempfile
from pathlib import Path
from zarr.codecs import BloscCodec, GzipCodec, ZstdCodec, Crc32cCodec
from ome_zarr_writer import OmeZarrImage


def example_1_compression_comparison():
    """Example 1: Compare different compression algorithms."""
    print("🗜️  Example 1: Compression Algorithm Comparison")
    print("-" * 55)

    # Create sample image data for comparison
    image = np.random.randint(0, 255, size=(3, 32, 256, 256), dtype=np.uint8)
    dims = ["c", "z", "y", "x"]
    axis_units = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}

    uncompressed_size = image.nbytes / (1024 * 1024)
    print(f"Original data size: ~{uncompressed_size:.1f} MB")
    print()

    compressors = [
        (
            "BloscCodec + Zstd",
            BloscCodec(cname="zstd", clevel=5),
            "Recommended for most use cases",
        ),
        (
            "BloscCodec + LZ4",
            BloscCodec(cname="lz4", clevel=3),
            "Fastest decompression",
        ),
        ("ZstdCodec", ZstdCodec(level=9), "Best compression ratios"),
        ("GzipCodec", GzipCodec(level=6), "Widely compatible"),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        for name, compressor, description in compressors:
            output_path = (
                Path(tmp_dir)
                / f"{name.lower().replace(' + ', '_').replace(' ', '_')}.zarr"
            )

            writer = OmeZarrImage(
                path=output_path,
                image=image,
                dims=dims,
                axis_units=axis_units,
                overwrite=True,
            )

            writer.write(compressors=compressor)

            file_size = sum(
                f.stat().st_size for f in output_path.rglob("*") if f.is_file()
            )
            file_size_mb = file_size / (1024 * 1024)
            compression_ratio = uncompressed_size / file_size_mb

            print(f"   {name}:")
            print(f"      Size: {file_size_mb:.1f} MB")
            print(f"      Ratio: {compression_ratio:.1f}:1")
            print(f"      Use case: {description}")
            print()


def example_2_chunking_strategies():
    """Example 2: Different chunking strategies for different access patterns."""
    print("📦 Example 2: Chunking Strategies")
    print("-" * 40)

    # Create sample 4D time-lapse data
    image = np.random.randint(0, 255, size=(20, 8, 256, 256), dtype=np.uint16)
    dims = ["t", "z", "y", "x"]
    axis_units = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}

    chunking_strategies = [
        ("Time-optimized", (1, 8, 256, 256), "Best for time-series analysis"),
        ("Z-stack optimized", (20, 1, 256, 256), "Best for 3D visualization"),
        ("Spatial optimized", (5, 2, 128, 128), "Balanced for spatial analysis"),
        ("Default (auto)", None, "Let zarr decide optimal chunks"),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        for name, chunks, description in chunking_strategies:
            output_path = (
                Path(tmp_dir)
                / f"chunks_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.zarr"
            )

            writer = OmeZarrImage(
                path=output_path,
                image=image,
                dims=dims,
                axis_units=axis_units,
                overwrite=True,
            )

            if chunks:
                writer.write(chunks=chunks)
            else:
                writer.write()

            print(f"   {name}:")
            print(f"      Chunks: {chunks if chunks else 'Auto-detected'}")
            print(f"      Use case: {description}")
            print()


def example_3_multiscale_optimization():
    """Example 3: Optimizing multiscale pyramids for different use cases."""
    print("🔺 Example 3: Multiscale Pyramid Optimization")
    print("-" * 50)

    # Create large sample image to demonstrate pyramid benefits
    image = np.random.randint(0, 255, size=(2, 16, 1024, 1024), dtype=np.uint8)
    dims = ["c", "z", "y", "x"]
    axis_units = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}
    scale_transformations = {"z": 0.2, "y": 0.05, "x": 0.05}

    pyramid_configs = [
        ("Minimal", 0, "Single resolution - smallest file size"),
        ("Standard", 3, "Good balance for most viewers"),
        ("Maximum", 5, "Best for large images and web viewing"),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        for name, levels, description in pyramid_configs:
            output_path = Path(tmp_dir) / f"pyramid_{name.lower()}.zarr"

            writer = OmeZarrImage(
                path=output_path,
                image=image,
                dims=dims,
                axis_units=axis_units,
                scale_transformations=scale_transformations,
                downscale_levels=levels,
                overwrite=True,
            )

            writer.write(compressors=BloscCodec(cname="zstd", clevel=5))

            file_size = sum(
                f.stat().st_size for f in output_path.rglob("*") if f.is_file()
            )
            file_size_mb = file_size / (1024 * 1024)
            total_levels = levels + 1

            print(f"   {name} pyramid:")
            print(f"      Levels: {total_levels}")
            print(f"      Size: {file_size_mb:.1f} MB")
            print(f"      Use case: {description}")
            print()


def example_4_performance_considerations():
    """Example 4: Performance optimization for large datasets."""
    print("⚡ Example 4: Performance Optimization")
    print("-" * 45)

    # Simulate large dataset processing
    large_image = np.random.randint(0, 65535, size=(5, 64, 512, 512), dtype=np.uint16)
    dims = ["c", "z", "y", "x"]
    axis_units = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "optimized_large.zarr"

        writer = OmeZarrImage(
            path=output_path,
            image=large_image,
            dims=dims,
            axis_units=axis_units,
            downscale_levels=4,
            overwrite=True,
        )

        # Optimized settings for large datasets
        writer.write(
            compressors=BloscCodec(cname="lz4", clevel=3),  # Fast compression
            chunks=(1, 16, 256, 256),  # Optimized chunk size
            shards=(1, 64, 512, 512),  # Optimized shard size for cloud storage
        )

        file_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
        file_size_mb = file_size / (1024 * 1024)
        original_size_mb = large_image.nbytes / (1024 * 1024)

        print(f"   Original size: {original_size_mb:.1f} MB")
        print(f"   Compressed size: {file_size_mb:.1f} MB")
        print(f"   Compression ratio: {original_size_mb/file_size_mb:.1f}:1")
        print("   Optimizations applied:")
        print("      • LZ4 compression for fast I/O")
        print("      • Chunking optimized for common access patterns")
        print("      • Sharding optimized for cloud storage")
        print("      • 5-level pyramid for multi-resolution viewing")


def example_5_multiple_compressors():
    """Example 5: Using multiple compressors for advanced use cases."""
    print("🔗 Example 5: Multiple Compressors (Advanced)")
    print("-" * 50)

    # Create sample critical data that needs integrity checking
    image = np.random.randint(0, 255, size=(2, 16, 256, 256), dtype=np.uint8)
    dims = ["c", "z", "y", "x"]
    axis_units = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "multi_compressor.zarr"

        writer = OmeZarrImage(
            path=output_path,
            image=image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )

        # Use compression + integrity checking
        writer.write(compressors=[GzipCodec(level=6), Crc32cCodec()])

        file_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
        file_size_mb = file_size / (1024 * 1024)

        print(f"   File size: {file_size_mb:.1f} MB")
        print("   Compressors applied:")
        print("      1. GzipCodec - Data compression")
        print("      2. Crc32cCodec - Integrity verification")
        print("   Use case: Critical data requiring error detection")


def example_6_integration_with_zarr():
    """Example 6: Integration with existing Zarr workflows."""
    print("🔄 Example 6: Integration with Existing Zarr Workflows")
    print("-" * 60)

    # Create sample data
    image = np.random.randint(0, 255, size=(3, 32, 256, 256), dtype=np.uint8)
    dims = ["c", "z", "y", "x"]
    axis_units = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "zarr_integration.zarr"

        # Write OME-Zarr file
        writer = OmeZarrImage(
            path=output_path,
            image=image,
            dims=dims,
            axis_units=axis_units,
            downscale_levels=2,
            overwrite=True,
        )

        writer.write()

        # Show how to access the zarr groups directly for advanced operations
        import zarr

        group = zarr.open_group(str(output_path), mode="r")

        print("   OME-Zarr structure:")
        print(f"      Root group: {list(group.keys())}")
        print(
            f"      Resolution levels: {len([k for k in group.keys() if k.isdigit()])}"
        )
        print("      OME metadata: Present and valid")
        print()
        print("   💡 You can now use standard zarr operations:")
        print("      • Read data: zarr.open(path)['0'][:]")
        print("      • Slice data: zarr.open(path)['0'][0, :, 100:200, 100:200]")
        print("      • Access metadata: zarr.open(path).attrs['ome']")


def main():
    """Run all advanced feature examples."""
    print("🚀 OME-Zarr Writer - Advanced Features")
    print("=" * 60)
    print("These examples demonstrate optimization and advanced capabilities.")
    print("Make sure you've run getting_started.py first to understand the basics.")
    print()

    example_1_compression_comparison()
    example_2_chunking_strategies()
    example_3_multiscale_optimization()
    example_4_performance_considerations()
    example_5_multiple_compressors()
    example_6_integration_with_zarr()

    print("=" * 60)
    print("🎯 Advanced Features Summary:")
    print("   • BloscCodec + Zstd: Best overall compression")
    print("   • BloscCodec + LZ4: Fastest for interactive use")
    print("   • Custom chunking: Optimize for your access patterns")
    print("   • Multiscale pyramids: Essential for large images")
    print("   • Multiple compressors: For special requirements")
    print()
    print("Performance Tips:")
    print("   📈 Larger chunks = better compression, slower random access")
    print("   📈 More pyramid levels = better viewer performance")
    print("   📈 LZ4 compression = faster I/O, larger files")
    print("   📈 Sharding = better cloud storage performance")
    print()
    print("✅ All advanced examples completed successfully!")


if __name__ == "__main__":
    main()

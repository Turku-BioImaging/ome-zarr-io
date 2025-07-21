#!/usr/bin/env python3
"""
Example demonstrating how to use the compressors parameter in OmeZarrImage.write().

This example shows various ways to specify compression when writing OME-Zarr files,
including single compressors, multiple compressors, and different compression algorithms.
"""

import numpy as np
import tempfile
from pathlib import Path
from zarr.codecs import BloscCodec, GzipCodec, ZstdCodec, Crc32cCodec
from ome_zarr_writer import OmeZarrImage


def main():
    """Demonstrate different compressor options."""
    print("🗜️  OME-Zarr Writer - Compressors Example")
    print("=" * 50)
    
    # Create sample image data
    image = np.random.randint(0, 255, size=(3, 64, 256, 256), dtype=np.uint8)
    dims = ["c", "z", "y", "x"]
    axis_units = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}
    
    print(f"Sample image shape: {image.shape} ({dims})")
    print(f"Data type: {image.dtype}")
    print(f"Uncompressed size: ~{image.nbytes / (1024*1024):.1f} MB")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        
        # Example 1: BloscCodec (fast compression with good ratios)
        print("\n📦 Example 1: BloscCodec (recommended for most use cases)")
        output_path1 = Path(tmp_dir) / "blosc_compression.zarr"
        
        writer1 = OmeZarrImage(
            path=output_path1,
            image=image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )
        
        # BloscCodec with zstd algorithm
        writer1.write(compressors=BloscCodec(cname="zstd", clevel=5))
        
        file_size1 = sum(f.stat().st_size for f in output_path1.rglob('*') if f.is_file())
        print(f"   Output: {output_path1.name}")
        print(f"   Compressed size: ~{file_size1 / (1024*1024):.1f} MB")
        print(f"   Compression ratio: {image.nbytes / file_size1:.1f}:1")
        
        # Example 2: ZstdCodec (excellent compression ratios)
        print("\n📦 Example 2: ZstdCodec (best compression ratios)")
        output_path2 = Path(tmp_dir) / "zstd_compression.zarr"
        
        writer2 = OmeZarrImage(
            path=output_path2,
            image=image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )
        
        # ZstdCodec with high compression level
        writer2.write(compressors=ZstdCodec(level=9))
        
        file_size2 = sum(f.stat().st_size for f in output_path2.rglob('*') if f.is_file())
        print(f"   Output: {output_path2.name}")
        print(f"   Compressed size: ~{file_size2 / (1024*1024):.1f} MB")
        print(f"   Compression ratio: {image.nbytes / file_size2:.1f}:1")
        
        # Example 3: GzipCodec (widely compatible)
        print("\n📦 Example 3: GzipCodec (widely compatible)")
        output_path3 = Path(tmp_dir) / "gzip_compression.zarr"
        
        writer3 = OmeZarrImage(
            path=output_path3,
            image=image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )
        
        # GzipCodec with medium compression level
        writer3.write(compressors=GzipCodec(level=6))
        
        file_size3 = sum(f.stat().st_size for f in output_path3.rglob('*') if f.is_file())
        print(f"   Output: {output_path3.name}")
        print(f"   Compressed size: ~{file_size3 / (1024*1024):.1f} MB")
        print(f"   Compression ratio: {image.nbytes / file_size3:.1f}:1")
        
        # Example 4: Multiple compressors (advanced usage)
        print("\n📦 Example 4: Multiple compressors (GzipCodec + Crc32cCodec)")
        output_path4 = Path(tmp_dir) / "multi_compression.zarr"
        
        writer4 = OmeZarrImage(
            path=output_path4,
            image=image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )
        
        # Multiple compressors: compression + integrity check
        writer4.write(compressors=[GzipCodec(level=4), Crc32cCodec()])
        
        file_size4 = sum(f.stat().st_size for f in output_path4.rglob('*') if f.is_file())
        print(f"   Output: {output_path4.name}")
        print(f"   Compressed size: ~{file_size4 / (1024*1024):.1f} MB")
        print(f"   Compression ratio: {image.nbytes / file_size4:.1f}:1")
        print("   Note: Includes CRC32C checksum for data integrity")
        
        # Example 5: With multiscale downscaling
        print("\n📦 Example 5: Compressor with multiscale downscaling")
        output_path5 = Path(tmp_dir) / "multiscale_compressed.zarr"
        
        writer5 = OmeZarrImage(
            path=output_path5,
            image=image,
            dims=dims,
            axis_units=axis_units,
            downscale_levels=3,  # Create 3 downscale levels
            overwrite=True,
        )
        
        # BloscCodec with LZ4 for fast decompression
        writer5.write(compressors=BloscCodec(cname="lz4", clevel=3))
        
        file_size5 = sum(f.stat().st_size for f in output_path5.rglob('*') if f.is_file())
        print(f"   Output: {output_path5.name}")
        print(f"   Compressed size: ~{file_size5 / (1024*1024):.1f} MB")
        print("   Total resolution levels: 4 (original + 3 downscaled)")
        print("   Note: All levels use the same compression settings")
        
        # Example 6: No compression (for comparison)
        print("\n📦 Example 6: No compression (default zarr behavior)")
        output_path6 = Path(tmp_dir) / "no_compression.zarr"
        
        writer6 = OmeZarrImage(
            path=output_path6,
            image=image,
            dims=dims,
            axis_units=axis_units,
            overwrite=True,
        )
        
        # Write without specifying compressors (uses zarr defaults)
        writer6.write()
        
        file_size6 = sum(f.stat().st_size for f in output_path6.rglob('*') if f.is_file())
        print(f"   Output: {output_path6.name}")
        print(f"   Size: ~{file_size6 / (1024*1024):.1f} MB")
        print("   Note: Uses zarr's default compression settings")
        
        print("\n🎯 Compression Recommendations:")
        print("   • BloscCodec with 'zstd': Best overall balance of speed and compression")
        print("   • ZstdCodec: Best compression ratios for archival storage")
        print("   • BloscCodec with 'lz4': Fastest decompression for interactive use")
        print("   • GzipCodec: Most widely compatible across different systems")
        print("   • Multiple compressors: Advanced usage for specific requirements")
        
        print("\n✅ All compressor examples completed successfully!")


if __name__ == "__main__":
    main()

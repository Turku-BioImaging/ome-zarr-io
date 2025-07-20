#!/usr/bin/env python3
"""Test the write method with chunks and shards parameters."""

import numpy as np
from pathlib import Path
from ome_zarr_writer.image import OmeZarrImage
from ome_zarr_writer.schema_models import ScaleTransformation, TranslationTransformation

def test_chunks_and_shards():
    """Test the write method with custom chunks and shards."""
    print("🧪 Testing chunks and shards functionality...")
    
    try:
        # Create test image (4D: channels, z, y, x)
        image = np.random.randint(0, 255, size=(3, 10, 256, 256), dtype=np.uint16)
        dims = ['c', 'z', 'y', 'x']
        axis_units = {'unit': 'micrometer'}
        
        print(f"Created test image: {image.shape}")
        
        # Define coordinate transformations
        coord_transforms = [
            ScaleTransformation(scale=[1.0, 0.5, 0.1, 0.1]),  # c=1, z=0.5μm, y=0.1μm, x=0.1μm
            TranslationTransformation(translation=[0.0, 0.0, 5.0, 5.0])  # 5μm offset in Y,X
        ]
        
        # Save to repository directory for inspection
        output_path = Path('output_chunks_shards_test.zarr')
        print(f"Output path: {output_path.absolute()}")
        
        # Create OmeZarrImage instance with custom chunks and shards
        ome_zarr_image = OmeZarrImage(
            path=output_path,
            image=image,
            dims=dims,
            axis_units=axis_units,
            coordinate_transformations=coord_transforms,
            downscale_levels=2,
            downscale_factor=2,
            overwrite=True,
            chunks=(1, 5, 128, 128),  # Custom chunk size: 1 channel, 5 z-slices, 128x128 xy
            shards=(3, 10, 256, 256)  # Custom shard size: full volume
        )
        print("OmeZarrImage instance created with custom chunks and shards")
        
        # Write the image
        ome_zarr_image.write()
        print("Write method completed")
        
        # Verify the output
        import zarr
        group = zarr.open_group(str(output_path), mode='r')
        print(f"Group keys: {list(group.keys())}")
        
        # Check each level and their chunk/shard configuration
        for level in ['0', '1', '2']:
            if level in group:
                level_array = group[level]
                print(f"Level {level}:")
                print(f"  - Shape: {level_array.shape}")
                print(f"  - Dtype: {level_array.dtype}")
                print(f"  - Chunks: {level_array.chunks}")
                if hasattr(level_array, 'shards') and level_array.shards:
                    print(f"  - Shards: {level_array.shards}")
                print(f"  - Size: {level_array.nbytes / 1024:.1f} KB")
        
        # Check metadata
        print(f"\nHas ome metadata: {'ome' in group.attrs}")
        
        if 'ome' in group.attrs:
            ome_meta = group.attrs['ome']
            if isinstance(ome_meta, dict):
                print(f"OME version: {ome_meta.get('version', 'unknown')}")
                multiscales = ome_meta.get('multiscales', [])
                if multiscales:
                    ms = multiscales[0]
                    datasets = ms.get('datasets', [])
                    print(f"Number of datasets: {len(datasets)}")
                    print(f"Number of axes: {len(ms.get('axes', []))}")
                    
                    # Check coordinate transformations
                    print("\nCoordinate transformations:")
                    for i, dataset in enumerate(datasets):
                        transforms = dataset.get('coordinateTransformations', [])
                        print(f"  Level {i}: {len(transforms)} transform(s)")
                        for j, transform in enumerate(transforms):
                            if transform['type'] == 'scale':
                                print(f"    Scale: {transform['scale']}")
                            elif transform['type'] == 'translation':
                                print(f"    Translation: {transform['translation']}")
        
        print("\n✅ Chunks and shards test completed successfully!")
        print(f"📁 Output saved to: {output_path.absolute()}")
        print("💡 You can inspect the zarr configuration:")
        print(f"   - zarr info {output_path}")
        print(f"   - Python: zarr.open_group('{output_path}', mode='r')")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_chunks_and_shards()

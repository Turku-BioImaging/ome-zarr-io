from pathlib import Path
from ome_zarr_io import Writer
import argparse
from skimage import io
import numpy as np
import zarr.codecs

parser = argparse.ArgumentParser()
parser.add_argument("--source-tiff", type=str, required=True)
args = parser.parse_args()

img = io.imread(args.source_tiff)

writer = Writer(
    path="test.ome.zarr",
    image=np.expand_dims(np.invert(img), axis=0),
    dims=["c", "z", "y", "x"],
    axis_units={"z": "pixel", "y": "pixel", "x": "pixel"},
    downscale_method="mean",
    downscale_levels=4,
    channels={"em_raw": {"color": "ffffff", "window": (0, 65535)}},
    overwrite=True,
)

writer.write(
    chunks=(1, 5, 256, 256),
    shards=(1, 10, 2048, 2048),
    compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=4, shuffle="shuffle"),
)

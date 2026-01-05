"""
Generic test for visual quality of generated OME-Zarr files.
"""
from skimage import data

from ome_zarr_writer import OmeZarrImage

array = data.kidney()
array = array[8, :, :, 0]
image = OmeZarrImage(
    path="visual_quality.ome.zarr",
    image=array,
    dims=["y", "x"],
    downscale_levels=3,
    overwrite=True,
    axis_units={'y': 'micrometer', 'x': 'micrometer'},
)
image.write(chunks=(256, 256))

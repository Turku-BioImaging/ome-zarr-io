from skimage import data, color
from ome_zarr_io import Writer

img = color.rgb2gray(data.skin())
print(img.shape)

Writer

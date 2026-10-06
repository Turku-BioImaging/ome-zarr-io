# Downscaling methods

Level `L` of the pyramid is `downscale_factor ** L` times smaller than the original in Y and X
(`downscale_factor` must be an integer >= 2; default `2`). Pixels that don't fill a whole block at
the bottom/right edge are dropped at coarser levels.

- `"mean"` (default): averages each block of pixels (a box filter followed by subsampling). Use for intensity images.
- `"nearest"`: takes one pixel per block. Preserves discrete values; use for labels and segmentation masks.
- `"gaussian"`: deprecated alias for `"mean"`.

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from regions import Regions


outerdir = '/Users/administrator/Astro/LLAMA/ALMA/pipeline_cubes/'
name = 'NGC5845'

filename = name + '_12m_co21_strictmask.fits'
regname = 'mask2.reg'

file = outerdir + name + '/' + filename
regfile = outerdir + name + '/' + regname


# Spectral channel range to modify
chan_start = 44
chan_end = 118      # inclusive


# Read cube
with fits.open(file) as hdul:
    data = hdul[0].data.copy()
    header = hdul[0].header.copy()

wcs = WCS(header)


# Read DS9 region
regions = Regions.read(regfile, format='ds9')

if len(regions) == 0:
    raise ValueError("No regions found in DS9 region file.")

# Use the first region
region = regions[0]


# Convert region to a pixel region using the celestial WCS
pixel_region = region.to_pixel(wcs.celestial)


# Make a 2D spatial mask
ny, nx = data.shape[-2:]
spatial_mask = pixel_region.to_mask(mode='center').to_image((ny, nx))

spatial_mask = spatial_mask.astype(bool)


# Apply to selected spectral channels
data[chan_start:chan_end + 1, spatial_mask] = 1
data[:,~spatial_mask] = 0


# Write output
outfile = outerdir + name + '/' + name + '_12m_co21_barolomask.fits'

fits.writeto(
    outfile,
    data,
    header,
    overwrite=True
)

print(f"Modified channels {chan_start}–{chan_end}")
print(f"Pixels inside region set to 1")
print(f"Output: {outfile}")
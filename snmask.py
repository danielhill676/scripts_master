from astropy.io import fits
from reproject import reproject_interp
import numpy as np

mask = 'broad'

image_file = f"/Users/administrator/Astro/LLAMA/ALMA/pipeline_m0/NGC5845/NGC5845_12m_co21_{mask}_mom0.fits"
emap_file = f"/Users/administrator/Astro/LLAMA/ALMA/pipeline_m0/NGC5845/NGC5845_12m_co21_{mask}_emom0.fits"

output_file = f"/Users/administrator/Astro/LLAMA/ALMA/pipeline_m0/NGC5845/NGC5845_12m_co21_{mask}_mom0_masked_manual.fits"

# ----------------------------
# Read image
# ----------------------------
with fits.open(image_file) as hdul:
    image = hdul[0].data.squeeze().astype(float)
    image_header = hdul[0].header

# ----------------------------
# Read error map
# ----------------------------
with fits.open(emap_file) as hdul:
    emap = hdul[0].data.squeeze().astype(float)
    emap_header = hdul[0].header

# ----------------------------
# Reproject error map onto the
# image WCS and pixel grid
# ----------------------------
emap_reproj, footprint = reproject_interp(
    (emap, emap_header),
    image_header
)

# Pixels outside the original error map become NaN
emap_reproj[footprint == 0] = np.nan

# ----------------------------
# Apply S/N mask
# ----------------------------
sn_mask = (
    np.isfinite(image) &
    np.isfinite(emap_reproj) &
    (np.abs(image) > 22 * np.abs(emap_reproj))
)



image_masked = image.copy()
image_masked[~sn_mask] = 0.0

noise_manual = 4.0

image_masked_manual = image.copy()

sn_mask = np.abs(image_masked_manual) > 3 * noise_manual
image_masked_manual[~sn_mask] = 0.0

# ----------------------------
# Write output
# ----------------------------
fits.writeto(
    output_file,
    image_masked_manual,
    image_header,
    overwrite=True
)

print(f"Saved masked image to:\n{output_file}")
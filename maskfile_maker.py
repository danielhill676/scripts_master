import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from regions import Regions
from reproject import reproject_interp


outerdir = '/Users/administrator/Astro/LLAMA/ALMA/pymakeplots/'
name = 'NGC5845'

# Mask to modify
filename = '_mask.fits'

# FITS file defining the desired output grid
template_filename = name + '_12m_co21_strictmask.fits'

regname = 'emission_region.reg'

file = outerdir + name + '/' + filename
template_file = outerdir + name + '/' + template_filename
template_file = '/Users/administrator/Astro/LLAMA/ALMA/pipeline_cubes/NGC5845/NGC5845_12m_co21_strictmask_backup.fits'
regfile = outerdir + name + '/' + regname


# ------------------------------------------------------------
# Read input mask
# ------------------------------------------------------------

with fits.open(file) as hdul:
    data = hdul[0].data.copy()
    header = hdul[0].header.copy()

input_wcs = WCS(header)


# ------------------------------------------------------------
# Read template
# ------------------------------------------------------------

with fits.open(template_file) as hdul:
    template_data = hdul[0].data
    template_header = hdul[0].header.copy()

template_wcs = WCS(template_header)

template_shape = template_data.shape


# Check that we have a 3D cube
if data.ndim != 3 or len(template_shape) != 3:
    raise ValueError(
        f"Both input and template must be 3D cubes. "
        f"Got {data.shape} and {template_shape}"
    )


print(f"Input shape:    {data.shape}")
print(f"Template shape: {template_shape}")


# ------------------------------------------------------------
# Create spatial region mask on INPUT grid
# ------------------------------------------------------------

regions = Regions.read(regfile, format='ds9')

if len(regions) == 0:
    raise ValueError(
        f"No regions found in DS9 region file {regfile}."
    )

region = regions[0]

pixel_region = region.to_pixel(input_wcs.celestial)

ny, nx = data.shape[-2:]

spatial_mask = pixel_region.to_mask(
    mode='center'
).to_image((ny, nx))

spatial_mask = spatial_mask.astype(bool)


# ------------------------------------------------------------
# Apply spatial region mask to INPUT cube
# ------------------------------------------------------------

data[:, ~spatial_mask] = 0


# ------------------------------------------------------------
# Reproject spatially + spectrally onto template grid
# ------------------------------------------------------------

# Output array is automatically initialised to zero.
reprojected = np.zeros(template_shape, dtype=data.dtype)


# WCS for the spatial axes
input_celestial_wcs = input_wcs.celestial
template_celestial_wcs = template_wcs.celestial


# Determine spectral WCS
input_spectral = input_wcs.sub(['spectral'])
template_spectral = template_wcs.sub(['spectral'])


# Spectral coordinates of input and template channels
input_spec = np.arange(data.shape[0])
template_spec = np.arange(template_shape[0])

input_spec_world = input_spectral.pixel_to_world_values(input_spec)
template_spec_world = template_spectral.pixel_to_world_values(template_spec)


# ------------------------------------------------------------
# Find which template channels overlap the input cube
# ------------------------------------------------------------

input_min = np.nanmin(input_spec_world)
input_max = np.nanmax(input_spec_world)

spec_overlap = (
    (template_spec_world >= min(input_min, input_max)) &
    (template_spec_world <= max(input_min, input_max))
)

template_channels = np.where(spec_overlap)[0]


print(
    f"Spectral overlap: "
    f"{len(template_channels)} / {template_shape[0]} template channels"
)


# ------------------------------------------------------------
# Reproject each overlapping spectral channel
# ------------------------------------------------------------

for template_chan in template_channels:

    target_world = template_spec_world[template_chan]

    # Find nearest input spectral channel
    input_chan = np.argmin(
        np.abs(input_spec_world - target_world)
    )

    # Reproject spatially
    reprojected[template_chan], footprint = reproject_interp(
        (
            data[input_chan],
            input_celestial_wcs
        ),
        template_celestial_wcs,
        shape_out=template_shape[-2:],
        order='nearest-neighbor'
    )

    # Pixels outside the input footprint become NaN.
    # Convert these to zero.
    reprojected[template_chan][
        ~np.isfinite(reprojected[template_chan])
    ] = 0


# ------------------------------------------------------------
# Convert back to a binary mask
# ------------------------------------------------------------

reprojected = (reprojected > 0).astype(data.dtype)


# ------------------------------------------------------------
# Save using EXACTLY the template header
# ------------------------------------------------------------

outfile = (
    outerdir + name + '/' +
    name + 'mask_pruned.fits'
)

fits.writeto(
    outfile,
    reprojected,
    template_header,
    overwrite=True
)


print(f"Output shape: {reprojected.shape}")
print(f"Output: {outfile}")
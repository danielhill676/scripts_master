from astropy.io import fits
import numpy as np

name = 'NGC7172'
root = '/Users/administrator/Astro/LLAMA/ALMA/AGN_images/'

with fits.open(root+name+'_moment_0.fits') as hdu:
    moment0 = hdu[0].data
print(moment0.shape)

with fits.open(root+name+'.sigma.fits') as hdu:
    sigma = hdu[0].data
print(sigma.shape)

with fits.open('/Users/administrator/Astro/LLAMA/ALMA/AGN/NGC7172.fits') as hdu:
    cube = hdu[0].data
print(cube.shape)

mask = moment0 > sigma
mask = mask.astype(np.int16)
print(mask.shape)

cube[mask == 0] = np.nan

cube_hdu = fits.PrimaryHDU(cube)
path = root+name+'.mom1_mask_cube.fits'
cube_hdu.writeto(path, overwrite=True)
print(f"cube saved to {path}")


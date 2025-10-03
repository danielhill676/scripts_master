


from astropy.io import fits
import numpy as np

with fits.open('/Users/administrator/Astro/LLAMA/ALMA/AGN_images/NGC3081.moment.standard_deviation.fits') as hdul:
    
    data = hdul[0].data
    
    data *=500
    print(data.shape)

    # new_hdu = fits.PrimaryHDU(np.squeeze(data))
    
    new_hdu = fits.PrimaryHDU(np.squeeze(np.full(data.shape,0.1)))

    new_file_path = '/Users/administrator/Astro/LLAMA/ALMA/AGN_images/test_sigma.fits'
    new_hdu.writeto(new_file_path, overwrite=True,)


    # mask = not np.isfinite(data) or np.isfinite(sigma)

    # sigma should only be for non line channels

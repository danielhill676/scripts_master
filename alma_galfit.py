from astroquery.alma import Alma
from astropy.io import fits
import os

# alma = Alma()

# alma.login("danielhill", store_password=True)


def galfit_header(folder,outfolder):

    for name in os.listdir(folder):
            name_short = name.removesuffix('.fits')

            if name == '.DS_Store':    
                continue

        # data = alma.query_object(f'{name_short}')

        # for i in range(len(data)):
        #     if (data['member_ous_uid'][i] == 'uid://A001/X2fe/X665'):
        #         EXPTIME = data['t_exptime'][i]
        #         break

            GAIN = 1.0
            RDNOISE = 0.0
            NCOMBINE = 1
            EXPTIME = 1000

            os.chdir(f'{outfolder}')
            data, header = fits.getdata(f"{name_short}_moment_0.fits", header=True)

            header['EXPTIME'] = EXPTIME
            header['GAIN'] = GAIN
            header['RDNOISE'] = RDNOISE
            header['NCOMBINE'] = NCOMBINE
            fits.writeto(f'{name_short}_moment_0_galfit.fits', data, header, overwrite=True)

galfit_header('/Users/administrator/Astro/LLAMA/ALMA/AGN','/Users/administrator/Astro/LLAMA/ALMA/AGN_images')
galfit_header('/Users/administrator/Astro/LLAMA/ALMA/inactive','/Users/administrator/Astro/LLAMA/ALMA/inactive_images')
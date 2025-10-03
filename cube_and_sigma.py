import matplotlib.pyplot as plt
import math
import os
import time
import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from spectral_cube import SpectralCube
from astropy.table import Table
from regions import CircleSkyRegion # type: ignore
from regions import RectangleSkyRegion # type: ignore
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm
from astropy.visualization.wcsaxes import add_scalebar
from astropy.visualization.wcsaxes import add_beam
from astropy.stats import sigma_clipped_stats
from astropy.io import fits
from astropy.constants import c
import warnings
from astroquery.ipac.ned import Ned

warnings.filterwarnings("ignore", category=UserWarning, message="WCS1 is missing card PV2_2")
warnings.filterwarnings("ignore", category=UserWarning, message="WCS1 is missing card PV2_1")
warnings.filterwarnings('ignore', message='WCS1 is missing card TIMESYS')



st=time.time()

llamatab = Table.read('/Users/administrator/Astro/LLAMA/llama_main_properties.fits',format='fits')
#llamatab.show_in_browser()

def cube_imaging(folder,restf,dv,outfolder,FOV,square=False):
    """folder = path to .fits location
        zfile = path to csv of object redshift in order of processing
        restf = line rest frequency GHz
        dv = ± velocity km/s
        outfolder = desired path to output plots
        square = True if square region, False if circular
        FOV = field of view in pc
        """

    for name in os.listdir(folder):
        name_short = name.removesuffix('.fits')

        if name == '.DS_Store':    
            continue

        # if name_short not in ['ESO137']:
        #     continue

        # if name_short != 'NGC4388':
        #     continue

        # Read in spectral cube    
        print(f"running {name}")
        cube = SpectralCube.read(f"{folder}/{name}")
        cube.allow_huge_operations=True

        # Get important parameters from llama table and fits header
        for i in range(len(llamatab)):
            if llamatab['id'][i] == f'{name_short}':
                D = llamatab['D [Mpc]'][i]
                z = llamatab['redshift'][i]
                Ned_table = Ned.query_object(llamatab['name'][i])
                RA = Ned_table['RA'][0]
                DEC = Ned_table['DEC'][0]



        
        with fits.open(f'{folder}/{name}') as hdul:
             header = hdul[0].header
        pixel_x = header['CDELT2']
        pixel_y = header['CDELT1']

        # Perform spectral masking for moment maps and sigma image
        restfreq = restf/(1+z) * u.GHz
        cube2=cube.with_spectral_unit(u.km / u.s, velocity_convention='radio', rest_value=restfreq)
        del cube # free up memory
        print('cube shape:',cube2.shape)
        slab = cube2.spectral_slab(-dv * u.km / u.s, +dv * u.km / u.s)
        print('applying mask')
        mask = (cube2.spectral_axis < -dv * u.km/ u.s ) | (cube2.spectral_axis > dv * u.km/ u.s)
        print('mask shape:',mask.shape)
        expanded_mask = mask[:, np.newaxis, np.newaxis]
        expanded_mask = np.broadcast_to(expanded_mask, cube2.shape)
        print('expanded linemask shape', expanded_mask.shape)
        mask_slab = cube2.with_mask(expanded_mask)
        print('done applying mask')

        # Perform spacial masking for moment maps and sigma image
        gal_cen = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs')
        theta = math.degrees(math.atan(FOV/(D*1e6))) # in degrees
        region = CircleSkyRegion(center=gal_cen, radius=theta * u.deg)
        # add square region here
        square_region = RectangleSkyRegion(center=gal_cen, width= 2*theta * u.deg,height= 2*theta * u.deg)
        subcube = slab.subcube_from_regions([region])
        if square:
            subcube = slab.subcube_from_regions([square_region])
        mask_subcube = mask_slab.subcube_from_regions([region])
        if square:
            mask_subcube = mask_slab.subcube_from_regions([square_region])
        
        # Calculate sigma image
        print('mask subcube:',mask_subcube.shape)
        N = subcube.shape[0]
        chan_width = 2*dv/N
        sigma_image = (np.nanstd(mask_subcube,axis=0) * chan_width)/np.sqrt(N)  # in Jy/beam km/s
        print('sigma image shape:',sigma_image.shape)

        # Create bad pixel mask
        bad_pixel_mask = np.any(~np.isfinite(subcube), axis=0) | ~np.isfinite(sigma_image)
        # Convert boolean mask to integer type
        bad_pixel_mask = bad_pixel_mask.astype(np.int16).squeeze()
        print('badpix shape',bad_pixel_mask.shape)

        # Create 2-sigma threshold mask for emission
        threshold = 1.0 * sigma_image  # 2 × noise
        # Broadcast threshold to match cube shape
        threshold_cube = np.broadcast_to(threshold, subcube.shape)


        # Apply threshold mask: True = keep pixel
        emission_mask = (np.abs(subcube) > threshold_cube) & np.isfinite(subcube)

        # Mask the cube so moments are only calculated where emission > 2σ
        emission_only_cube = subcube.with_mask(emission_mask)

        # NOTE: experiment with different velocity ranges

        BMAJ = fits.open(f"{folder}/{name}")[0].header['BMAJ']
        BMIN = fits.open(f"{folder}/{name}")[0].header['BMIN']  

        def image_plot(ord, moment0=None):
            """ ord = moment order index """
    
            print(f"constructing moment {ord}...")
            

            if ord == 8:
                moment = subcube.with_spectral_unit(u.km/u.s).max(axis=0)
                moment.write(outfolder+f'/{name_short}_moment_{ord}.fits', overwrite=True)

                # moment_large = slab.with_spectral_unit(u.km/u.s).max(axis=0)
                # moment_large.write(outfolder+f'/{name_short}_moment_{ord}_large.fits', overwrite=True)


            elif ord == 0:
                moment = subcube.with_spectral_unit(u.km/u.s).moment(order=0)
                moment.write(outfolder+f'/{name_short}_moment_{ord}.fits', overwrite=True)


            elif ord in (1, 2):
                if moment0 is None:
                    raise ValueError("Moment 0 must be provided for masking moments 1 and 2.")

                m01_mask = (moment0.value > threshold) & np.isfinite(moment0.value)
                m01_mask_cube = np.broadcast_to(m01_mask, subcube.shape)
                cube_for_moment = subcube.with_mask(m01_mask_cube)

                moment = cube_for_moment.with_spectral_unit(u.km/u.s).moment(order=ord)
                moment.write(outfolder+f'/{name_short}_moment_{ord}.fits', overwrite=True)

            # NOTE: definitions of moments found here: https://spectral-cube.readthedocs.io/en/latest/api/spectral_cube.SpectralCube.html#spectral_cube.SpectralCube.moment
            
            mean,med,sd = sigma_clipped_stats(moment.data)
            min = float(med - 1.5*sd)
            
            print("Moment data shape:", moment.data.shape)
            print("Number of finite pixels:", np.sum(np.isfinite(moment.data)))
            print("Min, max (finite):", np.nanmin(moment.data), np.nanmax(moment.data))

            # Initialise plot
            plt.rcParams.update({'font.size': 35})
            fig = plt.figure(figsize=(18 , 18))
            ax = fig.add_subplot(111, projection=moment.wcs)
            ax.margins(x=0,y=0)
            # ax.set_axis_off()

            if ord == 0:
                add_scalebar(ax,1/3600,label="1''",corner='top left',color='black',borderpad=0.5,size_vertical=0.5)
                add_beam(ax,major=BMAJ,minor=BMIN,angle=0,corner='bottom right',color='black',borderpad=0.5,fill=False,linewidth=3,hatch='///')

            fig.tight_layout()
            if np.isfinite(moment.data).any():
                norm_sqrt = simple_norm(moment.data, 'sqrt', vmin=np.nanmin(moment.data))
            else:
                print("Moment data empty or all NaNs — skipping normalization.")
                # Handle the empty case, e.g., set norm_sqrt = None or use a fallback norm
                norm_sqrt = None
            #plt.title(f'{name_short}',fontsize=75)
            im=plt.imshow(moment.data,origin='lower',norm=norm_sqrt,cmap='RdBu_r')
            # im.axes.get_xaxis().set_visible(False)
            # im.axes.get_yaxis().set_visible(False)

            if np.isfinite(moment.data).any():

                if ord == 0:
                    # fig.colorbar(im, label=r'Surface brightness ($Wsr^{-1}$)') 
                    plt.savefig(outfolder+f'/m0_plots/{name_short}.png',bbox_inches='tight',pad_inches=0.0)
                elif ord == 8:
                    # plt.colorbar(im, label=r'Flux density ($Jy/beam$)')
                    plt.savefig(outfolder+f'/m8_plots/{name_short}.png',bbox_inches='tight',pad_inches=0.0)
                elif ord == 1:
                    # plt.colorbar(im, label=r'Velocity ($km/s$)')
                    plt.savefig(outfolder+f'/m1_plots/{name_short}.png',bbox_inches='tight',pad_inches=0.0)
                elif ord == 2:
                # plt.colorbar(im, label=r'Velocity ($km/s$)')
                    plt.savefig(outfolder+f'/m2_plots/{name_short}.png',bbox_inches='tight',pad_inches=0.0)
                
                if ord == 0:
                    return moment
                else:
                    return None

        moment0_data = image_plot(0)
        image_plot(8)
        image_plot(1, moment0=moment0_data)
        image_plot(2, moment0=moment0_data)

        # Save the sigma image
        sigma_hdu = fits.PrimaryHDU(sigma_image)
        sigma_path = f'{outfolder}/{name_short}.sigma.fits'
        sigma_hdu.writeto(sigma_path, overwrite=True)
        print(f"Sigma image saved to {sigma_path}")

        # Save the bad pixel mask
        bad_pixel_hdu = fits.PrimaryHDU(bad_pixel_mask)
        bad_pixel_path = f'{outfolder}/{name_short}.bad_pixels.fits'
        bad_pixel_hdu.writeto(bad_pixel_path, overwrite=True)
        print(f"Bad pixel mask saved to {bad_pixel_path}")

        if np.isnan(bad_pixel_mask).any():
            print("Bad pixel mask contains NaNs")
        else:
            print("Bad pixel mask does not contain NaNs")



        with open(outfolder+f'/{name_short}_galfit_input','w') as f:
                f.write(f"""
            ================================================================================
            # IMAGE and GALFIT CONTROL PARAMETERS
            A) {name_short}_moment_0_galfit.fits            # Input data image (FITS file)
            B) {name_short}_imblock.fits       # Output data image block
            C) {name_short}.sigma.fits                # Sigma image name (made from data if blank or "none")
            D) none   #        # Input PSF image and (optional) diffusion kernel
            E) 1                   # PSF fine sampling factor relative to data
            F) {name_short}.bad_pixels.fits                # Bad pixel mask (FITS image or ASCII coord list)
            G) none                # File with parameter constraints (ASCII file)
            H)  0 {int(sigma_image.shape[1])} 0 {int(sigma_image.shape[0])}   # Image region to fit (xmin xmax ymin ymax)
            # I)       # Size of the convolution box (x y)
            J) 25             # Magnitude photometric zeropoint
            K) {pixel_x}  {pixel_y}        # Plate scale (dx dy)   [arcsec per pixel]
            O) regular             # Display type (regular, curses, both)
            P) 0                   # Options: 0=normal run; 1,2=make model/imgblock & quit

            # Exponential function

            0) expdisk            # Object type
            1) {int(0.5*sigma_image.shape[1])}  {int(0.5*sigma_image.shape[0])}  1 1     # position x, y        [pixel]
            3) 33       1       # total magnitude
            4) {int(0.25*sigma_image.shape[0])}       1       #     Rs               [Pixels]
            9) 1        1       # axis ratio (b/a)
            10) 0         1       # position angle (PA)  [Degrees: Up=0, Left=90]
            Z) 0                 #  Skip this model in output image?  (yes=1, no=0)')
            """)

        galfit_script_path = outfolder+'/galfit_execute.sh'
        
        if not os.path.exists(galfit_script_path):
            with open(galfit_script_path, 'w') as f:
                f.write("#!/bin/bash\n")
        
        with open(galfit_script_path, 'a') as f:
            f.write(f"galfit {name_short}_galfit_input\n")
            print(f"Appended galfit command for {name_short} to galfit_execute.sh")
        
        # plt.show()          #comment or uncomment for testing
        # break


dv = 250
CO2_1 = 230.538
CO3_2 = 345.796
square = True
FOV = 1000  # in pc

print('Beginning AGN CO imaging...')
AGN_21 = r"/Users/administrator/Astro/LLAMA/ALMA/AGN"
AGN_32 = r"/Users/administrator/Astro/LLAMA/ALMA/CO32/AGN"

cube_imaging(AGN_21,CO2_1,dv,'/Users/administrator/Astro/LLAMA/ALMA/AGN_images',FOV, square=True)
cube_imaging(AGN_32,CO3_2,dv,'/Users/administrator/Astro/LLAMA/ALMA/AGN_images',FOV,square=True)


print('Beginning inactive galaxy CO imaging...')
inactive = r"/Users/administrator/Astro/LLAMA/ALMA/inactive"

cube_imaging(inactive,CO2_1,dv,'/Users/administrator/Astro/LLAMA/ALMA/inactive_images',FOV,square=True)

et=time.time()
runtime=et-st
print('runtime=',runtime)



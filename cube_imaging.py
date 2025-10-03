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
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm
from astropy.visualization.wcsaxes import add_scalebar
from astropy.visualization.wcsaxes import add_beam
from astropy.stats import sigma_clipped_stats
from astropy.io import fits



st=time.time()

llamatab = Table.read('/Users/administrator/Astro/LLAMA/llama_main_properties.fits',format='fits')
#llamatab.show_in_browser()
def cube_imaging(folder,restf,dv,outfolder):
    """folder = path to .fits location
        zfile = path to csv of object redshift in order of processing
        restf = line rest frequency GHz
        dv = ± velocity km/s
        outfolder = desired path to output plots
        """

    for name in os.listdir(folder):
        name_short = name.removesuffix('.fits')

        if name == '.DS_Store':    
            continue

        # if name_short != 'NGC6814':                 #Commment or uncomment this block for testing
        #     continue

        print(f"running {name}")
        cube = SpectralCube.read(f"{folder}/{name}")
        cube.allow_huge_operations=True

        for i in range(len(llamatab)):
            if llamatab['id'][i] == f'{name_short}':
                RA = llamatab['RA (deg)'][i]
                DEC = llamatab['DEC (deg)'][i]
                D = llamatab['D [Mpc]'][i]
                z = llamatab['redshift'][i]

        cube2=cube.with_spectral_unit(u.km / u.s, velocity_convention='radio', rest_value=restf/(1+z)  * u.GHz)
        slab = cube2.spectral_slab(-dv * u.km / u.s, +dv * u.km / u.s)
        gal_cen = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs')
        theta = math.degrees(math.atan(1/(D*1000)))
        region = CircleSkyRegion(center=gal_cen, radius=theta * u.deg)
        subcube = slab.subcube_from_regions([region])
        # NOTE: experiment with different velocity ranges

        BMAJ = fits.open(f"{folder}/{name}")[0].header['BMAJ']
        BMIN = fits.open(f"{folder}/{name}")[0].header['BMIN']  

        def image_plot(ord):
            """ ord = moment order index """
    
            print(f"constructing moment {ord}...")
            if ord == 8:
                moment = subcube.with_spectral_unit(u.km/u.s).max(axis=0)
                moment.write(outfolder+f'/{name_short}_moment_{ord}.fits',overwrite=True)
            else:
                moment = subcube.with_spectral_unit(u.km/u.s).moment(order=ord)
                moment.write(outfolder+f'/{name_short}_moment_{ord}.fits',overwrite=True)
            # NOTE: definitions of moments found here: https://spectral-cube.readthedocs.io/en/latest/api/spectral_cube.SpectralCube.html#spectral_cube.SpectralCube.moment
            
            mean,med,sd = sigma_clipped_stats(moment.data)
            min = float(med - 1.5*sd)
            
            plt.rcParams.update({'font.size': 35})
            fig = plt.figure(figsize=(18 , 18))
            ax = fig.add_subplot(111, projection=moment.wcs)
            ax.margins(x=0,y=0)
            ax.set_axis_off()
            add_scalebar(ax,1/3600,label="1''",corner='top left',color='black',borderpad=0.5,size_vertical=0.5)
            add_beam(ax,major=BMAJ,minor=BMIN,angle=0,corner='bottom right',color='black',borderpad=0.5,fill=False,linewidth=3,hatch='///')
            fig.tight_layout()
            norm_sqrt = simple_norm(moment.data,'sqrt', vmin= min)
            plt.title(f'{name_short}',fontsize=75)
            im=plt.imshow(moment.data,origin='lower',norm=norm_sqrt,cmap='RdBu_r')
            im.axes.get_xaxis().set_visible(False)
            im.axes.get_yaxis().set_visible(False)

            if ord == 0:
                # fig.colorbar(im, label=r'Surface brightness ($Wsr^{-1}$)') 
                plt.savefig(outfolder+f'/m0_plots/{name_short}.png',bbox_inches='tight',pad_inches=0.0)
            elif ord == 8:
                    # plt.colorbar(im, label=r'Flux density ($Jy/beam$)')
                    plt.savefig(outfolder+f'/m8_plots/{name_short}.png',bbox_inches='tight',pad_inches=0.0)

        image_plot(0)
        image_plot(8)
        
        # plt.show()          #comment or uncomment for testing
        # break


dv = 250
CO2_1 = 230.538
CO3_2 = 345.796

print('Beginning AGN CO imaging...')
AGN_21 = r"/Users/administrator/Astro/LLAMA/ALMA/AGN"
AGN_32 = r"/Users/administrator/Astro/LLAMA/ALMA/CO32/AGN"

cube_imaging(AGN_21,CO2_1,dv,'/Users/administrator/Astro/LLAMA/ALMA/AGN_images')
cube_imaging(AGN_32,CO3_2,dv,'/Users/administrator/Astro/LLAMA/ALMA/AGN_images')


print('Beginning inactive galaxy CO imaging...')
inactive = r"/Users/administrator/Astro/LLAMA/ALMA/inactive"

cube_imaging(inactive,CO2_1,dv,'/Users/administrator/Astro/LLAMA/ALMA/inactive_images')

et=time.time()
runtime=et-st
print('runtime=',runtime)



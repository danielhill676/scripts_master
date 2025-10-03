import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from spectral_cube import SpectralCube
import os
import csv
from astropy.table import Table
import math
from regions import CircleSkyRegion
from astropy.coordinates import SkyCoord

llamatab = Table.read('/Users/administrator/Astro/LLAMA/llama_main_properties.fits',format='fits')

def cube_imaging(folder,zfile,restf,dv,outfolder):
    """folder = path to .fits location
        zfile = path to csv of object redshift in order of processing
        restf = line rest frequency GHz
        dv = ± velocity km/s
        outfolder = desired path to output plots
        """

    for name in os.listdir(folder):
        name_short = name.removesuffix('.fits')
        if name_short != 'NGC6814':
            continue
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
        print(f'subcube centre = {RA} RA,{DEC} DEC radius = {theta*3600} arcsec')
        print(f'D = {D} Mpc = {D*1000} kpc, 1/D*1000 = {1/(D*1000)}, theta = {math.atan(1/(D*1000))} deg = {3600*math.atan(1/(D*1000))} arcsec')
        region = CircleSkyRegion(center=gal_cen, radius=theta * u.deg)
        subcube = slab.subcube_from_regions([region])
        # Take subcube for central kpc radius, find galactic centre coordinates.
        # experiment with different velocity ranges

        print("constructing moment 0")
        moment_0 = subcube.with_spectral_unit(u.km/u.s).moment(order=0)
        # check how moment 0 is done, make sure units are correct.

        fig = plt.figure(figsize=(15, 15))
        ax=fig.add_subplot(projection=moment_0.wcs)
        ax.set_xlabel("RA",fontsize=22)
        ax.set_ylabel("Dec",fontsize=22)
        plt.title(f'{name_short}',fontsize=22)
        plt.imshow(moment_0.data)
        cbar = plt.colorbar()
        plt.show()
        # Change scaling (stretch) in order to best accentuate the noise, do this based on statistics of each image (how?)
        break




print('Beginning AGN CO imaging...')
AGN = r"/Users/administrator/Astro/LLAMA/ALMA/AGN"
AGN_z = r"/Users/administrator/Astro/LLAMA/AGN_z_noname.csv"

cube_imaging(AGN,AGN_z,230.538,500,'/Users/administrator/Astro/LLAMA/ALMA/AGN_images')
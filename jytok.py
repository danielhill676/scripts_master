from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt


file = '/Users/administrator/Astro/LLAMA/ALMA/phangs_cubes_for_test/NGC5845_mom0_pymakeplots.fits'
jytok_override = 7.806367792610E+01



with fits.open(file) as hdul:
    data = hdul[0].data
    header = hdul[0].header
try: jytok = header['JYTOK']
except: 
    jytok = jytok_override
    header['JYTOK'] = jytok_override

data_k = data * jytok

fits.writeto('/Users/administrator/Astro/LLAMA/ALMA/phangs_cubes_for_test/NGC5845_mom0_pymakeplots_K.fits', data_k, header, overwrite=True)


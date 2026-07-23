from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt


file = '/Users/administrator/Astro/LLAMA/ALMA/pymakeplots/NGC5845/_mom0.fits'
jytok_override = 7.806367792610E+01



with fits.open(file) as hdul:
    data = hdul[0].data
    header = hdul[0].header
try: jytok = header['JYTOK']
except: 
    jytok = jytok_override
    header['JYTOK'] = jytok_override

data_k = data * jytok
header['BUNIT'] = 'K.km.s-1'

fits.writeto('/Users/administrator/Astro/LLAMA/ALMA/pipeline_m0/NGC5845/NGC5845_12m_co21_broad_mom0.fits', data_k, header, overwrite=True)


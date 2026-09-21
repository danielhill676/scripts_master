from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt


file = '/Users/administrator/Astro/LLAMA/ALMA/pipeline_m0/NGC4429/NGC4429_12m_co21_broad_mom0.fits'
jytok_override = 7.806367792610E+01



with fits.open(file) as hdul:
    data = hdul[0].data
    header = hdul[0].header
try: jytok = header['JYTOK']
except:
    try:
        print('deriving jytok from other properties')
        bmaj = header['BMAJ'] * 3600
        bmin = header['BMIN'] * 3600
        restfreq = header['RESTFRQ'] /1e9
        jytok = 1.222e6 / (restfreq**2 * bmaj * bmin)
    except:
        print('cannot calculated jytok')
        raise ValueError("cannot calculated jytok")
        # print('using override !')
        # jytok = jytok_override
        # header['JYTOK'] = jytok_override

print('jytok=',jytok)
data_k = data * jytok
header['BUNIT'] = 'K.km.s-1'

fits.writeto('/Users/administrator/Astro/LLAMA/ALMA/pipeline_m0/NGC4429/NGC4429_12m_co32_strict_mom0.fits', data_k, header, overwrite=True)


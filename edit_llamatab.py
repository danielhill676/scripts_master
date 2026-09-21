from astropy.table import Table
import numpy as np


llamatab = Table.read('/Users/administrator/Astro/LLAMA/llama_main_properties.fits', format='fits')
print(llamatab)

RA_3368 = (10+46/60+45.6555327440/3600)*360/24 
DEC_3368 = (11 + 49/60 + 11.9591577548/3600) 

RA_4429 = (12+27/60+26.5022097065/3600)*360/24 
DEC_4429 = (11 + 6/60 + 27.6003996800/3600) 


llamatab.add_row([
    'NGC3368',   # id
    'NGC 3368',   # name
    18.0,          # D [Mpc]
    'i',           # type
    10.67,          # log Mstar
    RA_3368,             # RA (deg)
    DEC_3368,            # DEC (deg)
    0.002962,       # redshift
    np.nan,          # D25_maj (arcsec)
    np.nan,         # D25_min (arcsec)
    165.0,           # PA
    60.0,           # Inclination (deg) 
])

llamatab.add_row([
    'NGC4429',   # id
    'NGC 4429',   # name
    16.5,          # D [Mpc]
    'i',           # type
    11.17,          # log Mstar
    RA_4429,             # RA (deg)
    DEC_4429,            # DEC (deg)
    0.003683,       # redshift
    np.nan,          # D25_maj (arcsec)
    np.nan,         # D25_min (arcsec)
    93.2,           # PA
    66.8,           # Inclination (deg) 
])
print('\n')
print(llamatab)

llamatab.write(
    '/Users/administrator/Astro/LLAMA/llama_main_properties_newcontrols.fits',
    format='fits',
    overwrite=True
)


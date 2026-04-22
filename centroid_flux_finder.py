from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from photutils.aperture import aperture_photometry
from regions import CircleSkyRegion # type: ignore


# --- USER INPUT ---

names = ['NGC3717','NGC5921','NGC4254','NGC4224','NGC5037','NGC3749']
ras   = [172.88294,230.48537,184.70681,184.14078,198.74743,173.97176]
decs  = [-30.30766,5.07064,14.41649,7.46211,-16.59027,-37.99746]

for name, ra, dec in zip(names, ras, decs):
    print('\n',name, ra, dec)

    c = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    ra_str = c.ra.to_string(unit=u.hour, sep=':', precision=4)
    dec_str = c.dec.to_string(unit=u.deg, sep=':', precision=3, alwayssign=True)

    print(ra_str)
    print(dec_str)



    fits_file = f"/Users/administrator/Astro/LLAMA/ALMA/pipeline_m0/{name}/{name}_12m_co21_broad_mom0.fits"

    # --- OPEN FITS ---
    with fits.open(fits_file) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    # --- HANDLE CUBES (optional) ---
    if data.ndim > 2:
        data = data[0]  # take first slice; adjust if needed

    # --- SET UP WCS ---
    wcs = WCS(header).celestial

    # --- CONVERT RA/DEC TO PIXEL ---
    sky = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    x_pix, y_pix = wcs.world_to_pixel(sky)

    x_int = int(np.round(x_pix))
    y_int = int(np.round(y_pix))
    
    gal_cen = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    aperture1as = CircleSkyRegion(center=gal_cen, radius=(0.5/3600) * u.deg)
    f_phot = aperture_photometry(data, aperture1as, method='exact', wcs=wcs)
    f_sum = f_phot['aperture_sum'][0]


    jytok = header['JYTOK']
    beam = header['BMAJ'] * 3600  # arcsec
    pixel = header['CDELT1']*3600
    beam_new = 0.1                # arcsec

    jybeamkms = f_sum / jytok # in jy/beam km/s over 1'' aperture

    co32_jybeamkms = 0.52 * jybeamkms



    scale = (0.13/1) * (0.13/beam) 
    co32_jybeamkms = co32_jybeamkms * scale # jy/beam km/s over 0.13'' aperture and beam (linear dilution)

    fd_pred_mjybeam_10kmschannel = (co32_jybeamkms/20)*1000  # jy/beam over 0.13'' aperture and 10km/s channel

    print(fd_pred_mjybeam_10kmschannel)

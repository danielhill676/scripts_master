from photutils.aperture import aperture_photometry
import numpy as np
from astropy.io import fits
import time
import requests
from astroquery.ipac.ned import Ned
from astroquery.exceptions import RemoteServiceError
from astroquery.simbad import Simbad
from astropy.table import Table
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
simbad = Simbad()
import astropy.units as u
from regions import CircleSkyRegion # type: ignore
from astropy.coordinates import SkyCoord
from astropy.nddata import NDData
from astropy.wcs import WCS


def SCOdv_single(
    image,
    jy_per_K,        
    beam_area_arcsec2, 
    pixel_area_arcsec2,
    aperture=None
):
    pix_per_beam = np.pi*beam_area_arcsec2/4 / pixel_area_arcsec2

    I_phot = aperture_photometry(image, aperture, method='exact', wcs=wcs)
    I_sum = I_phot['aperture_sum'][0]
    print('I_sum=',I_sum,'K km/s')

    # Convert to total flux (Jy km/s)
    print('F=',(I_sum / jy_per_K),'Jy/beam km/s')
    S_CO_dv = (I_sum / jy_per_K) / pix_per_beam

    return S_CO_dv

def LCO_single(
    image,
    pixel_area_arcsec2,
    beam_area_arcsec2,
    beam_area_pc2,
    R_21,
    R_31,
    alpha_CO,
    name,
    D_Mpc,
    co32=False,
    aperture=None
):
    """
    Total L'CO in K km s^-1 pc^2 (beam-correct, resolution-independent).
    image must be in K km s^-1 (per beam).
    """

    pix_per_beam = beam_area_arcsec2 / pixel_area_arcsec2

    I_phot = aperture_photometry(image, aperture, method='exact', wcs=wcs)
    I_sum = I_phot['aperture_sum'][0]
    Lprime = I_sum * 23.5 * beam_area_arcsec2/pix_per_beam * D_Mpc**2

    if co32:
        Lprime = (Lprime / R_31) * R_21

    return Lprime

def total_mass_single(
    image,
    pixel_area_arcsec2,
    beam_area_arcsec2,
    beam_area_pc2,
    R_21,
    R_31,
    alpha_CO,
    name,
    D_Mpc,
    co32=False,
    aperture=None
):
    """
    Total H2 mass from beam-correct L'CO.
    """

    Lprime = LCO_single(
        image=image,
        pixel_area_arcsec2=pixel_area_arcsec2,
        beam_area_arcsec2=beam_area_arcsec2,
        beam_area_pc2=beam_area_pc2,
        R_21=R_21,
        R_31=R_31,
        alpha_CO=alpha_CO,
        name=name,
        D_Mpc=D_Mpc,
        co32=co32,
        aperture=aperture
    )

    Lprime_10 = Lprime / (R_31 if co32 else R_21)
    return alpha_CO * Lprime_10


names = ['NGC3717','NGC5921','NGC4254','NGC4224','NGC5037','NGC3749']
ras   = [172.88294,230.48537,184.70681,184.14078,198.74743,173.97176]
decs  = [-30.30766,5.07064,14.41649,7.46211,-16.59027,-37.99746]
D = np.array([24.0,21.0,15,41,35,42])

for name, ra, dec,D_Mpc in zip(names, ras, decs,D):
    print('\n',name, ra, dec)

    c = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    ra_str = c.ra.to_string(unit=u.hour, sep=':', precision=4)
    dec_str = c.dec.to_string(unit=u.deg, sep=':', precision=3, alwayssign=True)

    print(ra_str)
    print(dec_str)


    file = f'/Users/administrator/Astro/LLAMA/ALMA/pipeline_m0/{name}/{name}_12m_co21_broad_mom0.fits'

    # --- OPEN FITS ---
    with fits.open(file) as hdul:
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


    gal_cen = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')

    jytok = header['JYTOK']

    beam = header['BMAJ'] * 3600  # arcsec
    beam_area_arcsec2 = beam**2
    pixel = header['CDELT1']*3600
    pixel_area_arcsec2 = pixel**2
    beam_sr = (np.pi/(4*np.log(2)))*((beam/3600)*(2*np.pi)/360)**2




    aperture1as = CircleSkyRegion(center=gal_cen, radius=(0.5/3600) * u.deg)

    aperture_sr = (np.pi/(4*np.log(2)))*((1/3600)*(2*np.pi)/360)**2

    R_21, R_31, alpha_CO = 0.65, 0.32, 4.35
    pc_per_arcsec = (D_Mpc * 1e6) / 206265
    beam_area_pc2 = beam_area_arcsec2 * pc_per_arcsec**2

    jy_1arcsec = SCOdv_single(data, jytok,beam_area_arcsec2, pixel_area_arcsec2,aperture1as)
    mass_1arcsec = total_mass_single(data,pixel_area_arcsec2,beam_area_arcsec2,beam_area_pc2,R_21,R_31,alpha_CO,name,D_Mpc,co32=False,aperture=aperture1as)
    mass_sd_1arcsec = mass_1arcsec/(np.pi*(0.5*pc_per_arcsec)**2)

    print('mass=',mass_1arcsec)
    print('mass_sd=',mass_sd_1arcsec)

    I = jy_1arcsec / aperture_sr


    print(name,"flux (Jy km/s) 1as", round(jy_1arcsec, 3))

    area_ratio = (0.1/1)
    fluxnewmJyco32 = jy_1arcsec*1e3*area_ratio*0.52
    fluxOT = (fluxnewmJyco32/(200/10))
    snr = fluxOT/0.55


    print("fd (myJy/beam)", round(fluxOT, 3))
    print('S/N',round(snr,1))





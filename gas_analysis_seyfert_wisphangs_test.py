import os
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import uniform_filter
from astroquery.ipac.ned import Ned
from astroquery.exceptions import RemoteServiceError
import requests
from astroquery.simbad import Simbad
simbad = Simbad()
from astroquery.vizier import Vizier
import time
from astropy.coordinates import SkyCoord
from regions import EllipsePixelRegion, PixCoord
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from astropy.visualization.wcsaxes import add_scalebar
from astropy.visualization.wcsaxes import add_beam
from astropy.nddata import NDData
from astropy.wcs import WCS
from matplotlib.patches import Ellipse
from scipy.ndimage import gaussian_filter
from radio_beam import Beam
from astropy.convolution import convolve_fft
from radio_beam.utils import BeamError

# ------------------ Metrics ------------------

def gini(image, mask, **kwargs):
    valid_data = image[~mask & np.isfinite(image)].flatten()
    if len(valid_data) == 0:
        return np.nan
    sorted_vals = np.sort(valid_data)
    if np.nanmin(sorted_vals) < 0:
        sorted_vals -= np.nanmin(sorted_vals)  # boost so minimum is 0
    sorted_vals += 1e-7  # boost so minimum is not 0
    n = len(sorted_vals)
    total = np.sum(sorted_vals)
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1)

    mean = total / (n-1)
    G = 1/(mean*n*(n-1)) * np.sum((2*index - n - 1) * sorted_vals)
    return G

def asymmetry(image, mask, **kwargs):
    image_rot = np.rot90(image, 2)
    mask_rot = np.rot90(mask, 2)
    if np.sum(~mask) == 0 or np.sum(~mask_rot) == 0:
        return np.nan
    diff = abs(image - image_rot)
    total = image[~mask]
    return np.sum(diff[~mask]) / np.sum(total) if np.sum(total) > 0 else np.nan

def smoothness(image, mask, pc_per_arcsec, pixel_scale_arcsec, **kwargs): 
    smoothing_sigma_pc = 500 
    smoothing_sigma = (smoothing_sigma_pc / pc_per_arcsec) / pixel_scale_arcsec 
    size = max(1, int(round(smoothing_sigma))) 
    image_filled = np.nan_to_num(image, nan=0.0) 
    smooth_image = uniform_filter(image_filled, size=size, mode='constant',cval=0.0)
    valid_smooth = (~mask) & np.isfinite(image) & np.isfinite(smooth_image)
    if np.sum(valid_smooth) == 0: return np.nan 
    diff_smooth = image[valid_smooth] - smooth_image[valid_smooth] # original image and smoothed image use orginal mask
    total_flux = np.sum(image[valid_smooth])
    return np.sum(diff_smooth[(diff_smooth>0)]) / total_flux if total_flux > 0 else np.nan #

def normalize_name(name):
    n = str(name)
    # Use plain str.replace for single string inputs (no regex kwarg)
    return n.replace('–', '-').replace('−', '-').strip().upper()

def make_projected_region_mask(
    shape, R_kpc, pc_per_arcsec, pixel_scale_arcsec, PA, I
):
    """
    Create a boolean mask for a circular region of radius R_kpc in the galaxy
    plane, projected onto the sky as an ellipse.

    Returns
    -------
    mask : 2D boolean array
        True inside the projected ellipse
    aperture : EllipticalAperture
        Photutils aperture object defining the ellipse
    """
    R_pc = R_kpc * 1000.0
    R_arcsec = R_pc / pc_per_arcsec
    R_pix = R_arcsec / pixel_scale_arcsec
    R_multip = 1

    # ---- Apply inclination: b = a cos(i)
    I_rad = np.deg2rad(I)
    a = R_pix * R_multip
    b = R_pix * R_multip * np.cos(I_rad)

    # ---- Convert astronomical PA → photutils angle
    # Photutils θ is CCW from +x (east)
    # Astronomical PA is measured east of north
    #
    # Therefore:
    theta = np.deg2rad(90.0 + PA)
    angle = theta * u.rad

    # ---- Build elliptical aperture
    cx_trim = shape[1] // 2
    cy_trim = shape[0] // 2
    center = PixCoord(x=cx_trim, y=cy_trim)
    aperture = EllipsePixelRegion(center=center, width=2*a, height=2*b, angle=angle)

    # ---- Convert to mask image
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    coords = PixCoord(xx, yy)
    mask = ~aperture.contains(coords)

    return mask, aperture, R_kpc


def plot_moment_map(image, outfolder, name_short, BMAJ, BMIN, R_kpc, mask, aperture=None, norm_type='linear'):
    # Initialise plot
    fontsize = 35 * R_kpc
    plt.rcParams.update({'font.size': fontsize})
    figsize = 18 * R_kpc
    fig = plt.figure(figsize=(figsize , figsize),constrained_layout=True)
    ax = fig.add_subplot(111, projection=image.wcs.celestial)
    ax.margins(x=0,y=0)
    ax.set_axis_off()

    add_scalebar(ax,1/3600,label="1''",corner='top left',color='lime',borderpad=2,size_vertical=0.5)
    linewith = 2 * R_kpc
    add_beam(ax,major=BMAJ,minor=BMIN,angle=0,corner='bottom right',color='lime',borderpad=2,fill=True,linewidth=linewith)
    vmin = 0
    vmax = np.nanmax(image.data[np.isfinite(image.data)])
    if np.isfinite(image.data).any():
        if norm_type == 'sqrt':
            norm = simple_norm(image.data, 'sqrt', vmin=vmin, vmax=vmax)
        elif norm_type == 'linear':
            norm = simple_norm(image.data, 'linear', vmin=vmin, vmax=vmax)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")
    else:
        print("Moment data empty or all NaNs — skipping normalization.")
        norm = None
    #plt.title(f'{name_short}',fontsize=75)
    im=plt.imshow(image.data,origin='lower',norm=norm,cmap='RdBu_r')
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)


    cx = aperture.center.x
    cy = aperture.center.y
    width = aperture.width     # = 2*a in pixels
    height = aperture.height   # = 2*b in pixels
    angle_deg = aperture.angle.to_value("deg")

    ellipse_patch = Ellipse(
        (cx, cy), width=width, height=height,
        angle=angle_deg,
        edgecolor='lime',       # visible bright color
        facecolor='none',
        linewidth=3,
    )
    ax.add_patch(ellipse_patch)


    plt.savefig(outfolder,bbox_inches='tight',pad_inches=0.0)
    plt.close(fig)


def fix_ned_name(s: str) -> str:
    s_new = s.replace("ngc","NGC")
    s_new = s_new.replace("NGC0", "NGC")
    s_new = s_new.replace("NGC","NGC ")
    return s_new
# ------------------ Processing ------------------

def process_fits_file(filepath,phangs_df,wis_df):

  ######### Set parameters ####################################

    R_kpc = 1.5

######### Load fits data ####################################

    try:
        image_untrimmed = fits.getdata(filepath, memmap=True)
    except Exception as e:
        print(f"Skipping {filepath}: {e}")
        return None

    if image_untrimmed is None:
        return None
    
    mask_untrimmed = np.isnan(image_untrimmed)

    name = os.path.basename(filepath).replace(".fits", "")

######### Query databases ####################################

    name = normalize_name(name)

    match_phangs = phangs_df.loc[phangs_df["Name"] == name]
    match_wis = wis_df.loc[wis_df["Name"] == name]

    if len(match_phangs):
        row_phangs = match_phangs.iloc[0]

        D_Mpc = row_phangs["Distance (Mpc)"]
        PA = row_phangs["PA"]
        I = row_phangs["i"]
        RA = row_phangs['RA']
        DEC = row_phangs['DEC']
        print('Found in phangs df')
    else:
        row_wis = match_wis.iloc[0]
        D_Mpc = row_wis['Distance (Mpc)']
        PA = row_wis["PA"]
        I = row_wis["i"]
        RA = row_wis['RA']
        DEC = row_wis['DEC']
        print('found in wis df')

######### Read header info ####################################

    header = fits.getheader(filepath)
    pixel_scale_arcsec = np.abs(header.get("CDELT1", 0)) * 3600
    R_pixel = int(R_kpc * (206.265 / D_Mpc) / pixel_scale_arcsec)
    BMAJ = header.get("BMAJ", np.nan)
    BMIN = header.get("BMIN", np.nan)
    beam_arcsec = np.sqrt(np.abs(BMAJ * BMIN)) * 3600
    pc_per_arcsec = (D_Mpc * 1e6) / 206265
    pixel_scale_pc = pixel_scale_arcsec * pc_per_arcsec
    beam_scale_pc = beam_arcsec * pc_per_arcsec


######### Adjust image size ####################################

    gal_cen = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs')
    wcs_full = WCS(header).celestial

    ny, nx = image_untrimmed.shape
    try:
        cx, cy = gal_cen.to_pixel(wcs_full)
        cx, cy = int(cx), int(cy)
    except Exception as e:
        print(f"WARNING: WCS conversion failed for {name}: {e}")
        cx, cy = nx // 2, ny // 2

    target_size = 2 * R_pixel
    nx_full, ny_full = nx, ny

    x1, x2 = cx - R_pixel, cx + R_pixel
    y1, y2 = cy - R_pixel, cy + R_pixel

    image = np.full((target_size, target_size), np.nan)
    mask = np.ones_like(image, dtype=bool)

    x1i, x2i = max(0, x1), min(nx_full, x2)
    y1i, y2i = max(0, y1), min(ny_full, y2)

    xp1, xp2 = x1i - x1, x1i - x1 + (x2i - x1i)
    yp1, yp2 = y1i - y1, y1i - y1 + (y2i - y1i)

    image[yp1:yp2, xp1:xp2] = image_untrimmed[y1i:y2i, x1i:x2i]
    mask[yp1:yp2, xp1:xp2] = mask_untrimmed[y1i:y2i, x1i:x2i]

########## ---------- NaN handling ---------- ##################
    nan_pixels = np.isnan(image)
    if nan_pixels.any():
        image[nan_pixels] = 0.0
        mask[nan_pixels] = False

    # smoothing

    rebin = 120 # target resolution (pc)
    #Native beam (assumed stored in degrees) 
    beam = Beam( major=BMAJ * u.deg, minor=BMIN * u.deg, pa= PA * u.deg )
    # Target circular beam corresponding to 120 pc
    target_fwhm_arcsec = rebin / pc_per_arcsec
    target_beam = Beam( major=target_fwhm_arcsec * u.arcsec, minor=target_fwhm_arcsec * u.arcsec, pa=0 * u.deg)

    try:
        kernel = target_beam.deconvolve(beam).as_kernel(
            pixel_scale_arcsec * u.arcsec
        )

        image = convolve_fft(
            image,
            kernel,
            boundary="fill",
            fill_value=0.0,
            normalize_kernel=True,
            preserve_nan=True,
        )

        BMAJ = target_beam.major.to(u.deg).value
        BMIN = target_beam.minor.to(u.deg).value
        BPA = target_beam.pa.to(u.deg).value

    except BeamError:
        print(f"{name}: {beam_scale_pc} native beam is already larger than or incompatible with a {rebin} pc circular beam.")

    # ---------- Update WCS ----------
    wcs_trimmed = wcs_full.deepcopy()
    wcs_trimmed.wcs.crpix[0] -= x1
    wcs_trimmed.wcs.crpix[1] -= y1
    image_nd = NDData(data=image, wcs=wcs_trimmed)  

    # ---------- Flux mask ----------

    total_flux = np.nansum(image[~mask])
    f = 2.0
    ratio = 1.0
    while ratio > 0.9:
        flux_mask_90, flux_aperture_90, R_90 = make_projected_region_mask(
            image.shape, R_kpc * f * 1.42 , pc_per_arcsec, pixel_scale_arcsec, PA, I
        )
        flux_90 = np.nansum(image[~flux_mask_90 & ~mask])
        ratio = flux_90 / total_flux if total_flux > 0 else 0.0
        f -= 0.005

    print(
        f"Flux in aperture = {ratio} of total. R_90 (kpc): ",
        round(R_90, 2)
    )
    mask = flux_mask_90 | mask
    aperture_to_plot = flux_aperture_90

    output_dir = "/Users/administrator/Astro/LLAMA/ALMA/comp_samples/"
    flux_mask_path = output_dir + '/masks'+ f'/{name}_flux90_mask.fits'  
    if not os.path.exists(os.path.dirname(flux_mask_path)):
        os.makedirs(os.path.dirname(flux_mask_path))
    fits.writeto(flux_mask_path, mask.astype(int), overwrite=True)

    plots_path = output_dir+'m0_plots/'
    if not os.path.exists(os.path.dirname(plots_path)):
        os.makedirs(os.path.dirname(plots_path))
    plot_moment_map(image_nd, plots_path+f'{name}_mom0.png', name, BMAJ, BMIN, R_kpc, mask, aperture=aperture_to_plot)


    gin = gini(image, mask)
    asym = asymmetry(image,mask)
    smooth = smoothness(image, mask, pixel_scale_arcsec, pc_per_arcsec)



    return {
        "name": name,
        "Gini": round(gin,3),
        "Asymmetry": round(asym,3),
        "Smoothness_davis": round(smooth,3)
    }


# ------------------ Main ------------------

def run_directory(input_dir, output_csv):
    results = []
    phangs_df = pd.read_csv("/Users/administrator/Astro/LLAMA/ALMA/comp_samples/phangs_new.csv")
    wis_df = pd.read_csv("/Users/administrator/Astro/LLAMA/ALMA/comp_samples"+"/wis_new.csv")

    for f in os.listdir(input_dir):
        if not f.endswith(".fits"):
            continue

        filepath = os.path.join(input_dir, f)
        print(f"Processing {f}")

        row = process_fits_file(filepath,phangs_df,wis_df)
        if row is not None:
            results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")


# ------------------ Execute ------------------

if __name__ == "__main__":
    input_dir_wis = "/Users/administrator/Astro/LLAMA/ALMA/comp_samples/m0/wis"
    output_csv_wis = "/Users/administrator/Astro/LLAMA/ALMA/comp_samples/m0_metrics_wis.csv"

    run_directory(input_dir_wis, output_csv_wis)

    input_dir_phangs = "/Users/administrator/Astro/LLAMA/ALMA/comp_samples/m0/phangs"
    output_csv_phangs = "/Users/administrator/Astro/LLAMA/ALMA/comp_samples/m0_metrics_phangs.csv"

    run_directory(input_dir_phangs, output_csv_phangs)
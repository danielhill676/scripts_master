import os
import gc
import time
import traceback
import multiprocessing
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import uniform_filter
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from astropy.visualization.wcsaxes import add_scalebar
from astropy.visualization.wcsaxes import add_beam
from astropy.nddata import NDData
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from regions import EllipsePixelRegion, PixCoord
import astropy.units as u
from astroquery.ipac.ned import Ned
from astroquery.exceptions import RemoteServiceError
import requests
from multiprocessing import shared_memory
from matplotlib.patches import Ellipse
from astroquery.simbad import Simbad
simbad = Simbad()
from astroquery.vizier import Vizier
from scipy.ndimage import gaussian_filter

np.seterr(all='ignore')
co32 = False
LLAMATAB = None
outer_dir_co32 = '/data/c3040163/llama/alma/phangs_imaging_scripts-master/CO32_all_arrays/reduction/derived/'
outer_dir_co21 = '/data/c3040163/llama/alma/phangs_imaging_scripts-master/full_run_newkeys_all_arrays/reduction/derived'
# ------------------ Monte Carlo Helpers ------------------

def generate_random_images(image, error_map, n_iter=1000, seed=None):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=image, scale=error_map, size=(n_iter, *image.shape))


def process_mc_chunk_shm(
    n_iter_chunk,
    shm_name_image,
    shm_name_error,
    shape,
    dtype_str,
    mask,
    metric_kwargs,
    isolate=None,
    seed=None
):
    """
    Monte Carlo metrics for a chunk of images.
    """
    # attach to shared memory blocks
    shm_img = shared_memory.SharedMemory(name=shm_name_image)
    shm_err = shared_memory.SharedMemory(name=shm_name_error)
    dtype = np.dtype(dtype_str)
    image = np.ndarray(shape, dtype=dtype, buffer=shm_img.buf)
    errmap = np.ndarray(shape, dtype=dtype, buffer=shm_err.buf)

    rng = np.random.default_rng(seed)

    # Lists to collect metrics
    gini_vals = []
    asym_vals = []
    smooth_vals = []
    conc_vals = []
    tm_vals = []
    mw_vals = []
    aw_vals = []
    clump_vals = []
    LCO_vals = []
    SCOdv_vals = []
    LCO_JCMT_vals = []
    LCO_APEX_vals = []

    for _ in range(n_iter_chunk):
        noise = rng.standard_normal(size=shape) * errmap
        mc_img = image + noise

        # --- Metrics ---
        try:
            gini_vals.append(gini_single(mc_img, mask))
        except Exception:
            gini_vals.append(np.nan)

        try:
            asym_vals.append(asymmetry_single(mc_img, mask))
        except Exception:
            asym_vals.append(np.nan)

        try:
            smooth_vals.append(smoothness_single(
                mc_img, mask,
                pc_per_arcsec=metric_kwargs["pc_per_arcsec"],
                pixel_scale_arcsec=metric_kwargs["pixel_scale_arcsec"]
            ))
        except Exception:
            smooth_vals.append(np.nan)

        try:
            conc_vals.append(concentration_single(
                mc_img, mask,
                pixel_scale_arcsec=metric_kwargs["pixel_scale_arcsec"],
                pc_per_arcsec=metric_kwargs["pc_per_arcsec"]
            ))
        except Exception:
            conc_vals.append(np.nan)


        try:
            SCOdv_vals.append(SCOdv_single(
                mc_img,
                mask,
                jy_per_K=metric_kwargs["jy_per_K"],        
                beam_area_arcsec2=metric_kwargs["beam_area_arcsec2"],
                pixel_area_arcsec2=metric_kwargs["pixel_area_arcsec2"]
            ))
        except Exception:
            SCOdv_vals.append(np.nan)


        # --- L'CO and derived quantities ---
        try:
            LCO_vals.append(LCO_single(
                mc_img, mask,
                pixel_area_arcsec2=metric_kwargs["pixel_area_arcsec2"],
                beam_area_arcsec2=metric_kwargs["beam_area_arcsec2"],
                beam_area_pc2=metric_kwargs["beam_area_pc2"],
                R_21=metric_kwargs["R_21"],
                R_31=metric_kwargs["R_31"],
                alpha_CO=metric_kwargs["alpha_CO"],
                name=metric_kwargs["name"],
                D_Mpc=metric_kwargs["D_Mpc"],
                co32=metric_kwargs["co32"]
            ))
        except Exception:
            LCO_vals.append(np.nan)

        try:
            tm_vals.append(total_mass_single(
                mc_img, mask,
                pixel_area_arcsec2=metric_kwargs["pixel_area_arcsec2"],
                beam_area_arcsec2=metric_kwargs["beam_area_arcsec2"],
                beam_area_pc2=metric_kwargs["beam_area_pc2"],
                R_21=metric_kwargs["R_21"],
                R_31=metric_kwargs["R_31"],
                alpha_CO=metric_kwargs["alpha_CO"],
                name=metric_kwargs["name"],
                D_Mpc=metric_kwargs["D_Mpc"],
                co32=metric_kwargs["co32"]
            ))
        except Exception:
            tm_vals.append(np.nan)

        try:
            mw_vals.append(mass_weighted_sd_single(
                mc_img, mask,
                pixel_area_pc2=metric_kwargs["pixel_area_pc2"],
                pixel_area_arcsec2=metric_kwargs["pixel_area_arcsec2"],
                beam_area_arcsec2=metric_kwargs["beam_area_arcsec2"],
                beam_area_pc2=metric_kwargs["beam_area_pc2"],
                R_21=metric_kwargs["R_21"],
                R_31=metric_kwargs["R_31"],
                alpha_CO=metric_kwargs["alpha_CO"],
                name=metric_kwargs["name"],
                D_Mpc=metric_kwargs["D_Mpc"],
                co32=metric_kwargs["co32"]
            ))
        except Exception:
            mw_vals.append(np.nan)

        try:
            aw_vals.append(area_weighted_sd_single(
                mc_img, mask,
                pixel_area_pc2=metric_kwargs["pixel_area_pc2"],
                pixel_area_arcsec2=metric_kwargs["pixel_area_arcsec2"],
                beam_area_arcsec2=metric_kwargs["beam_area_arcsec2"],
                beam_area_pc2=metric_kwargs["beam_area_pc2"],
                R_21=metric_kwargs["R_21"],
                R_31=metric_kwargs["R_31"],
                alpha_CO=metric_kwargs["alpha_CO"],
                name=metric_kwargs["name"],
                D_Mpc=metric_kwargs["D_Mpc"],
                co32=metric_kwargs["co32"]
            ))
        except Exception:
            aw_vals.append(np.nan)

        try:
            clump_vals.append(clumping_factor_single(
                mc_img, mask,
                pixel_area_pc2=metric_kwargs["pixel_area_pc2"],
                pixel_area_arcsec2=metric_kwargs["pixel_area_arcsec2"],
                beam_area_arcsec2=metric_kwargs["beam_area_arcsec2"],
                beam_area_pc2=metric_kwargs["beam_area_pc2"],
                R_21=metric_kwargs["R_21"],
                R_31=metric_kwargs["R_31"],
                alpha_CO=metric_kwargs["alpha_CO"],
                name=metric_kwargs["name"],
                D_Mpc=metric_kwargs["D_Mpc"],
                co32=metric_kwargs["co32"]
            ))
        except Exception as e:
            print(f"Error in clump: {e}")
            clump_vals.append(np.nan)

        # Single-dish L'CO
        try:
            LCO_JCMT_vals.append(LCO_single_JCMT(
                mc_img, mask,
                pixel_scale_arcsec=metric_kwargs["pixel_scale_arcsec"],
                pixel_area_arcsec2=metric_kwargs["pixel_area_arcsec2"],
                beam_area_arcsec2=metric_kwargs["beam_area_arcsec2"],
                beam_area_pc2=metric_kwargs["beam_area_pc2"],
                R_21=metric_kwargs["R_21"],
                R_31=metric_kwargs["R_31"],
                alpha_CO=metric_kwargs["alpha_CO"],
                name=metric_kwargs["name"],
                D_Mpc=metric_kwargs["D_Mpc"],
                co32=metric_kwargs["co32"]
            ))
        except Exception:
            LCO_JCMT_vals.append(np.nan)

        try:
            LCO_APEX_vals.append(LCO_single_APEX(
                mc_img, mask,
                pixel_scale_arcsec=metric_kwargs["pixel_scale_arcsec"],
                pixel_area_arcsec2=metric_kwargs["pixel_area_arcsec2"],
                beam_area_arcsec2=metric_kwargs["beam_area_arcsec2"],
                beam_area_pc2=metric_kwargs["beam_area_pc2"],
                R_21=metric_kwargs["R_21"],
                R_31=metric_kwargs["R_31"],
                alpha_CO=metric_kwargs["alpha_CO"],
                name=metric_kwargs["name"],
                D_Mpc=metric_kwargs["D_Mpc"],
                co32=metric_kwargs["co32"]
            ))
        except Exception:
            LCO_APEX_vals.append(np.nan)

    shm_img.close()
    shm_err.close()

    return {
        "gini": gini_vals,
        "asym": asym_vals,
        "smooth": smooth_vals,
        "conc": conc_vals,
        "tmass": tm_vals,
        "mw": mw_vals,
        "aw": aw_vals,
        "clump": clump_vals,
        "SCOdv": SCOdv_vals,
        "LCO": LCO_vals,
        "LCO_JCMT": LCO_JCMT_vals,
        "LCO_APEX": LCO_APEX_vals
    }


# ------------------ Metric Functions ------------------

def gini_single(image, mask, **kwargs):
    valid_data = image[~mask & np.isfinite(image)].flatten()
    valid_data = valid_data[valid_data >= 0]
    if len(valid_data) == 0:
        return np.nan
    sorted_vals = np.sort(valid_data)
    n = len(sorted_vals)
    total = np.sum(sorted_vals)
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_vals) / (n * total)) - (n + 1) / n

def asymmetry_single(image, mask, **kwargs):
    image_rot = np.rot90(image, 2)
    mask_rot = np.rot90(mask, 2)
    valid_mask = (~mask) & (~mask_rot) & np.isfinite(image) & np.isfinite(image_rot)
    if np.sum(valid_mask) == 0:
        return np.nan
    diff = np.abs(image[valid_mask] - image_rot[valid_mask])
    total = np.abs(image[valid_mask])
    return np.sum(diff) / np.sum(total) if np.sum(total) > 0 else np.nan

def smoothness_single(image, mask, pc_per_arcsec, pixel_scale_arcsec, **kwargs):
    smoothing_sigma_pc = 500
    smoothing_sigma = (smoothing_sigma_pc / pc_per_arcsec) / pixel_scale_arcsec
    size = max(1, int(round(smoothing_sigma)))
    image_filled = np.nan_to_num(image, nan=0.0)
    valid_mask = (~mask) & np.isfinite(image)
    smooth_image = uniform_filter(image_filled, size=size, mode='reflect')
    smooth_mask = uniform_filter(valid_mask.astype(float), size=size, mode='reflect')
    with np.errstate(invalid='ignore', divide='ignore'):
        image_smooth = smooth_image / smooth_mask
    image_smooth[smooth_mask == 0] = np.nan
    valid_smooth = (~mask) & np.isfinite(image) & np.isfinite(image_smooth)
    if np.sum(valid_smooth) == 0:
        return np.nan
    diff_smooth = np.abs(image[valid_smooth] - image_smooth[valid_smooth])
    total_flux = np.abs(image[valid_smooth])
    return np.sum(diff_smooth) / np.sum(total_flux) if np.sum(total_flux) > 0 else np.nan

def concentration_single(image, mask, pixel_scale_arcsec, pc_per_arcsec, **kwargs):
    y, x = np.indices(image.shape)
    center = (x.max() / 2, y.max() / 2)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r_pc = r * pixel_scale_arcsec * pc_per_arcsec
    valid = (~mask) & np.isfinite(image)
    flux_50 = np.sum(image[(r_pc < 50) & valid]) / 50**2
    flux_200 = np.sum(image[(r_pc < 200) & valid]) / 200**2
    if flux_50 <= 0 or flux_200 <= 0:
        return np.nan
    return np.log10(flux_50 / flux_200)


def SCOdv_single(
    image,
    mask,
    jy_per_K,        
    beam_area_arcsec2, 
    pixel_area_arcsec2
):
    pix_per_beam = beam_area_arcsec2 / pixel_area_arcsec2
    # Sum integrated intensity in aperture (K km/s)
    I_sum = np.nansum(image[~mask])

    # Convert to total flux (Jy km/s)
    S_CO_dv = (I_sum / jy_per_K) / pix_per_beam

    return S_CO_dv


def LCO_single(
    image,
    mask,
    pixel_area_arcsec2,
    beam_area_arcsec2,
    beam_area_pc2,
    R_21,
    R_31,
    alpha_CO,
    name,
    D_Mpc,
    co32=False,
    **kwargs
):
    """
    Total L'CO in K km s^-1 pc^2 (beam-correct, resolution-independent).
    image must be in K km s^-1 (per beam).
    """

    pix_per_beam = beam_area_arcsec2 / pixel_area_arcsec2

    I_sum = np.nansum(image[~mask])  # K km/s summed over pixels
    Lprime = I_sum * 23.5 * beam_area_arcsec2/pix_per_beam * D_Mpc**2

    if co32:
        Lprime = (Lprime / R_31) * R_21

    return Lprime


############################
# Total molecular mass
############################

def total_mass_single(
    image,
    mask,
    pixel_area_arcsec2,
    beam_area_arcsec2,
    beam_area_pc2,
    R_21,
    R_31,
    alpha_CO,
    name,
    D_Mpc,
    co32=False,
    **kwargs
):
    """
    Total H2 mass from beam-correct L'CO.
    """

    Lprime = LCO_single(
        image=image,
        mask=mask,
        pixel_area_arcsec2=pixel_area_arcsec2,
        beam_area_arcsec2=beam_area_arcsec2,
        beam_area_pc2=beam_area_pc2,
        R_21=R_21,
        R_31=R_31,
        alpha_CO=alpha_CO,
        name=name,
        D_Mpc=D_Mpc,
        co32=co32,
    )

    Lprime_10 = Lprime / (R_31 if co32 else R_21)
    return alpha_CO * Lprime_10


############################
# Aperture helpers
############################

def _aperture_mask(image, pixel_scale_arcsec, radius_arcsec):
    y, x = np.indices(image.shape)
    center = (x.max() / 2, y.max() / 2)
    r_pix = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r_arcsec = r_pix * pixel_scale_arcsec
    return r_arcsec <= radius_arcsec


############################
# Single-dish equivalents
############################

def LCO_single_JCMT(
    image,
    mask,
    pixel_scale_arcsec,
    pixel_area_arcsec2,
    beam_area_arcsec2,
    beam_area_pc2,
    R_21,
    R_31,
    alpha_CO,
    name,
    D_Mpc,
    co32=False,
    **kwargs
):
    jcmt_mask = _aperture_mask(image, pixel_scale_arcsec, 20.4 / 2)
    combined_mask = mask | ~jcmt_mask

    return LCO_single(
        image=image,
        mask=combined_mask,
        pixel_area_arcsec2=pixel_area_arcsec2,
        beam_area_arcsec2=beam_area_arcsec2,
        beam_area_pc2=beam_area_pc2,
        R_21=R_21,
        R_31=R_31,
        alpha_CO=alpha_CO,
        name=name,
        D_Mpc=D_Mpc,
        co32=co32,
    )


def LCO_single_APEX(
    image,
    mask,
    pixel_scale_arcsec,
    pixel_area_arcsec2,
    beam_area_arcsec2,
    beam_area_pc2,
    R_21,
    R_31,
    alpha_CO,
    name,
    D_Mpc,
    co32=False,
    **kwargs
):
    apex_mask = _aperture_mask(image, pixel_scale_arcsec, 27.1 / 2)
    combined_mask = mask | ~apex_mask

    return LCO_single(
        image=image,
        mask=combined_mask,
        pixel_area_arcsec2=pixel_area_arcsec2,
        beam_area_arcsec2=beam_area_arcsec2,
        beam_area_pc2=beam_area_pc2,
        R_21=R_21,
        R_31=R_31,
        alpha_CO=alpha_CO,
        name=name,
        D_Mpc=D_Mpc,
        co32=co32,
    )


############################
# Surface density machinery
############################

def _Sigma_H2_map(
    image,
    mask,
    pixel_area_pc2,
    pixel_area_arcsec2,
    beam_area_arcsec2,
    beam_area_pc2,
    R_21,
    R_31,
    alpha_CO,
    D_Mpc,
    co32=False,
):
    """
    Returns Σ_H2 in M_sun pc^-2 for each pixel (mask applied).
    """

    pix_per_beam = beam_area_arcsec2 / pixel_area_arcsec2

    # L'CO per pixel (beam-correct)
    Lprime_pix = image * 23.5 * beam_area_arcsec2/pix_per_beam * D_Mpc**2
    if co32:
        Lprime_pix = (Lprime_pix / R_31) * R_21

    MH2_pix = alpha_CO * Lprime_pix
    Sigma = MH2_pix / pixel_area_pc2

    return Sigma[~mask]


def mass_weighted_sd_single(
    image,
    mask,
    pixel_area_pc2,
    pixel_area_arcsec2,
    beam_area_arcsec2,
    beam_area_pc2,
    R_21,
    R_31,
    alpha_CO,
    name,
    D_Mpc,
    co32=False,
    **kwargs
):
    Sigma = _Sigma_H2_map(
        image,
        mask,
        pixel_area_pc2,
        pixel_area_arcsec2,
        beam_area_arcsec2,
        beam_area_pc2,
        R_21,
        R_31,
        alpha_CO,
        D_Mpc,
        co32,
    )

    if Sigma.size == 0:
        return np.nan

    numerator = np.sum(Sigma**2 * pixel_area_pc2)
    denominator = np.sum(Sigma * pixel_area_pc2)

    return numerator / denominator if denominator > 0 else np.nan


def area_weighted_sd_single(
    image,
    mask,
    pixel_area_pc2,
    pixel_area_arcsec2,
    beam_area_arcsec2,
    beam_area_pc2,
    R_21,
    R_31,
    alpha_CO,
    name,
    D_Mpc,
    co32=False,
    **kwargs
):
    Sigma = _Sigma_H2_map(
        image,
        mask,
        pixel_area_pc2,
        pixel_area_arcsec2,
        beam_area_arcsec2,
        beam_area_pc2,
        R_21,
        R_31,
        alpha_CO,
        D_Mpc,
        co32,
    )

    if Sigma.size == 0:
        return np.nan

    total_area = Sigma.size * pixel_area_pc2
    return np.sum(Sigma * pixel_area_pc2) / total_area if total_area > 0 else np.nan

def clumping_factor_single(
    image, mask,
    pixel_area_pc2,
    pixel_area_arcsec2,
    beam_area_arcsec2,
    beam_area_pc2,
    R_21,
    R_31,
    alpha_CO,
    name,
    D_Mpc,
    co32=False,
    **kwargs
):
    mw = mass_weighted_sd_single(
        image, mask,
        pixel_area_pc2,
        pixel_area_arcsec2,
        beam_area_arcsec2,
        beam_area_pc2,
        R_21,
        R_31,
        alpha_CO,
        name,
        D_Mpc,
        co32=co32
    )
    aw = area_weighted_sd_single(
        image, mask,
        pixel_area_pc2,
        pixel_area_arcsec2,
        beam_area_arcsec2,
        beam_area_pc2,
        R_21,
        R_31,
        alpha_CO,
        name,
        D_Mpc,
        co32=co32
    )
    return mw / aw if aw and aw > 0 else np.nan

def radial_profile_with_errors(data, errors, mask, center=None, nbins=30):
    y, x = np.indices(data.shape)
    if center is None:
        center = (x.max() / 2, y.max() / 2)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    valid = ~mask
    r, data, errors = r[valid], data[valid], errors[valid]
    r_max = r.max()
    bin_edges = np.linspace(0, r_max, nbins + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_indices = np.digitize(r, bin_edges) - 1
    radial_mean = np.full(nbins, np.nan)
    radial_std_err = np.full(nbins, np.nan)
    for i in range(nbins):
        in_bin = bin_indices == i
        if np.any(in_bin):
            values = data[in_bin]
            errs = errors[in_bin]
            radial_mean[i] = np.mean(values)
            radial_std_err[i] = np.sqrt(np.sum(errs**2)) / len(values)
    return bin_centers, radial_mean, radial_std_err

def exp_profile(r, Sigma0, rs):
    return Sigma0 * np.exp(-r / rs)

# ----------------- Moment map plotting ------------------

def plot_moment_map(image, outfolder, name_short, BMAJ, BMIN, R_kpc, rebin, mask, flux_mask, aperture=None, norm_type='linear', res_src='native'): 
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

    # fig.tight_layout()
    if np.isfinite(image.data).any():
        if norm_type == 'sqrt':
            norm = simple_norm(image.data, 'sqrt', vmin=np.nanmin(image.data), vmax=np.nanmax(image.data))
        elif norm_type == 'linear':
            norm = simple_norm(image.data, 'linear', vmin=0, vmax=np.nanmax(image.data))
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")
    else:
        print("Moment data empty or all NaNs — skipping normalization.")
        norm = None
    #plt.title(f'{name_short}',fontsize=75)
    im=plt.imshow(image.data,origin='lower',norm=norm,cmap='RdBu_r')
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)

    if aperture is not None:
        # Regions EllipsePixelRegion parameters:
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

    if rebin is not None:
        if not flux_mask:
            plt.savefig(outfolder+f'/m0_plots/{R_kpc}_{rebin}_{mask}_{name_short}.png',bbox_inches='tight',pad_inches=0.0)
        else:
            plt.savefig(outfolder+f'/m0_plots/{R_kpc}_{rebin}_flux90_{mask}_{name_short}.png',bbox_inches='tight',pad_inches=0.0)    
    else:
        plt.savefig(outfolder+f'/m0_plots/{R_kpc}_no_rebin_{mask}_{name_short}_{res_src}.png',bbox_inches='tight',pad_inches=0.0)
    plt.close(fig)
        
# ------------------ Processing ------------------

def init_worker(table):
    global LLAMATAB
    LLAMATAB = table

def safe_process(args):
    name = os.path.basename(args[0]).split("_12m")[0]  # extract galaxy name
    try:
        return process_file(args)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error processing {name}: {e}")
        return ("__ERROR__", name, str(e), tb)
    
def normalize_name(name):
    n = str(name)
    # Use plain str.replace for single string inputs (no regex kwarg)
    return n.replace('–', '-').replace('−', '-').strip().upper()

def resolve_galaxy_beam_scale(
    name,
    fits_file,
    llamatab,
    max_retries=3
):
    header = fits.getheader(fits_file)

    # ---- defaults (critical!) ----
    RA = DEC = PA = I = D_Mpc = np.nan

    # ---- normalize name ----
    if name.endswith(("_phangs", "_wis")):
        base_name = normalize_name(name.rsplit("_", 1)[0].upper())
    else:
        base_name = name

    def query_ned_with_retries(query_name):
        for attempt in range(max_retries):
            try:
                return Ned.query_object(query_name)
            except (requests.exceptions.ConnectionError,
                    RemoteServiceError,
                    requests.exceptions.ReadTimeout):
                if attempt == max_retries - 1:
                    raise
                time.sleep(5)

    # ---- determine NED target ----
    if base_name in llamatab['id']:
        ned_name = llamatab[llamatab['id'] == base_name]['name'][0]
        needs_extended = False
    else:
        ned_name = base_name
        needs_extended = name.endswith(("_phangs", "_wis"))

    # ---- metadata resolution ----
    if not needs_extended:
        Ned_table = query_ned_with_retries(ned_name)
        RA = Ned_table['RA'][0]
        DEC = Ned_table['DEC'][0]

        # Special replacements:
        if base_name == 'NGC2992':
            RA = 146.42476
            DEC = -14.326274
        if base_name == 'NGC1365':
            RA = 53.401542
            DEC = -36.14039

        try:
            D_Mpc = llamatab[llamatab['id'] == base_name]['D [Mpc]'][0]
            I = llamatab[llamatab['id'] == base_name]['Inclination (deg)'][0]
            PA = llamatab[llamatab['id'] == base_name]['PA'][0]
        except Exception:
            pass

    else:
        Ned_table = query_ned_with_retries(ned_name)
        RA = Ned_table['RA'][0]
        DEC = Ned_table['DEC'][0]

        simbad.add_votable_fields('mesdistance')
        tbl = simbad.query_object(ned_name)
        D_Mpc = tbl['mesdistance.dist'][0]

        diam_table = Ned.get_table(ned_name, table='diameters')
        PA = diam_table['Position Angle'][0]

        try:
            result = Vizier.query_object(ned_name, catalog="J/ApJS/197/21/cgs")
            I = np.nanmedian(result[0]['i'])
        except Exception:
            result = Vizier.query_object(ned_name, catalog="VII/145/catalog")
            I = np.nanmedian(result[0]['i'])

    # ---- beam scale ----
    pixel_scale_arcsec = np.abs(header.get("CDELT1", 0)) * 3600
    pixel_area_arcsec2 = pixel_scale_arcsec**2
    BMAJ = header.get("BMAJ", np.nan)
    BMIN = header.get("BMIN", np.nan)
    beam_arcsec = np.sqrt(np.abs(BMAJ * BMIN)) * 3600
    beam_area_arcsec2 = (np.pi/(4*np.log(2)))*(BMAJ*3600)*(BMIN*3600)
    pc_per_arcsec = (D_Mpc * 1e6) / 206265
    beam_scale_pc = beam_arcsec * pc_per_arcsec
    beam_area_pc2 = beam_area_arcsec2 * pc_per_arcsec**2
    pixel_scale_pc = pixel_scale_arcsec * pc_per_arcsec
    pixel_area_pc2 = pixel_scale_pc**2

    return {
        "RA": RA,
        "DEC": DEC,
        "PA": PA,
        "Inclination": I,
        "D_Mpc": D_Mpc,
        "BMAJ": BMAJ,
        "BMIN": BMIN,
        "beam_scale_pc": beam_scale_pc,
        "pixel_scale_arcsec": pixel_scale_arcsec,
        "pc_per_arcsec": pc_per_arcsec,
        "pixel_area_arcsec2": pixel_area_arcsec2,
        "beam_area_pc2": beam_area_pc2,
        "pixel_area_arsec2": pixel_area_arcsec2,
        "beam_area_arcsec2": beam_area_arcsec2,
        "pixel_area_pc2": pixel_area_pc2,
        "header": header

    }

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
    theta = np.deg2rad(90.0 - PA)
    angle = theta * u.rad

    # ---- Build elliptical aperture
    cx_trim = shape[1] // 2
    cy_trim = shape[0] // 2
    center = PixCoord(x=cx_trim, y=cy_trim)
    aperture = EllipsePixelRegion(center=center, width=2*a, height=2*b, angle=angle)

    # ---- Convert to mask image
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    coords = PixCoord(xx, yy)
    mask = aperture.contains(coords)

    return mask, aperture, R_kpc


def process_file(args, images_too_small, isolate=None, manual_rebin=False, save_exp=False):
    mom0_file, emom0_file, outer_dir, subdir, output_dir, co32, rebin, PHANGS_mask, R_kpc, flux_mask = args
    file = mom0_file
    error_map_file = emom0_file

    # Galaxy name extraction (now robust)
    base = os.path.basename(file)
    extension = os.path.splitext(base)[1]

    name = base.split("_12m")[0]

    if name not in ['NGC4254','ngc4254_phangs','NGC3351','ngc3351_phangs']:
        return

    pair_names = []

    if rebin == None:
    
        try:    

            if normalize_name(llamatab[llamatab['id'] == name]['name'][0]) in df_pairs["Active Galaxy"].values:
                rows = df_pairs[df_pairs["Active Galaxy"].str.strip() == normalize_name(llamatab[llamatab['id'] == name]['name'][0])]
                for _, row in rows.iterrows():
                    pair_name = row["Inactive Galaxy"].strip()
                    pair_id = llamatab[llamatab['name'] == pair_name]['id'][0]
                    pair_names.append(pair_id)
            elif normalize_name(llamatab[llamatab['id'] == name]['name'][0]) in df_pairs["Inactive Galaxy"].values:
                rows = df_pairs[df_pairs["Inactive Galaxy"].str.strip() == normalize_name(llamatab[llamatab['id'] == name]['name'][0])]
                for _, row in rows.iterrows():
                    pair_name = row["Active Galaxy"].strip()
                    pair_id = llamatab[llamatab['name'] == pair_name]['id'][0]
                    pair_names.append(pair_id)
        except:
            print(f"No pair found for {name} in df_pairs.")


    # Skip incompatible galaxies
    co32_list = ['NGC4388','NGC6814','NGC5728']
    if not co32 and name in co32_list:
        return None
    if co32 and name not in co32_list:
        return None

    print(f"Processing {name}...")

    # Load FITS
    image_untrimmed = fits.getdata(file, memmap=True)
    if error_map_file is not np.nan:
        error_map_untrimmed = fits.getdata(error_map_file, memmap=True)
    else:
        error_map_untrimmed = np.full_like(image_untrimmed, 0)

    mask_untrimmed = np.isnan(image_untrimmed) | np.isnan(error_map_untrimmed)

    # cut off wierd bit of NGC 3351

    if name == 'NGC3351':
            image_untrimmed = image_untrimmed[:, :1600]
            error_map_untrimmed = error_map_untrimmed[:, :1600]
            mask_untrimmed = mask_untrimmed[:, :1600]

    main_meta = resolve_galaxy_beam_scale(
    name=name,
    fits_file=mom0_file,
    llamatab=llamatab
)
    RA = main_meta["RA"]
    DEC = main_meta["DEC"]
    PA = main_meta["PA"]
    I = main_meta["Inclination"]
    beam_scale_pc = main_meta["beam_scale_pc"]
    D_Mpc = main_meta["D_Mpc"]
    BMAJ = main_meta["BMAJ"]
    BMIN = main_meta["BMIN"]
    pixel_scale_arcsec = main_meta["pixel_scale_arcsec"]
    pixel_area_arcsec2 = main_meta["pixel_area_arcsec2"]
    beam_area_pc2 = main_meta["beam_area_pc2"]
    beam_area_arcsec2 = main_meta["beam_area_arcsec2"]
    pixel_area_pc2 = main_meta["pixel_area_pc2"]
    pc_per_arcsec = main_meta["pc_per_arcsec"]
    header = main_meta["header"]

    jy_per_K = float(header.get("JYTOK", 0))

    beam_scales_pc = [beam_scale_pc]
    beam_scale_labels = [name]

    if rebin is None:

        beam_scales_pc = [beam_scale_pc]
        beam_scale_labels = [name]

        for pair_name in pair_names:

            pair_name_norm = normalize_name(pair_name)
            pair_subdir = os.path.join(outer_dir, pair_name_norm)
            
            if co32 and pair_name_norm not in co32_list:
                pair_subdir = os.path.join(outer_dir_co21, pair_name_norm)
            if not co32 and pair_name_norm in co32_list:
                pair_subdir = os.path.join(outer_dir_co32, pair_name_norm)


            pair_fits = os.path.join(
                os.path.dirname(pair_subdir),
                pair_name_norm,
                os.path.basename(mom0_file).replace(name, pair_name_norm)
            )
            if co32 and pair_name_norm not in co32_list:
                pair_fits = pair_fits.replace("_co32_", "_co21_")
            if not co32 and pair_name_norm in co32_list:
                pair_fits = pair_fits.replace("_co21_", "_co32_")

            if not os.path.exists(pair_fits):
                beam_scales_pc.append(np.nan)
                beam_scale_labels.append(pair_name_norm)
                print(f"Missing FITS for path {pair_fits}")
                continue

            try:
                pair_meta = resolve_galaxy_beam_scale(
                    name=pair_name_norm,
                    fits_file=pair_fits,
                    llamatab=llamatab
                )

                beam_scales_pc.append(pair_meta["beam_scale_pc"])
                beam_scale_labels.append(pair_name_norm)

            except Exception as e:
                beam_scales_pc.append(np.nan)
                beam_scale_labels.append(pair_name_norm)
                print(f"Failed to process pair {pair_name_norm}: {e}")

    print(f"Beam scales (pc): {dict(zip(beam_scale_labels, beam_scales_pc))}")


    ####################### loop through matched pair resolutions #######################
    images = []
    errormaps = []
    BMAJs = []
    BMINs = []
    res_values = []
    res_sources = []


    # ---------- build resolution list ----------
    res_list = []

    native_res = float(beam_scale_pc)
    res_list.append(("native", native_res))

    if rebin is None:
        for src, bs_pc in zip(beam_scale_labels, beam_scales_pc):
            if np.isnan(bs_pc):
                continue
            if bs_pc > native_res:
                res_list.append((src, float(bs_pc)))

    # preserve order, unique
    res_list = list(dict.fromkeys(res_list))

    for src, res in res_list:
        assert src == "native" or src in beam_scale_labels

    # ---------- smoothing ----------
    for src, res in res_list:

        image_copy = image_untrimmed.copy()
        error_map_copy = error_map_untrimmed.copy()
        beam_scale_pc_copy = native_res

        BMAJ_new = BMAJ
        BMIN_new = BMIN

        smooth_factor = res / beam_scale_pc_copy

        if res is not None and smooth_factor > 1:
            pixel_scale_pc = pixel_scale_arcsec * pc_per_arcsec
            sigma_kernel_pc = np.sqrt(res**2 - beam_scale_pc_copy**2)
            sigma_kernel_pix = sigma_kernel_pc / pixel_scale_pc

            image_copy = gaussian_filter(image_copy, sigma=sigma_kernel_pix)
            error_map_copy = gaussian_filter(error_map_copy, sigma=sigma_kernel_pix)

            beam_scale_pc_copy = res
            BMAJ_new = beam_scale_pc_copy / (pc_per_arcsec * 3600)
            BMIN_new = BMAJ_new

        images.append(image_copy)
        errormaps.append(error_map_copy)
        BMAJs.append(BMAJ_new)
        BMINs.append(BMIN_new)
        res_values.append(float(res))
        res_sources.append(src)


    ######################## carry out manual rebin if missing ########################
    if manual_rebin:
        smooth_factor = rebin / native_res
        if rebin is not None and smooth_factor > 1:
            pixel_scale_pc = pixel_scale_arcsec * pc_per_arcsec
            sigma_kernel_pc = np.sqrt(rebin**2 - native_res**2)
            sigma_kernel_pix = sigma_kernel_pc / pixel_scale_pc

            image_rb = gaussian_filter(image_untrimmed, sigma=sigma_kernel_pix)
            error_rb = gaussian_filter(error_map_untrimmed, sigma=sigma_kernel_pix)

            BMAJ_rb = rebin / (pc_per_arcsec * 3600)
            BMIN_rb = BMAJ_rb

            images.append(image_rb)
            errormaps.append(error_rb)
            BMAJs.append(BMAJ_rb)
            BMINs.append(BMIN_rb)
            res_values.append(float(rebin))
            res_sources.append("rebin")
        else:
            print(
                f"No rebinning applied for {name}: requested rebin {rebin} pc "
                f"is not larger than beam scale {native_res:.2f} pc."
            )

    ##################################################################################

    rows = []

    for img_idx, (
            image_untrimmed,
            error_map_untrimmed,
            BMAJ,
            BMIN,
            res_pc,
            res_src
        ) in enumerate(zip(images, errormaps, BMAJs, BMINs, res_values, res_sources)):

        beam_scale_pc = res_pc
        R_21, R_31, alpha_CO = 0.65, 0.32, 4.35
        R_pixel = int(R_kpc * (206.265 / D_Mpc) / pixel_scale_arcsec)

        pixel_scale_arcsec = np.abs(header.get("CDELT1", 0)) * 3600
        pixel_area_arcsec2 = pixel_scale_arcsec**2
        beam_arcsec = np.sqrt(np.abs(BMAJ * BMIN)) * 3600
        beam_area_arcsec2 = (np.pi/(4*np.log(2)))*(BMAJ*3600)*(BMIN*3600)
        pc_per_arcsec = (D_Mpc * 1e6) / 206265
        beam_scale_pc = beam_arcsec * pc_per_arcsec
        beam_area_pc2 = beam_area_arcsec2 * pc_per_arcsec**2
        pixel_scale_pc = pixel_scale_arcsec * pc_per_arcsec
        pixel_area_pc2 = pixel_scale_pc**2
        pixels_per_beam = beam_area_arcsec2/pixel_area_arcsec2

        print(name,beam_scale_pc,'pc res')
        print('beam_area_arcsec2', beam_area_arcsec2,'arcsec2')
        print('beam_area_pc2', beam_area_pc2,'pc2')
        print('LCO factor using omega_beam (should be equal to beam_area_pc2)',beam_area_arcsec2*23.5*D_Mpc**2)
        print('LCO factor from using physical pixel size',pixel_area_pc2*pixels_per_beam)
        print('JYTOK=',jy_per_K)
        print('flux factor', 1/(pixels_per_beam*jy_per_K))


        gal_cen = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs')
        wcs_full = WCS(header)

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
        error_map = np.full_like(image, np.nan)
        mask = np.ones_like(image, dtype=bool)

        x1i, x2i = max(0, x1), min(nx_full, x2)
        y1i, y2i = max(0, y1), min(ny_full, y2)

        xp1, xp2 = x1i - x1, x1i - x1 + (x2i - x1i)
        yp1, yp2 = y1i - y1, y1i - y1 + (y2i - y1i)

        image[yp1:yp2, xp1:xp2] = image_untrimmed[y1i:y2i, x1i:x2i]
        error_map[yp1:yp2, xp1:xp2] = error_map_untrimmed[y1i:y2i, x1i:x2i]
        mask[yp1:yp2, xp1:xp2] = mask_untrimmed[y1i:y2i, x1i:x2i]

        # ---------- NaN handling ----------
        nan_pixels = np.isnan(image)
        if nan_pixels.any():
            image[nan_pixels] = 0.0
            mask[nan_pixels] = False
            error_map[nan_pixels] = np.nanmean(error_map)

        emission_pixels = np.count_nonzero(np.abs(image) > 1e-10)
        emission_fraction = emission_pixels / image.size

        # ---------- Update WCS ----------
        wcs_trimmed = wcs_full.deepcopy()
        wcs_trimmed.wcs.crpix[0] -= x1
        wcs_trimmed.wcs.crpix[1] -= y1

        image_nd = NDData(data=image, wcs=wcs_trimmed)

        # ---------- Flux mask ----------
        if flux_mask == True:
            total_flux = np.nansum(image[~mask])
            f = 1.0
            ratio = 1.0
            while ratio > 0.9:
                flux_mask_90, flux_aperture_90, R_90 = make_projected_region_mask(
                    image.shape, R_kpc * f * 1.42 , pc_per_arcsec, pixel_scale_arcsec, PA, I
                )
                flux_90 = np.nansum(image[flux_mask_90 & ~mask])
                ratio = flux_90 / total_flux if total_flux > 0 else 0.0
                f -= 0.001

            print(
                f"Flux in aperture = {ratio} of total. R_90 (kpc): ",
                round(R_90, 2)
            )
            mask = flux_mask_90 | mask
            aperture_to_plot = flux_aperture_90
        else:
            aperture_to_plot = None

        # ---------- Plot ----------
        if isolate is None or 'plot' in isolate:
            plot_moment_map(
                image_nd, output_dir, name,
                BMAJ, BMIN, R_kpc, rebin,
                PHANGS_mask, flux_mask,
                aperture=aperture_to_plot, res_src=res_src
            )

        if isolate == None or any(m in isolate for m in ['gini','asym','smooth','conc','tmass','LCO','mw','aw','clump']):

            # Generate Monte-Carlo images (full set)
            N_MC = 1000
            images_mc = generate_random_images(image, error_map, n_iter=N_MC)

            # ---- PARALLEL MC PROCESSING HERE ----
            dtype = image.dtype  # e.g. np.float64
            shape = image.shape

            shm_img = shared_memory.SharedMemory(create=True, size=image.nbytes)
            shm_err = shared_memory.SharedMemory(create=True, size=error_map.nbytes)
            # write into shared memory
            shm_array_img = np.ndarray(shape, dtype=dtype, buffer=shm_img.buf)
            shm_array_err = np.ndarray(shape, dtype=dtype, buffer=shm_err.buf)
            shm_array_img[:] = image[:]       # copy once
            shm_array_err[:] = error_map[:]   # copy once

            cpu = min( max(1, multiprocessing.cpu_count()-1), 8 )  # e.g., reserve one core
            iters_per_worker = [N_MC // cpu] * cpu
            for i in range(N_MC % cpu):
                iters_per_worker[i] += 1

            metric_kwargs_small = dict(
                name=name,
                co32=co32,
                pixel_area_pc2=pixel_area_pc2,
                beam_area_pc2=beam_area_pc2,
                beam_scale_pc=beam_scale_pc,
                pixel_area_arcsec2=pixel_area_arcsec2,
                beam_area_arcsec2=beam_area_arcsec2,
                D_Mpc=D_Mpc,
                R_21=R_21,
                R_31=R_31,
                alpha_CO=alpha_CO,
                pc_per_arcsec=pc_per_arcsec,
                pixel_scale_arcsec=pixel_scale_arcsec,
                jy_per_K = jy_per_K
            )

            with ProcessPoolExecutor(max_workers=cpu) as ex:
                futures = []
                for w, n_iter_chunk in enumerate(iters_per_worker):
                    seed = np.random.SeedSequence().entropy + w
                    futures.append(ex.submit(
                        process_mc_chunk_shm,
                        n_iter_chunk,
                        shm_img.name,
                        shm_err.name,
                        shape,
                        str(dtype),
                        mask,
                        metric_kwargs_small,
                        isolate,
                        seed
                    ))
                results = [f.result() for f in futures]

            # When done, unlink the shared memory
            shm_img.close()
            shm_img.unlink()
            shm_err.close()
            shm_err.unlink()

            def merge_global(metric_name):
                """Concatenate raw MC values across chunks and compute global stats."""
                
                # Collect lists from every chunk
                all_values = []
                for r in results:
                    if metric_name in r:
                        all_values.extend(r[metric_name])

                # Convert to array
                arr = np.array(all_values, dtype=float)

                # Handle case where everything failed
                if len(arr) == 0 or np.all(np.isnan(arr)):
                    print(f"All MC calculations failed for metric {metric_name} in galaxy {name}.")
                    return np.nan, np.nan

                # Global median and std
                return float(np.nanmedian(arr)), float(np.nanstd(arr))

            gini, gini_err = merge_global("gini")
            asym, asym_err = merge_global("asym")
            smooth, smooth_err = merge_global("smooth")
            conc, conc_err = merge_global("conc")
            total_mass, total_mass_err = merge_global("tmass")
            SCOdv, SCOdv_err = merge_global("SCOdv")
            LCO, LCO_err = merge_global("LCO")
            LCO_JCMT, LCO_JCMT_err = merge_global("LCO_JCMT")
            LCO_APEX, LCO_APEX_err = merge_global("LCO_APEX")
            mw_sd, mw_sd_err = merge_global("mw")
            aw_sd, aw_sd_err = merge_global("aw")
            clump, clump_err = merge_global("clump")
        else:
            print("Skipping metric calculations as per isolate parameter.")
            gini, gini_err = np.nan, np.nan
            asym, asym_err = np.nan, np.nan
            smooth, smooth_err = np.nan, np.nan
            conc, conc_err = np.nan, np.nan
            total_mass, total_mass_err = np.nan, np.nan
            SCOdv, SCOdv_err = np.nan, np.nan,
            LCO, LCO_err = np.nan, np.nan
            LCO_JCMT, LCO_JCMT_err = np.nan, np.nan
            LCO_APEX, LCO_APEX_err = np.nan, np.nan
            mw_sd, mw_sd_err = np.nan, np.nan
            aw_sd, aw_sd_err = np.nan, np.nan
            clump, clump_err = np.nan, np.nan

        if isolate == None or 'expfit' in isolate:

            # Radial profile unchanged
            try:
                radii, profile, profile_err = radial_profile_with_errors(image, error_map, mask, nbins=10)
                valid = np.isfinite(profile) & np.isfinite(profile_err)
                radii, profile, profile_err = radii[valid], profile[valid], profile_err[valid]
            except:
                radii, profile, profile_err = np.array([]), np.array([]), np.array([])
                print(f"Radial profile extraction failed for {name}.")

            try:
                popt, pcov = curve_fit(exp_profile, radii, profile, sigma=profile_err,
                                    absolute_sigma=True, p0=[np.max(profile), 20], maxfev=2000)
                perr = np.sqrt(np.diag(pcov))
                sigma0 = f"{popt[0]:.2e} ± {perr[0]:.2e}"
                rs_pc = popt[1] * pc_per_arcsec * pixel_scale_arcsec
                rs_pc_err = perr[1] * pc_per_arcsec * pixel_scale_arcsec
                rs = f"{rs_pc:.2f} ± {rs_pc_err:.2f}"

                # --- Create decently sized figure and control font sizes ---
                plt.figure(figsize=(8, 6))           # Larger canvas
                plt.rcParams.update({
                    'font.size': 10,                 # base font size
                    'axes.titlesize': 12,            # title
                    'axes.labelsize': 11,            # axis labels  
                    'xtick.labelsize': 10,
                    'ytick.labelsize': 10,
                    'legend.fontsize': 10
    })

                bin_widths = np.diff(np.linspace(0, radii.max(), len(radii)+1))
                bin_widths_pc = bin_widths * pixel_scale_arcsec * pc_per_arcsec
                radii_pc = radii * pixel_scale_arcsec * pc_per_arcsec
                plt.errorbar(radii_pc, profile, yerr=profile_err, fmt='x', label="Data", capsize=3, xerr=bin_widths_pc / 2)
                plt.plot(radii_pc, exp_profile(radii, *popt), label="Fit", color='orange')
                plt.xlabel("Radius (pc)")
                plt.ylabel("Integrated intensity [Jy/beam km/s]")
                plt.title(name)
                plt.legend()
                plt.tight_layout()
                plot_path = os.path.join(output_dir, f"{name}_{PHANGS_mask}_{rebin}_{R_kpc}kpc_expfit.png")
                if save_exp == True:
                    plt.savefig(plot_path,dpi=200)
                plt.close()

            except:
                sigma0, rs = "fit failed", "fit failed"
        else:
            sigma0, rs = np.nan, np.nan
        
        print('L\'CO=',round(LCO, 3))
        print("flux Jykm/s=",round(SCOdv,3))


        # ---------- assemble row ----------
        rows.append({
            "Galaxy": name,
            "Resolution (pc)": round(res_pc, 2),
            "resolution_source": res_src,
            "pc_per_arcsec": round(pc_per_arcsec, 1),
            "RA (deg)": RA,
            "DEC (deg)": DEC,
            "PA (deg)": PA,
            "Inclination (deg)": I,
            "Gini": round(gini, 3), "Gini_err": round(gini_err, 3),
            "Asymmetry": round(asym, 3), "Asymmetry_err": round(asym_err, 3),
            "Smoothness": round(smooth, 3), "Smoothness_err": round(smooth_err, 3),
            "Concentration": round(conc, 3), "Concentration_err": round(conc_err, 3),
            "Sigma0 (Jy/beam km/s)": sigma0,
            "rs (pc)": rs,
            "clumping_factor": round(clump, 3), "clumping_factor_err": round(clump_err, 3),
            "total_mass (M_sun)": round(total_mass, 2),
            "total_mass_err (M_sun)": round(total_mass_err, 2),
            "L'CO (K km_s pc2)": round(LCO, 3),
            "L'CO_err (K km_s pc2)": round(LCO_err, 3),
            "L'CO_JCMT (K km s pc2)": round(LCO_JCMT, 3),
            "L'CO_JCMT_err (K km s pc2)": round(LCO_JCMT_err, 3),
            "L'CO_APEX (K km s pc2)": round(LCO_APEX, 3),
            "L'CO_APEX_err (K km s pc2)": round(LCO_APEX_err, 3),
            "mass_weighted_sd": round(mw_sd, 1),
            "mass_weighted_sd_err": round(mw_sd_err, 1),
            "area_weighted_sd": round(aw_sd, 1),
            "area_weighted_sd_err": round(aw_sd_err, 1),
            "emission_pixels": emission_pixels,
            "emission_fraction": emission_fraction
        })

    return rows

# ------------------ Parallel Directory Processing ------------------

def process_directory(
    outer_dir,
    llamatab,
    base_output_dir,
    co32,
    rebin=None,
    mask='broad',
    R_kpc=1,
    flux_mask=False,
    isolate=None
):
    print(
        f"Processing directory: {outer_dir} "
        f"(CO32={co32}, rebin={rebin}, mask={mask}, R_kpc={R_kpc}), "
        f"isolate={isolate})"
    )

    valid_names = set(llamatab['id'])
    subdirs = [
        d for d in os.listdir(outer_dir)
        if os.path.isdir(os.path.join(outer_dir, d))
    ]

    args_list = []
    meta_info = []

    for name in subdirs:
        subdir = os.path.join(outer_dir, name)
        manual_rebin = False

        # ---------------- File selection logic (UNCHANGED) ----------------
        if rebin is not None and not co32:
            if name in ["NGC4388", "NGC6814", "NGC5728"]:
                continue

            rebinned = os.path.join(
                subdir, f"{name}_12m_co21_{rebin}pc_{mask}_mom0.fits"
            )
            if os.path.exists(rebinned):
                mom0_file = rebinned
                emom0_file = os.path.join(
                    subdir, f"{name}_12m_co21_{rebin}pc_{mask}_emom0.fits"
                )
            else:
                print(f'{rebin} pc for {name}, using native res files or rebinning manually')
                manual_rebin = True
                mom0_file = os.path.join(
                    subdir, f"{name}_12m_co21_{mask}_mom0.fits"
                )
                emom0_file = os.path.join(
                    subdir, f"{name}_12m_co21_{mask}_emom0.fits"
                )

        elif rebin is not None and co32:
            if name not in ["NGC4388", "NGC6814", "NGC5728"]:
                continue

            rebinned = os.path.join(
                subdir, f"{name}_12m_co32_{rebin}pc_{mask}_mom0.fits"
            )
            if os.path.exists(rebinned):
                mom0_file = rebinned
                emom0_file = os.path.join(
                    subdir, f"{name}_12m_co32_{rebin}pc_{mask}_emom0.fits"
                )
            else:
                print(f'{rebin} pc for {name}, using native res files or rebinning manually')
                manual_rebin = True
                mom0_file = os.path.join(
                    subdir, f"{name}_12m_co32_{mask}_mom0.fits"
                )
                emom0_file = os.path.join(
                    subdir, f"{name}_12m_co32_{mask}_emom0.fits"
                )

        elif not rebin and not co32:
            if name in ["NGC4388", "NGC6814", "NGC5728"]:
                continue

            mom0_file = os.path.join(
                subdir, f"{name}_12m_co21_{mask}_mom0.fits"
            )
            emom0_file = os.path.join(
                subdir, f"{name}_12m_co21_{mask}_emom0.fits"
            )

        else:
            if name not in ["NGC4388", "NGC6814", "NGC5728"]:
                continue

            mom0_file = os.path.join(
                subdir, f"{name}_12m_co32_{mask}_mom0.fits"
            )
            emom0_file = os.path.join(
                subdir, f"{name}_12m_co32_{mask}_emom0.fits"
            )

        # ---------------- Group classification (UNCHANGED) ----------------
        try:
            type_val = llamatab[llamatab['id'] == name]['type'][0]
        except Exception:
            type_val = 'aux'

        group = (
            "inactive" if type_val == "i"
            else "aux" if type_val == "aux"
            else "AGN"
        )

        output_dir = os.path.join(base_output_dir, group)
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(mom0_file):
            print(f"Skipping {name}: required files not found")
            continue

        if not os.path.exists(emom0_file):
            emom0_file = np.nan

        args_list.append(
            (
                mom0_file,
                emom0_file,
                outer_dir,
                subdir,
                output_dir,
                co32,
                rebin,
                mask,
                R_kpc,
                flux_mask
            )
        )
        meta_info.append((name, group, output_dir, manual_rebin))

    # ---------------- Serial execution (UNCHANGED) ----------------
    results_raw = []
    images_too_small = []

    for args, meta in zip(args_list, meta_info):
        name, group, _, manual_rebin = meta
        res = process_file(
            args,
            images_too_small,
            isolate=isolate,
            manual_rebin=manual_rebin
        )
        if res is not None:
            # res is a LIST of rows
            for row in res:
                row["id"] = name
                row["group"] = group
            results_raw.extend(res)

    if len(results_raw) == 0:
        print("No results to save.")
        return

    df = pd.DataFrame(results_raw)

    # ---------------- Isolate handling (UNCHANGED LOGIC) ----------------
    isolate_colmap = {
        "gini":  ["Gini", "Gini_err"],
        "asym":  ["Asymmetry", "Asymmetry_err"],
        "smooth":["Smoothness", "Smoothness_err"],
        "conc":  ["Concentration", "Concentration_err"],
        "tmass": ["total_mass (M_sun)", "total_mass_err (M_sun)"],
        "LCO":   ["L'CO (K km s pc2)", "L'CO_err (K km s pc2)"],
        "mw":    ["mass_weighted_sd", "mass_weighted_sd_err"],
        "aw":    ["area_weighted_sd", "area_weighted_sd_err"],
        "clump": ["clumping_factor", "clumping_factor_err"],
        "expfit":["Sigma0 (Jy/beam km/s)", "rs (pc)"],
        "plot":  []
    }

    if isolate is None:
        isolates_list = None
    elif isinstance(isolate, (list, tuple, set)):
        isolates_list = [str(i) for i in isolate]
    else:
        isolates_list = [str(isolate)]

    if isolates_list is None:
        cols_updated = None
    else:
        cols_updated = []
        for iso in isolates_list:
            cols_updated += isolate_colmap.get(iso, [])
        cols_updated = list(dict.fromkeys(cols_updated))

    # ---------------- CSV writing (row identity = id + Resolution) ----------------
    for group in ["AGN", "inactive", "aux"]:
        group_df = df[df["group"] == group].copy()
        if group_df.empty:
            continue

        outdir = os.path.join(base_output_dir, group)
        os.makedirs(outdir, exist_ok=True)

        if rebin is not None:
            outfile = (
                f"gas_analysis_summary_{rebin}pc_"
                f"{'flux90_' if flux_mask else ''}"
                f"{mask}_{R_kpc}kpc.csv"
            )
        else:
            outfile = (
                f"gas_analysis_summary_"
                f"{'flux90_' if flux_mask else ''}"
                f"{mask}_{R_kpc}kpc.csv"
            )

        outfile = os.path.join(outdir, outfile)

        if not os.path.exists(outfile):
            if isolates_list is not None and "plot" in isolates_list:
                continue
            group_df.to_csv(outfile, index=False)
            continue

        existing_df = pd.read_csv(outfile)

        key_cols = ["id", "Resolution (pc)"]
        new_keys = set(tuple(x) for x in group_df[key_cols].values)

        if cols_updated is None:
            mask_keep = ~existing_df[key_cols].apply(tuple, axis=1).isin(new_keys)
            merged = pd.concat([existing_df[mask_keep], group_df], ignore_index=True)
            merged.to_csv(outfile, index=False)
            continue

        if "plot" in (isolates_list or []):
            continue

        for _, new_row in group_df.iterrows():
            key = tuple(new_row[k] for k in key_cols)
            mask_existing = existing_df[key_cols].apply(tuple, axis=1) == key

            if mask_existing.any():
                old_row = existing_df[mask_existing].iloc[0]
                for col in existing_df.columns:
                    if col in key_cols or col == "group":
                        continue
                    if col not in cols_updated:
                        group_df.loc[
                            (group_df[key_cols] == pd.Series(key, index=key_cols)).all(axis=1),
                            col
                        ] = old_row[col]

        existing_df = existing_df[
            ~existing_df[key_cols].apply(tuple, axis=1).isin(new_keys)
        ]
        merged = pd.concat([existing_df, group_df], ignore_index=True)
        merged.to_csv(outfile, index=False)

    print("images too small:", images_too_small)

### ------------------ Matched Pair Construction ------------------ ####

inactive_by_num = {
    1: "NGC 3351",
    2: "NGC 3175",
    3: "NGC 4254",
    4: "ESO 208-G021",
    5: "NGC 1079",
    6: "NGC 1947",
    7: "NGC 5921",
    8: "NGC 2775",
    9: "ESO 093-G003",
    10: "NGC 718",
    11: "NGC 3717",
    12: "NGC 5845",
    13: "NGC 7727",
    14: "IC 4653",
    15: "NGC 4260",
    16: "NGC 5037",
    17: "NGC 4224",
    18: "NGC 3749",
    19: "NGC 1375",
    20: "NGC 1315",
}

# Active galaxies with the exact numbers shown in their corners (from the image; left panel)
active_to_nums = {
    "NGC 1365": [7],
    "NGC 7582": [11, 17],
    "NGC 6814": [3],
    "NGC 4388": [11],
    "NGC 7213": [8],
    "MCG-06-30-015": [12],
    "NGC 5506": [2, 15, 16, 17, 18],
    "NGC 2110": [4, 6],
    "NGC 3081": [5, 9, 10],
    "MCG-05-23-016": [5, 19],
    "ESO 137-G034": [13],
    "NGC 2992": [2, 15, 16, 17, 18],
    "NGC 4235": [2, 16, 17, 18],
    "NGC 4593": [1, 8],
    "NGC 7172": [15, 16, 17],
    "NGC 3783": [10],
    "ESO 021-G004": [17],
    "NGC 5728": [13, 17],
    "MCG-05-14-012": [20],
}

# Build all pairs (one row per link)
rows = []
for active, nums in active_to_nums.items():
    for n in nums:
        rows.append({
            "pair_id": n,
            "Active Galaxy": active,
            "Inactive Galaxy": inactive_by_num[n],
        })

df_pairs = pd.DataFrame(rows).sort_values(["pair_id", "Active Galaxy"]).reset_index(drop=True)
print(len(df_pairs), "matched pairs constructed.")

llamatab = Table.read('/data/c3040163/llama/llama_main_properties.fits', format='fits')



# ------------------ Main ------------------

if __name__ == '__main__':
    llamatab = Table.read('/data/c3040163/llama/llama_main_properties.fits', format='fits')
    base_output_dir = '/data/c3040163/llama/alma/gas_analysis_results'
    isolate = None

    # CO(2-1)
    print("Starting CO(2-1) analysis...")
    # process_directory(outer_dir_co21, llamatab, base_output_dir, co32=False,rebin=120,mask='strict',R_kpc=1.5,flux_mask=True,isolate=isolate)
    # process_directory(outer_dir_co21, llamatab, base_output_dir, co32=False,rebin=120,mask='strict',R_kpc=1.5,flux_mask=False,isolate=isolate)

    # process_directory(outer_dir_co21, llamatab, base_output_dir, co32=False,rebin=None,mask='broad',R_kpc=1.5,isolate=isolate)
    process_directory(outer_dir_co21, llamatab, base_output_dir, co32=False,rebin=None,mask='strict',R_kpc=1.5,isolate=isolate)

    # process_directory(outer_dir_co21, llamatab, base_output_dir, co32=False,rebin=None,mask='broad',R_kpc=1,isolate=isolate)
    # process_directory(outer_dir_co21, llamatab, base_output_dir, co32=False,rebin=None,mask='strict',R_kpc=1,isolate=isolate)

    # process_directory(outer_dir_co21, llamatab, base_output_dir, co32=False,rebin=None,mask='broad',R_kpc=0.3,isolate=isolate)
    # process_directory(outer_dir_co21, llamatab, base_output_dir, co32=False,rebin=None,mask='strict',R_kpc=0.3,isolate=isolate)


    # # CO(3-2)
    co32 = True
    print("Starting CO(3-2) analysis...")
    # process_directory(outer_dir_co32, llamatab, base_output_dir, co32=True,rebin=120,mask='strict',R_kpc=1.5,isolate=isolate,flux_mask=True)
    # process_directory(outer_dir_co32, llamatab, base_output_dir, co32=True,rebin=120,mask='strict',R_kpc=1.5,isolate=isolate,flux_mask=False)

    # process_directory(outer_dir_co32, llamatab, base_output_dir, co32=True,rebin=None,mask='broad',R_kpc=1.5,isolate=isolate)
    # process_directory(outer_dir_co32, llamatab, base_output_dir, co32=True,rebin=None,mask='strict',R_kpc=1.5,isolate=isolate)

    # process_directory(outer_dir_co32, llamatab, base_output_dir, co32=True,rebin=None,mask='broad',R_kpc=1,isolate=isolate)
    # process_directory(outer_dir_co32, llamatab, base_output_dir, co32=True,rebin=None,mask='strict',R_kpc=1,isolate=isolate)

    # process_directory(outer_dir_co32, llamatab, base_output_dir, co32=True,rebin=None,mask='broad',R_kpc=0.3,isolate=isolate)
    # process_directory(outer_dir_co32, llamatab, base_output_dir, co32=True,rebin=None,mask='strict',R_kpc=0.3,isolate=isolate)


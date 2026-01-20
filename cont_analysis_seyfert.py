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
import time
from multiprocessing import shared_memory
from matplotlib.patches import Ellipse
from astroquery.simbad import Simbad
simbad = Simbad()
from astroquery.vizier import Vizier
import warnings
from astropy.wcs import FITSFixedWarning

warnings.filterwarnings("ignore", category=FITSFixedWarning)

np.seterr(all='ignore')
co32 = False
LLAMATAB = None

# ------------------ Monte Carlo Helpers ------------------

def generate_random_images(image, error_map, n_iter=1000, seed=None):
    """
    Generate Monte Carlo realizations of an image.

    If no valid error_map is provided (None, all-NaN, or non-physical),
    return None to signal that MC should be skipped.
    """
    rng = np.random.default_rng(seed)

    # No error map provided
    if error_map is None:
        return None

    error_map = np.asarray(error_map)

    # Shape safety
    if error_map.shape != image.shape:
        return None

    # Require at least one finite error value
    if not np.any(np.isfinite(error_map)):
        return None

    # Non-physical uncertainties → skip MC
    if np.any(error_map < 0):
        return None

    return rng.normal(
        loc=image,
        scale=error_map,
        size=(n_iter, *image.shape)
    )


def process_mc_chunk_shm(n_iter_chunk, shm_name_image, shm_name_error, shape, dtype_str, mask, metric_kwargs, isolate=None, seed=None):
    """
    n_iter_chunk: number of MC images this worker should generate
    shm_name_image: shared memory name for the base image (float64/float32)
    shm_name_error: shared memory name for the error map
    shape: (ny, nx)
    dtype_str: e.g. 'float64' or 'float32'
    mask: boolean array (this will be pickled but is small compared to images)
    metric_kwargs: small dict of params (picklable)
    """
    # attach to shared memory blocks
    shm_img = shared_memory.SharedMemory(name=shm_name_image)
    shm_err = shared_memory.SharedMemory(name=shm_name_error)
    dtype = np.dtype(dtype_str)
    image = np.ndarray(shape, dtype=dtype, buffer=shm_img.buf)
    errmap = np.ndarray(shape, dtype=dtype, buffer=shm_err.buf)

    rng = np.random.default_rng(seed)

    # lists to collect values per-image
    # gini_vals = []
    # asym_vals = []
    # smooth_vals = []
    # conc_vals = []
    # tm_vals = []
    # mw_vals = []
    # aw_vals = []
    # clump_vals = []
    # LCO_vals = []
    # LCO_JCMT_vals = []
    # LCO_APEX_vals = []
    cont_power_jybeam_vals = []

    for i in range(n_iter_chunk):
        # create one MC realisation locally: image + gaussian noise scaled by errmap
        if not np.any(errmap > 0):
            mc_img = image.copy()
        else:
            noise = rng.standard_normal(size=shape) * errmap
            mc_img = image + noise

        # compute metrics (use your existing functions; catch exceptions per metric)
        # try:
        #     gini_vals.append(gini_single(mc_img, mask))
        # except Exception:
        #     gini_vals.append(np.nan)
        # try:
        #     asym_vals.append(asymmetry_single(mc_img, mask))
        # except Exception:
        #     asym_vals.append(np.nan)
        # try:
        #     smooth_vals.append(smoothness_single(mc_img, mask,
        #                                          pc_per_arcsec=metric_kwargs["pc_per_arcsec"],
        #                                          pixel_scale_arcsec=metric_kwargs["pixel_scale_arcsec"]))
        # except Exception:
        #     smooth_vals.append(np.nan)
        # try:
        #     conc_vals.append(concentration_single(mc_img, mask,
        #                                           pixel_scale_arcsec=metric_kwargs["pixel_scale_arcsec"],
        #                                           pc_per_arcsec=metric_kwargs["pc_per_arcsec"]))
        # except Exception:
        #     conc_vals.append(np.nan)
        # try:
        #     tm_vals.append(total_mass_single(mc_img, mask,
        #                                      metric_kwargs["pixel_area_pc2"],
        #                                      metric_kwargs["R_21"], metric_kwargs["R_31"],
        #                                      metric_kwargs["alpha_CO"],
        #                                      metric_kwargs["name"],
        #                                      co32=metric_kwargs["co32"]))
        # except Exception:
        #     tm_vals.append(np.nan)
        # try:
        #     mw_vals.append(mass_weighted_sd_single(mc_img, mask,
        #                                            metric_kwargs["pixel_area_pc2"],
        #                                            metric_kwargs["R_21"], metric_kwargs["R_31"],
        #                                            metric_kwargs["alpha_CO"],
        #                                            metric_kwargs["name"],
        #                                            co32=metric_kwargs["co32"]))
        # except Exception:
        #     mw_vals.append(np.nan)
        # try:
        #     aw_vals.append(area_weighted_sd_single(mc_img, mask,
        #                                            metric_kwargs["pixel_area_pc2"],
        #                                            metric_kwargs["R_21"], metric_kwargs["R_31"],
        #                                            metric_kwargs["alpha_CO"],
        #                                            metric_kwargs["name"],
        #                                            co32=metric_kwargs["co32"]))
        # except Exception:
        #     aw_vals.append(np.nan)
        # try:
        #     clump_vals.append(clumping_factor_single(mc_img, mask,
        #                                              metric_kwargs["pixel_area_pc2"],
        #                                              metric_kwargs["R_21"], metric_kwargs["R_31"],
        #                                              metric_kwargs["alpha_CO"],
        #                                              metric_kwargs["name"],
        #                                              co32=metric_kwargs["co32"]))
        # except Exception:
        #     clump_vals.append(np.nan)
        # # LCO if you added it:
        # try:
        #     LCO_vals.append(LCO_single(mc_img, mask,
        #                         metric_kwargs["pixel_scale_arcsec"],
        #                         metric_kwargs["pixel_area_pc2"],
        #                         metric_kwargs["R_21"], metric_kwargs["R_31"],
        #                         metric_kwargs["alpha_CO"],
        #                         metric_kwargs["name"],
        #                         co32=metric_kwargs["co32"]))
        # except Exception:
        #     LCO_vals.append(np.nan)

        # try:
        #     LCO_JCMT_vals.append(LCO_single_JCMT(mc_img, mask,
        #                         metric_kwargs["pixel_scale_arcsec"],
        #                         metric_kwargs["pixel_area_pc2"],
        #                         metric_kwargs["R_21"], metric_kwargs["R_31"],
        #                         metric_kwargs["alpha_CO"],
        #                         metric_kwargs["name"],
        #                         co32=metric_kwargs["co32"]))
        # except Exception:
        #     LCO_JCMT_vals.append(np.nan)

        # try:
        #     LCO_APEX_vals.append(LCO_single_APEX(mc_img, mask,
        #                         metric_kwargs["pixel_scale_arcsec"],
        #                         metric_kwargs["pixel_area_pc2"],
        #                         metric_kwargs["R_21"], metric_kwargs["R_31"],
        #                         metric_kwargs["alpha_CO"],
        #                         metric_kwargs["name"],
        #                         co32=metric_kwargs["co32"]))
        # except Exception:
        #     LCO_APEX_vals.append(np.nan)

        try:
            cont_power_jybeam(mc_img, mask,
                              metric_kwargs["pixel_scale_arcsec2"],
                              metric_kwargs["beam_area_arcsec2"],
                              **metric_kwargs)
            
        except Exception:
            cont_power_jybeam_vals.append(np.nan)

    # close shared memory views (do NOT unlink here)
    shm_img.close()
    shm_err.close()

    # return lists (pickling these lists of scalars is small)
    return {
        # "gini": gini_vals,
        # "asym": asym_vals,
        # "smooth": smooth_vals,
        # "conc": conc_vals,
        # "tmass": tm_vals,
        # "mw": mw_vals,
        # "aw": aw_vals,
        # "clump": clump_vals,
        # "LCO": LCO_vals,
        # "LCO_JCMT": LCO_JCMT_vals,
        # "LCO_APEX": LCO_APEX_vals,
        "cont_power_jybeam": cont_power_jybeam_vals,
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

def total_mass_single(image, mask, pixel_area_pc2, R_21, R_31, alpha_CO, name, co32=False, **kwargs):
    map_LprimeCO = image * pixel_area_pc2
    map_LprimeCO10 = map_LprimeCO / (R_31 if co32 else R_21)
    map_MH2 = alpha_CO * map_LprimeCO10
    return np.nansum(map_MH2[~mask])

def LCO_single(image, mask, pixel_scale_arcsec, pixel_area_pc2, R_21, R_31, alpha_CO, name, co32=False, **kwargs):
    map_LprimeCO = image * pixel_area_pc2
    if co32:
        map_LprimeCO = (map_LprimeCO / R_31) * R_21
    return np.nansum(map_LprimeCO[~mask])

def LCO_single_JCMT(image, mask, pixel_scale_arcsec, pixel_area_pc2, R_21, R_31, alpha_CO, name, co32=False, **kwargs):
    map_LprimeCO = image * pixel_area_pc2
    if co32:
        map_LprimeCO = (map_LprimeCO / R_31) * R_21
    y, x = np.indices(image.shape)
    center = (x.max() / 2, y.max() / 2)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r_arcsec = r * pixel_scale_arcsec
    r_JCMT = 20.4 / 2  # JCMT beam radius in arcsec
    map_LprimeCO_JCMT = map_LprimeCO * (r_arcsec <= r_JCMT)
    return np.nansum(map_LprimeCO_JCMT[~mask])

def LCO_single_APEX(image, mask, pixel_scale_arcsec, pixel_area_pc2, R_21, R_31, alpha_CO, name, co32=False, **kwargs):
    map_LprimeCO = image * pixel_area_pc2
    if co32:
        map_LprimeCO = (map_LprimeCO / R_31) * R_21
    y, x = np.indices(image.shape)
    center = (x.max() / 2, y.max() / 2)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r_arcsec = r * pixel_scale_arcsec
    r_APEX = 27.1 / 2  # APEX beam radius in arcsec
    map_LprimeCO_APEX = map_LprimeCO * (r_arcsec <= r_APEX)
    return np.nansum(map_LprimeCO_APEX[~mask])

def mass_weighted_sd_single(image, mask, pixel_area_pc2, R_21, R_31, alpha_CO, name, co32=False, **kwargs):
    map_LprimeCO = image * pixel_area_pc2
    map_LprimeCO10 = map_LprimeCO / (R_31 if co32 else R_21)
    map_MH2 = alpha_CO * map_LprimeCO10
    Sigma = map_MH2 / pixel_area_pc2
    Sigma = Sigma[~mask]
    if Sigma.size == 0:
        return np.nan
    numerator = np.sum(Sigma**2 * pixel_area_pc2)
    denominator = np.sum(Sigma * pixel_area_pc2)
    return numerator / denominator if denominator > 0 else np.nan

def area_weighted_sd_single(image, mask, pixel_area_pc2, R_21, R_31, alpha_CO, name, co32=False, **kwargs):
    map_LprimeCO = image * pixel_area_pc2
    map_LprimeCO10 = map_LprimeCO / (R_31 if co32 else R_21)
    map_MH2 = alpha_CO * map_LprimeCO10
    Sigma = map_MH2 / pixel_area_pc2
    Sigma = Sigma[~mask]
    if Sigma.size == 0:
        return np.nan
    total_area = Sigma.size * pixel_area_pc2
    return np.sum(Sigma * pixel_area_pc2) / total_area if total_area > 0 else np.nan

def clumping_factor_single(image, mask, pixel_area_pc2, R_21, R_31, alpha_CO, name, co32=False, **kwargs):
    mw = mass_weighted_sd_single(image, mask, pixel_area_pc2, R_21, R_31, alpha_CO, name, co32=co32)
    aw = area_weighted_sd_single(image, mask, pixel_area_pc2, R_21, R_31, alpha_CO, name, co32=co32)
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

def cont_power_jybeam(image, mask, pixel_scale_arcsec2, beam_area_arcsec2, **kwargs):
    valid = (~mask) & np.isfinite(image)
    if np.sum(valid) == 0:
        return np.nan
    total_flux_jy = np.nansum(image[valid]) * (pixel_scale_arcsec2) / beam_area_arcsec2
    print(f"contpower result: {total_flux_jy}")
    return total_flux_jy

# ----------------- Moment map plotting ------------------

def plot_moment_map(image, outfolder, name_short, BMAJ, BMIN, R_kpc, rebin, aperture=None, norm_type='linear'): 
    # Initialise plot
    plt.rcParams.update({'font.size': 35})
    fig = plt.figure(figsize=(18 , 18),constrained_layout=True)
    ax = fig.add_subplot(111, projection=image.wcs.celestial)
    ax.margins(x=0,y=0)
    # ax.set_axis_off()

    add_scalebar(ax,1/3600,label="1''",corner='top left',color='black',borderpad=0.5,size_vertical=0.5)
    add_beam(ax,major=BMAJ,minor=BMIN,angle=0,corner='bottom right',color='black',borderpad=0.5,fill=False,linewidth=3,hatch='///')

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
        if not os.path.exists(outfolder+f'/plots/'):
            os.makedirs(outfolder+f'/plots/')
        plt.savefig(outfolder+f'/plots/{R_kpc}_{rebin}_{name_short}.png',bbox_inches='tight',pad_inches=0.0)
    else:
        if not os.path.exists(outfolder+f'/plots/'):
            os.makedirs(outfolder+f'/plots/')
        plt.savefig(outfolder+f'/plots/{R_kpc}_no_rebin_{name_short}.png',bbox_inches='tight',pad_inches=0.0)
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

def process_file(args, images_too_small, isolate=None, manual_rebin=False, save_exp=False):
    cont_im_file, econt_im_file, subdir, output_dir, co32, rebin, R_kpc, flux_mask = args
    file = cont_im_file
    error_map_file = econt_im_file

    # Galaxy name extraction (now robust)
    base = os.path.basename(file)

    name = base.split("_12m")[0]

    # Load LLAMA table once per galaxy
    llamatab = Table.read('/data/c3040163/llama/llama_main_properties.fits', format='fits')

    # Skip incompatible galaxies
    if not co32 and name in ['NGC4388','NGC6814','NGC5728']:
        return None
    if co32 and name not in ['NGC4388','NGC6814','NGC5728']:
        return None

    print(f"Processing {name}...")

    # Load FITS
    image_untrimmed = fits.getdata(file, memmap=True)
    print(error_map_file)
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

    header = fits.getheader(file)
    try:
        D_Mpc = llamatab[llamatab['id'] == name]['D [Mpc]'][0]
    except:
        D_Mpc = np.nan

    try:
        I = llamatab[llamatab['id'] == name]['Inclination (deg)'][0]
        PA = llamatab[llamatab['id'] == name]['PA'][0]
    except:
        I = np.nan
        PA = np.nan

    ny, nx = image_untrimmed.shape
        # --- Convert RA/DEC galaxy center → pixel coordinates ---
    max_retries = 3

    for attempt in range(max_retries):
        if name.endswith("_phangs"):
            name_ned = name.split("_phangs")[0]
            try:
                Ned_table = Ned.query_object(name_ned)

                RA = Ned_table['RA'][0]
                DEC = Ned_table['DEC'][0]
                simbad.add_votable_fields('mesdistance')
                tbl = simbad.query_object(name_ned)
                D_Mpc = tbl['mesdistance.dist'][0]
                D_Mpc_unit = tbl['mesdistance.unit'][0]
                diam_table = Ned.get_table(name_ned, table='diameters')
                PA = diam_table['Position Angle'][0]
                try:
                    result = Vizier.query_object(name_ned, catalog="J/ApJS/197/21/cgs")
                    I = np.nanmedian(np.array(result[0]['i']))
                except:
                    result = Vizier.query_object(name_ned, catalog="VII/145/catalog")
                    I = np.nanmedian(np.array(result[0]['i']))

                break  # success, exit retry loop
            except (requests.exceptions.ConnectionError, RemoteServiceError, requests.exceptions.ReadTimeout) as e:
                print(f"⚠️  NED query failed for {name_ned} (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print("Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print("❌ All NED attempts failed.")
                    # fallback: if your FITS table already includes RA/DEC, use them

                    print(f"Skipping {name_ned} — no coordinates available.")
                    continue

        elif name.endswith("_wis"):
            name_ned = name.split("_wis")[0]
            try:
                Ned_table = Ned.query_object(name_ned)
                RA = Ned_table['RA'][0]
                DEC = Ned_table['DEC'][0]
                simbad.add_votable_fields('mesdistance')
                tbl = simbad.query_object(name_ned)
                D_Mpc = tbl['mesdistance.dist'][0]
                D_Mpc_unit = tbl['mesdistance.unit'][0]
                diam_table = Ned.get_table(name_ned, table='diameters')
                PA = diam_table['Position Angle'][0]
                try:
                    result = Vizier.query_object(name_ned, catalog="J/ApJS/197/21/cgs")
                    I = np.nanmedian(np.array(result[0]['i']))
                except:
                    result = Vizier.query_object(name_ned, catalog="VII/145/catalog")
                    I = np.nanmedian(np.array(result[0]['i']))

                break  # success, exit retry loop
            except (requests.exceptions.ConnectionError, RemoteServiceError, requests.exceptions.ReadTimeout) as e:
                print(f"⚠️  NED query failed for {name_ned} (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print("Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print("❌ All NED attempts failed.")
                    # fallback: if your FITS table already includes RA/DEC, use them

                    print(f"Skipping {name_ned} — no coordinates available.")
                    continue

        else:
            try:
                Ned_table = Ned.query_object(llamatab[llamatab['id'] == name]['name'][0])
                RA = Ned_table['RA'][0]
                DEC = Ned_table['DEC'][0]
                break  # success, exit retry loop
            except (requests.exceptions.ConnectionError, RemoteServiceError, requests.exceptions.ReadTimeout) as e:
                print(f"⚠️  NED query failed for {llamatab[llamatab['id'] == name]['name'][0]} (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print("Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print("❌ All NED attempts failed.")
                    # fallback: if your FITS table already includes RA/DEC, use them
                    if 'RA' in llamatab.colnames and 'DEC' in llamatab.colnames:
                        RA = llamatab[llamatab['id'] == name]['RA'][0]
                        DEC = llamatab[llamatab['id'] == name]['DEC'][0]
                        print(f"Using fallback RA/DEC from llamatab for {llamatab[llamatab['id'] == name]['name'][0]}.")
                    else:
                        print(f"Skipping {llamatab[llamatab['id'] == name]['name'][0]} — no coordinates available.")
                        continue

    pixel_scale_arcsec = np.abs(header.get("CDELT1", 0)) * 3600
    pixel_scale_arcsec2 = pixel_scale_arcsec**2
    pc_per_arcsec = (D_Mpc * 1e6) / 206265
    BMAJ = header.get("BMAJ", 0)
    BMIN = header.get("BMIN", 0)
    beam_area_arcsec2 = (np.pi / (4 * np.log(2))) * (BMAJ * 3600) * (BMIN * 3600)
    beam_scale_pc = np.sqrt(np.abs(BMAJ * BMIN)) * 3600 * pc_per_arcsec

    ######################## carry out manual rebin if missing ########################

    if manual_rebin:
        smooth_factor = rebin / beam_scale_pc
        if rebin is not None and smooth_factor > 1:
            pixel_scale_pc = pixel_scale_arcsec * pc_per_arcsec
            sigma_kernel_pc = np.sqrt(rebin**2 - beam_scale_pc**2)
            sigma_kernel_pix = sigma_kernel_pc / pixel_scale_pc
            from scipy.ndimage import gaussian_filter
            image_untrimmed = gaussian_filter(image_untrimmed, sigma=sigma_kernel_pix)
            error_map_untrimmed = gaussian_filter(error_map_untrimmed, sigma=sigma_kernel_pix)

            beam_scale_pc = rebin
            BMAJ = beam_scale_pc / (pc_per_arcsec * 3600)
            BMIN = BMAJ
        else:
            print(f"No rebinning applied for {name}: requested rebin {rebin} pc is not larger than beam scale {beam_scale_pc:.2f} pc.")
           
    ##################################################################################

    pixel_area_pc2 = (pixel_scale_arcsec * pc_per_arcsec)**2
    R_21, R_31, alpha_CO = 0.7, 0.31, 4.35
    R_pixel = int(R_kpc * (206.265 / D_Mpc) / pixel_scale_arcsec)

    gal_cen = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs')

    wcs_full = WCS(header)

    try:
        cx, cy = gal_cen.to_pixel(wcs_full)
        cx, cy = int(cx), int(cy)
    except Exception as e:
        print(f"WARNING: WCS conversion failed for {name}: {e}")
        print("Falling back to image center.")
        cx, cy = nx // 2, ny // 2

    # Target padded size (square of side 2*R_pixel)
    target_size = 2 * R_pixel
    ny_full, nx_full = image_untrimmed.shape

    # Compute physical size for warning
    nx_kpc = nx_full * pixel_scale_arcsec * pc_per_arcsec / 1000  
    ny_kpc = ny_full * pixel_scale_arcsec * pc_per_arcsec / 1000  

    if nx_full < target_size or ny_full < target_size:
        print(f"Image too small for requested R_kpc={R_kpc} kpc. "
              f"Image size: {nx_kpc:.2f}×{ny_kpc:.2f} kpc. "
              f"Padding to {target_size}×{target_size} pixels.")
        images_too_small.append(name)

    # Compute slice boundaries centered on RA/DEC
    x1, x2 = cx - R_pixel, cx + R_pixel
    y1, y2 = cy - R_pixel, cy + R_pixel

    # Initialize padded outputs
    image = np.full((target_size, target_size), np.nan, dtype=image_untrimmed.dtype)
    error_map = np.full((target_size, target_size), np.nan, dtype=error_map_untrimmed.dtype)
    mask = np.ones((target_size, target_size), dtype=bool)

    # Find overlap with real image bounds
    x1_img, x2_img = max(x1, 0), min(x2, nx_full)
    y1_img, y2_img = max(y1, 0), min(y2, ny_full)

    # Compute placement inside padded array
    x1_pad = x1_img - x1
    x2_pad = x1_pad + (x2_img - x1_img)
    y1_pad = y1_img - y1
    y2_pad = y1_pad + (y2_img - y1_img)

    # Copy valid region
    image[y1_pad:y2_pad, x1_pad:x2_pad] = image_untrimmed[y1_img:y2_img, x1_img:x2_img]
    error_map[y1_pad:y2_pad, x1_pad:x2_pad] = error_map_untrimmed[y1_img:y2_img, x1_img:x2_img]
    mask[y1_pad:y2_pad, x1_pad:x2_pad] = mask_untrimmed[y1_img:y2_img, x1_img:x2_img]

    target_image  = image
    target_mask   = mask
    target_emap   = error_map
    

    # 1. Identify NaNs in the image region
    nan_pixels = np.isnan(target_image)

    if nan_pixels.any():
        print(f"WARNING: Image for {name} contains NaN values within the "
            f"{target_size}×{target_size} pixel target region.")

        # 2. Replace NaNs in image with 0
        target_image[nan_pixels] = 0.0
        print(f"Replaced {np.sum(nan_pixels)} NaN pixels with 0 in image for {name}.")

        # 3. Mask: set those pixels to False (invalid)
        target_mask[nan_pixels] = False

        # 4. Determine mean error for pixels in the region where image == 0
        zero_pixels = (target_image == 0) & (~np.isnan(target_emap))
        if zero_pixels.any():
            mean_err = np.nanmean(target_emap[zero_pixels])
        else:
            # fallback if weird case — use global mean of non-NaN error
            mean_err = np.nanmean(target_emap)

        # 5. Fill errormap at the NaN-image pixels with this mean value
        target_emap[nan_pixels] = mean_err

        images_too_small.append(name)

    # Write modified arrays back
    image[:] = target_image
    mask[:] = target_mask
    error_map[:] = target_emap

    emission_pixels = np.count_nonzero(np.abs(image) > 1e-10)
    total_pixels = image.size
    emission_fraction = emission_pixels / total_pixels


    # Update WCS for cutout
    wcs_trimmed = wcs_full.deepcopy()
    wcs_trimmed.wcs.crpix[0] -= x1
    wcs_trimmed.wcs.crpix[1] -= y1

    image_nd = NDData(data=image, wcs=wcs_trimmed)


    def make_projected_region_mask(
        shape, R_kpc
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

    if flux_mask == True:
        total_flux = np.nansum(image[~mask])
        f = 1.0
        ratio = 1.0
        while ratio > 0.9:
            flux_mask_90 , flux_aperture_90, R_90  = make_projected_region_mask(image.shape, R_kpc*f*1.42)
            flux_90 = np.nansum(image[flux_mask_90 & ~mask])
            ratio = flux_90 / total_flux if total_flux > 0 else 0.0
            f -= 0.001
        
        print(f"Flux in aperature = {ratio} of total. R_90 (kpc): ", round(R_90,2))
        mask = flux_mask_90 | mask

        aperture_to_plot = flux_aperture_90
    else:
        aperture_to_plot = None


    if isolate == None or 'plot' in isolate:
        plot_moment_map(image_nd, output_dir, name, BMAJ, BMIN, R_kpc, rebin, aperture=aperture_to_plot)

    if isolate == None or any(m in isolate for m in ['gini','asym','smooth','conc','tmass','LCO','mw','aw','clump']):

        use_mc = False  # <<< set to False to skip MC and just calculate once

        if use_mc:
            # --- Monte Carlo path ---
            # Determine per-worker iteration split
            N_MC = 1000
            cpu = min(max(1, multiprocessing.cpu_count() - 1), 8)
            iters_per_worker = [N_MC // cpu] * cpu
            for i in range(N_MC % cpu):
                iters_per_worker[i] += 1

            # Shared memory for image and error map
            dtype = image.dtype
            shape = image.shape
            shm_img = shared_memory.SharedMemory(create=True, size=image.nbytes)
            shm_err = shared_memory.SharedMemory(create=True, size=error_map.nbytes)
            shm_array_img = np.ndarray(shape, dtype=dtype, buffer=shm_img.buf)
            shm_array_err = np.ndarray(shape, dtype=dtype, buffer=shm_err.buf)
            shm_array_img[:] = image[:]
            shm_array_err[:] = error_map[:]

            metric_kwargs_small = dict(
                name=name,
                co32=co32,
                pixel_area_pc2=pixel_area_pc2,
                pixel_scale_arcsec2=pixel_scale_arcsec2,
                beam_area_arcsec2=beam_area_arcsec2,
                R_21=R_21,
                R_31=R_31,
                alpha_CO=alpha_CO,
                pc_per_arcsec=pc_per_arcsec,
                pixel_scale_arcsec=pixel_scale_arcsec
            )

            # Run MC in parallel
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

            # Cleanup shared memory
            shm_img.close()
            shm_img.unlink()
            shm_err.close()
            shm_err.unlink()

            # Combine results
            def merge_global(metric_name):
                all_values = []
                for r in results:
                    if metric_name in r:
                        all_values.extend(r[metric_name] if isinstance(r[metric_name], (list, np.ndarray)) else [r[metric_name]])
                arr = np.array(all_values, dtype=float)
                if len(arr) == 0 or np.all(np.isnan(arr)):
                    print(f"All MC calculations failed for metric {metric_name} in galaxy {name}.")
                    return np.nan, np.nan
                return float(np.nanmedian(arr)), float(np.nanstd(arr))

            cont_power, cont_power_err = merge_global("cont_power_jybeam")

        else:
            # --- Single calculation path ---
            val = cont_power_jybeam(image, mask, pixel_scale_arcsec2,beam_area_arcsec2)
            cont_power = val
            cont_power_err = np.nan  # No MC → no error estimate

    return {
        "Galaxy": name,
        "Resolution (pc)": round(beam_scale_pc, 2),
        "pc_per_arcsec": round(pc_per_arcsec, 1),
        "cont_power_jy": round(cont_power, 3),
        "cont_power_jy_err": round(cont_power_err, 3)
    }

# ------------------ Parallel Directory Processing ------------------

def process_directory(outer_dir, llamatab, base_output_dir, co32, rebin=None, R_kpc=1, flux_mask=False, isolate=None):
    print(f"Processing directory: {outer_dir} (CO32={co32}, rebin={rebin}, R_kpc={R_kpc}), isolate={isolate})")
    valid_names = set(llamatab['id'])
    subdirs = [d for d in os.listdir(outer_dir)
                if os.path.isdir(os.path.join(outer_dir, d))]# and d in valid_names]

    args_list, meta_info = [], []
    for name in subdirs:
        subdir = os.path.join(outer_dir, name)
        manual_rebin = False
        if rebin is not None and not co32:
            if name in ["NGC4388","NGC6814","NGC5728"]:
                continue
            if os.path.exists(os.path.join(subdir, f"{name}_12m_cont_{rebin}pc.fits")):
                cont_im_file = os.path.join(subdir, f"{name}_12m_cont_{rebin}pc.fits")
                econt_im_file = os.path.join(subdir, f"{name}_12m_cont_{rebin}pc.fits")
                econt_im_file = "Not exists"

            else:
                if name in ["NGC4388","NGC6814","NGC5728"]:
                    continue
                print(f'{rebin} pc for {name}, using native res files or rebinning manually')
                manual_rebin = True
                cont_im_file = os.path.join(subdir, f"{name}_12m_cont.fits")
                econt_im_file = os.path.join(subdir, f"{name}_12m_cont.fits")
                econt_im_file = "Not exists"


        elif rebin is not None and co32:
            if name not in ["NGC4388","NGC6814","NGC5728"]:
                continue
            if os.path.exists(os.path.join(subdir, f"{name}_12m_cont_{rebin}pc.fits")):
                cont_im_file = os.path.join(subdir, f"{name}_12m_cont_{rebin}pc.fits")

                econt_im_file = os.path.join(subdir, f"{name}_12m_cont_{rebin}pc.fits")
                econt_im_file = "Not exists"

            else:
                if name not in ["NGC4388","NGC6814","NGC5728"]:
                    continue
                print(f'{rebin} pc for {name}, using native res files or rebinning manually')
                manual_rebin = True
                cont_im_file = os.path.join(subdir, f"{name}_12m_cont.fits")
                econt_im_file = os.path.join(subdir, f"{name}_12m_cont.fits")
                econt_im_file = "Not exists"



        elif not rebin and not co32:
            if name in ["NGC4388","NGC6814","NGC5728"]:
                continue
            cont_im_file = os.path.join(subdir, f"{name}_12m_cont.fits")

            econt_im_file = os.path.join(subdir, f"{name}_12m_cont.fits")
            econt_im_file = "Not exists"
        else:
            if name not in ["NGC4388","NGC6814","NGC5728"]:
                continue
            cont_im_file = os.path.join(subdir, f"{name}_12m_cont.fits")

            econt_im_file = os.path.join(subdir, f"{name}_12m_cont.fits")
            econt_im_file = "Not exists"


        try:
            type_val = llamatab[llamatab['id'] == name]['type'][0]
        except:
            type_val = 'aux'
        output_dir = os.path.join(base_output_dir, "inactive" if type_val=="i" else "AGN")
        group = "inactive" if type_val=="i" else "aux" if type_val=="aux" else "AGN"
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(cont_im_file):
            if not os.path.exists(econt_im_file):
                econt_im_file = np.nan
            args_list.append((cont_im_file, econt_im_file, subdir, output_dir, co32,rebin,R_kpc,flux_mask))
            meta_info.append((name, group, output_dir))
            ############### Copy moment0 files to central location ###############
            cont_im_filename = os.path.basename(cont_im_file)
            if econt_im_file is np.nan:
                econt_im_filename = "nan"
            else:
                econt_im_filename = os.path.basename(econt_im_file)
            # os.system(f'mkdir -p /data/c3040163/llama/alma/pipeline_m0/{name}/')
            # os.system(f'cp {cont_im_file} /data/c3040163/llama/alma/pipeline_m0/{name}/{cont_im_filename}')
            # os.system(f'cp {econt_im_file} /data/c3040163/llama/alma/pipeline_m0/{name}/{econt_im_filename}')
            #######################################################################
        else:
            print(f"Skipping {name}: required files not found")

    parallel_args, parallel_meta = [],[]

    for args, meta in zip(args_list, meta_info):
        parallel_args.append(args)
        parallel_meta.append(meta)

    results_raw = []

    # ctx = multiprocessing.get_context("spawn")
    # with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count(),
    #                         initializer=init_worker,
    #                         initargs=(llamatab,), mp_context=ctx) as executor:
    #     results_raw = list(executor.map(safe_process, parallel_args))

    images_too_small = []

    for args in parallel_args:
        res = process_file(args, images_too_small, isolate=isolate, manual_rebin=manual_rebin)
        results_raw.append(res)

# ------------------ CSV merge with isolate-aware updates ------------------

    # Build results + metadata lists
    results, meta_clean = [], []
    for res, meta in zip(results_raw, parallel_meta):
        if res is not None:
            results.append(res)
            meta_clean.append(meta)

    if len(results) == 0:
        print("No results to save.")
    else:
        df = pd.DataFrame(results)
        df["id"] = [mi[0] for mi in meta_clean]
        df["group"] = [mi[1] for mi in meta_clean]

        # Map isolate tokens -> output column names in results rows
        isolate_colmap = {
            # "gini":  ["Gini", "Gini_err"],
            # "asym":  ["Asymmetry", "Asymmetry_err"],
            # "smooth":["Smoothness", "Smoothness_err"],
            # "conc":  ["Concentration", "Concentration_err"],
            # "tmass": ["total_mass (M_sun)", "total_mass_err (M_sun)"],
            # "LCO":   ["L'CO (K km_s pc2)", "L'CO_err (K km_s pc2)"],
            # "mw":    ["mass_weighted_sd", "mass_weighted_sd_err"],
            # "aw":    ["area_weighted_sd", "area_weighted_sd_err"],
            # "clump": ["clumping_factor", "clumping_factor_err"],
            # "expfit":["Sigma0 (Jy/beam km/s)", "rs (pc)"],
            # "plot":  [],  # plot-only: update nothing in CSV
            "cont_power": ["cont_power_jybeam", "cont_power_jybeam_err"]
        }

        # Accept isolate as string, list or None
        isolates = isolate
        if isolates is None:
            isolates_list = None
        elif isinstance(isolates, (list, tuple, set)):
            isolates_list = [str(i) for i in isolates]
        else:
            isolates_list = [str(isolates)]

        # Build the set of columns that we should consider 'updated' by this run
        if isolates_list is None:
            cols_updated = None   # means full replace
        else:
            cols_updated = []
            for isok in isolates_list:
                cols_updated += isolate_colmap.get(isok, [])
            cols_updated = list(dict.fromkeys(cols_updated))  # uniq, preserve order

        # Helper for building outfile path
        def _outfile_path(outdir, rebin, R_kpc):
            if rebin is not None:
                if flux_mask:
                    return os.path.join(outdir, f"cont_analysis_summary_{rebin}pc_flux90_{R_kpc}kpc.csv")
                else:
                    return os.path.join(outdir, f"cont_analysis_summary_{rebin}pc_{R_kpc}kpc.csv")
            else:
                if flux_mask:
                    return os.path.join(outdir, f"cont_analysis_summary_flux90_{R_kpc}kpc.csv")
                else:
                    return os.path.join(outdir, f"cont_analysis_summary_{R_kpc}kpc.csv")

        for group in ["AGN", "inactive", "aux"]:
            group_df = df[df["group"] == group].copy()
            if group_df.empty:
                continue

            outdir = os.path.join(base_output_dir, group)
            os.makedirs(outdir, exist_ok=True)
            outfile = _outfile_path(outdir, rebin, R_kpc)

            if not os.path.exists(outfile):
                # no existing file -> just write new (full rows)
                # But if we are in plot-only mode, don't create an empty/NaN row
                if isolates_list is not None and "plot" in isolates_list:
                    print(f"Skipping CSV update for plot-only run for group {group} (no outfile existed).")
                    continue
                group_df.to_csv(outfile, index=False)
                print(f"Results for {group} saved to {outfile} (new file).")
                continue

            # Existing file -> load and update intelligently
            existing_df = pd.read_csv(outfile)

            ids_new = set(group_df["id"].astype(str).values)

            # If full recompute (cols_updated is None) -> replace existing rows for these ids
            if cols_updated is None:
                # Remove old rows with same ids and append new rows
                existing_df = existing_df[~existing_df["id"].astype(str).isin(ids_new)]
                merged = pd.concat([existing_df, group_df], ignore_index=True)
                merged.to_csv(outfile, index=False)
                print(f"Results for {group} saved to {outfile} (replaced {len(ids_new)} rows).")
                continue

            # If isolate includes 'plot', do NOT touch the CSV for those ids
            if "plot" in (isolates_list or []):
                # remove any new rows for ids that already exist (we don't want to change CSV)
                ids_existing = set(existing_df["id"].astype(str).values)
                # keep only new rows whose id is not already present (if you want to append new ids even for plot, change this)
                append_ids = [i for i in ids_new if i not in ids_existing]
                if len(append_ids) == 0:
                    print(f"Plot-only run: no CSV updates for group {group} (all ids already present).")
                    continue
                # append only the truly new ids (these will be mostly empty metrics — probably undesired)
                rows_to_append = group_df[group_df["id"].astype(str).isin(append_ids)]
                if not rows_to_append.empty:
                    existing_df = pd.concat([existing_df, rows_to_append], ignore_index=True)
                    existing_df.to_csv(outfile, index=False)
                    print(f"Plot-only run: appended {len(rows_to_append)} new rows to {outfile}.")
                else:
                    print(f"Plot-only run: nothing to append for {outfile}.")
                continue

            # General isolated metric update:
            # For each id in the new results:
            for id_val in ids_new:
                mask_new = group_df["id"].astype(str) == str(id_val)
                new_row = group_df[mask_new].iloc[0]

                # If id exists in existing file -> preserve non-updated cols
                exists_mask = existing_df["id"].astype(str) == str(id_val)
                if exists_mask.any():
                    old_row = existing_df[exists_mask].iloc[0]

                    # Ensure all columns from existing_df are present in new_row (if not, we will create them)
                    for col in existing_df.columns:
                        if col not in group_df.columns:
                            # create column in group_df filled with NaN so assignment works
                            group_df.loc[:, col] = group_df.get(col, np.nan)

                    # For each column in existing_df:
                    for col in existing_df.columns:
                        if col in ["id", "group"]:
                            continue

                        # If this column is one we intend to update (cols_updated),
                        # then keep new value *if it is finite*; otherwise keep old.
                        if col in cols_updated:
                            new_val = new_row.get(col, np.nan)
                            if pd.isna(new_val):
                                # preserve old value if new is NaN
                                group_df.loc[mask_new, col] = old_row[col]
                            else:
                                # keep new_val (already in group_df)
                                pass
                        else:
                            # column is NOT being updated now: copy old value into the new row
                            group_df.loc[mask_new, col] = old_row[col]
                else:
                    # id not in existing_df -> we will append the new row.
                    # But ensure it has all columns existing_df expects: fill missing columns with NaN
                    for col in existing_df.columns:
                        if col not in group_df.columns:
                            group_df.loc[:, col] = group_df.get(col, np.nan)

            # Now remove old rows that match ids_new, then append the updated group_df rows
            existing_df = existing_df[~existing_df["id"].astype(str).isin(ids_new)]
            merged_df = pd.concat([existing_df, group_df], ignore_index=True)

            # Reorder columns to keep original CSV column order if possible
            try:
                merged_df = merged_df[existing_df.columns]
            except Exception:
                pass

            merged_df.to_csv(outfile, index=False)
            print(f"Results for {group} saved to {outfile} (updated {len(ids_new)} rows).")
            print('images too small:', images_too_small)


# ------------------ Main ------------------

if __name__ == '__main__':
    llamatab = Table.read('/data/c3040163/llama/llama_main_properties.fits', format='fits')
    base_output_dir = '/data/c3040163/llama/alma/cont_analysis_results'
    isolate = None

    # CO(2-1)
    outer_dir_cont = '/data/c3040163/llama/alma/phangs_imaging_scripts-master/full_run_newkeys_all_arrays/reduction/imaging/'
    print("Starting CO(2-1) analysis...")
    # process_directory(outer_dir_cont, llamatab, base_output_dir, co32=False,rebin=120,R_kpc=1.5,flux_mask=True)
    # process_directory(outer_dir_cont, llamatab, base_output_dir, co32=False,rebin=120,R_kpc=1.5,flux_mask=False)

    process_directory(outer_dir_cont, llamatab, base_output_dir, co32=False,rebin=None,R_kpc=1.5)
    # process_directory(outer_dir_cont, llamatab, base_output_dir, co32=False,rebin=None,R_kpc=1.5)

    # process_directory(outer_dir_cont, llamatab, base_output_dir, co32=False,rebin=None,R_kpc=1)
    # process_directory(outer_dir_cont, llamatab, base_output_dir, co32=False,rebin=None,R_kpc=1)

    # process_directory(outer_dir_cont, llamatab, base_output_dir, co32=False,rebin=None,R_kpc=0.3,isolate=isolate)
    # process_directory(outer_dir_cont, llamatab, base_output_dir, co32=False,rebin=None,R_kpc=0.3,isolate=isolate)


    # CO(3-2)
    co32 = True
    outer_dir_co32 = '/data/c3040163/llama/alma/phangs_imaging_scripts-master/CO32_all_arrays/reduction/imaging/'
    print("Starting CO(3-2) analysis...")
    # process_directory(outer_dir_co32, llamatab, base_output_dir, co32=True,rebin=120,R_kpc=1.5,isolate=isolate,flux_mask=True)
    # process_directory(outer_dir_co32, llamatab, base_output_dir, co32=True,rebin=120,R_kpc=1.5,isolate=isolate,flux_mask=False)

    process_directory(outer_dir_co32, llamatab, base_output_dir, co32=True,rebin=None,R_kpc=1.5,isolate=isolate)
    # process_directory(outer_dir_co32, llamatab, base_output_dir, co32=True,rebin=None,R_kpc=1.5,isolate=isolate)

    # process_directory(outer_dir_co32, llamatab, base_output_dir, co32=True,rebin=None,R_kpc=1,isolate=isolate)
    # process_directory(outer_dir_co32, llamatab, base_output_dir, co32=True,rebin=None,R_kpc=1,isolate=isolate)

    # process_directory(outer_dir_co32, llamatab, base_output_dir, co32=True,rebin=None,R_kpc=0.3,isolate=isolate)
    # process_directory(outer_dir_co32, llamatab, base_output_dir, co32=True,rebin=None,R_kpc=0.3,isolate=isolate)

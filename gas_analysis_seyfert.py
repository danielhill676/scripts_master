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
from astropy.stats import sigma_clipped_stats

from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.nddata import NDData
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.ipac.ned import Ned
from astroquery.exceptions import RemoteServiceError
import requests
import time


try:
    import psutil
except ImportError:
    psutil = None

np.seterr(all='ignore')
co32 = False
LLAMATAB = None

# ------------------ Monte Carlo Helpers ------------------

def generate_random_images(image, error_map, n_iter=1000, seed=None):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=image, scale=error_map, size=(n_iter, *image.shape))

# def monte_carlo_metric(func, images, mask, **kwargs):
#     values = []
#     for img in images:
#         try:
#             val = func(img, mask, **kwargs)
#         except Exception:
#             val = np.nan
#         values.append(val)
#     values = np.array(values)
#     return np.nanmedian(values), np.nanstd(values)

def process_mc_chunk(chunk, mask, metric_kwargs, isolate=None):
    """
    Worker: compute metrics over a chunk of MC images and return lists
    of per-image metric values for each metric.
    isolate: None | str | iterable-of-str. Valid tokens:
      'gini','asym','smooth','conc','tmass','mw','aw','clump','LCO'
      If None -> compute all metrics.
    """

    # Normalize isolate into a set (or None meaning "all")
    if isolate is None:
        isolate_set = None
    elif isinstance(isolate, str):
        isolate_set = {isolate}
    else:
        isolate_set = set(isolate)

    # prepare result lists
    gini_vals = []
    asym_vals = []
    smooth_vals = []
    conc_vals = []
    tm_vals = []
    mw_vals = []
    aw_vals = []
    clump_vals = []
    LCO_vals = []   # <-- NEW LIST

    for img in chunk:

        # ----- GINI -----
        if (isolate_set is None) or ('gini' in isolate_set):
            try:
                g = gini_single(img, mask)
            except Exception:
                g = np.nan
            gini_vals.append(g)

        # ----- ASYMMETRY -----
        if (isolate_set is None) or ('asym' in isolate_set):
            try:
                a = asymmetry_single(img, mask)
            except Exception:
                a = np.nan
            asym_vals.append(a)

        # ----- SMOOTHNESS -----
        if (isolate_set is None) or ('smooth' in isolate_set):
            try:
                s = smoothness_single(
                    img, mask,
                    pc_per_arcsec=metric_kwargs["pc_per_arcsec"],
                    pixel_scale_arcsec=metric_kwargs["pixel_scale_arcsec"]
                )
            except Exception:
                s = np.nan
            smooth_vals.append(s)

        # ----- CONCENTRATION -----
        if (isolate_set is None) or ('conc' in isolate_set):
            try:
                c = concentration_single(
                    img, mask,
                    pixel_scale_arcsec=metric_kwargs["pixel_scale_arcsec"],
                    pc_per_arcsec=metric_kwargs["pc_per_arcsec"]
                )
            except Exception:
                c = np.nan
            conc_vals.append(c)

        # ----- TOTAL MASS -----
        if (isolate_set is None) or ('tmass' in isolate_set):
            try:
                tm = total_mass_single(
                    img, mask,
                    metric_kwargs["pixel_area_pc2"],
                    metric_kwargs["R_21"], metric_kwargs["R_31"],
                    metric_kwargs["alpha_CO"],
                    metric_kwargs["name"],
                    co32=metric_kwargs["co32"]
                )
            except Exception:
                tm = np.nan
            tm_vals.append(tm)

        # ----- MASS-WEIGHTED SIGMA -----
        if (isolate_set is None) or ('mw' in isolate_set):
            try:
                mw = mass_weighted_sd_single(
                    img, mask,
                    metric_kwargs["pixel_area_pc2"],
                    metric_kwargs["R_21"], metric_kwargs["R_31"],
                    metric_kwargs["alpha_CO"],
                    metric_kwargs["name"],
                    co32=metric_kwargs["co32"]
                )
            except Exception:
                mw = np.nan
            mw_vals.append(mw)

        # ----- AREA-WEIGHTED SIGMA -----
        if (isolate_set is None) or ('aw' in isolate_set):
            try:
                aw = area_weighted_sd_single(
                    img, mask,
                    metric_kwargs["pixel_area_pc2"],
                    metric_kwargs["R_21"], metric_kwargs["R_31"],
                    metric_kwargs["alpha_CO"],
                    metric_kwargs["name"],
                    co32=metric_kwargs["co32"]
                )
            except Exception:
                aw = np.nan
            aw_vals.append(aw)

        # ----- CLUMPING FACTOR -----
        if (isolate_set is None) or ('clump' in isolate_set):
            try:
                cl = clumping_factor_single(
                    img, mask,
                    metric_kwargs["pixel_area_pc2"],
                    metric_kwargs["R_21"], metric_kwargs["R_31"],
                    metric_kwargs["alpha_CO"],
                    metric_kwargs["name"],
                    co32=metric_kwargs["co32"]
                )
            except Exception:
                cl = np.nan
            clump_vals.append(cl)

        # ----- NEW: LCO -----
        if (isolate_set is None) or ('LCO' in isolate_set):
            try:
                lco = LCO(
                    img, mask,
                    metric_kwargs["pixel_area_pc2"],
                    metric_kwargs["R_21"], metric_kwargs["R_31"],
                    metric_kwargs["alpha_CO"],
                    metric_kwargs["name"],
                    co32=metric_kwargs["co32"]
                )
            except Exception:
                lco = np.nan
            LCO_vals.append(lco)

    return {
        "gini":   gini_vals,
        "asym":   asym_vals,
        "smooth": smooth_vals,
        "conc":   conc_vals,
        "tmass":  tm_vals,
        "mw":     mw_vals,
        "aw":     aw_vals,
        "clump":  clump_vals,
        "LCO":    LCO_vals,   # <-- NEW RETURN VALUE
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

def LCO(image, mask, pixel_area_pc2, R_21, R_31, alpha_CO, name, co32=False, **kwargs):
    map_LprimeCO = image * pixel_area_pc2
    return np.nansum(map_LprimeCO[~mask])

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

# ----------------- Moment map plotting ------------------

def plot_moment_map(image, outfolder, name_short, BMAJ, BMIN, R_kpc, rebin, mask, norm_type='linear'):
    # Initialise plot
    plt.rcParams.update({'font.size': 35})
    fig = plt.figure(figsize=(18 , 18),constrained_layout=True)
    ax = fig.add_subplot(111, projection=image.wcs)
    ax.margins(x=0,y=0)
    # ax.set_axis_off()

    add_scalebar(ax,1/3600,label="1''",corner='top left',color='black',borderpad=0.5,size_vertical=0.5)
    add_beam(ax,major=BMAJ,minor=BMIN,angle=0,corner='bottom right',color='black',borderpad=0.5,fill=False,linewidth=3,hatch='///')

    # fig.tight_layout()
    if np.isfinite(image.data).any():
        if norm_type == 'sqrt':
            norm = simple_norm(image.data, 'sqrt', vmin=np.nanmin(image.data), vmax=np.nanmax(image.data))
        elif norm_type == 'linear':
            norm = simple_norm(image.data, 'linear', vmin=np.nanmin(image.data), vmax=np.nanmax(image.data))
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")
    else:
        print("Moment data empty or all NaNs — skipping normalization.")
        norm = None
    #plt.title(f'{name_short}',fontsize=75)
    im=plt.imshow(image.data,origin='lower',norm=norm,cmap='RdBu_r')
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)

    if rebin is not None:
        plt.savefig(outfolder+f'/m0_plots/{R_kpc}_{rebin}_{mask}_{name_short}.png',bbox_inches='tight',pad_inches=0.0)
    else:
        plt.savefig(outfolder+f'/m0_plots/{R_kpc}_no_rebin_{mask}_{name_short}.png',bbox_inches='tight',pad_inches=0.0)
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

def process_file(args, images_too_small, isolate=None):
    mom0_file, emom0_file, subdir, output_dir, co32, rebin, PHANGS_mask, R_kpc = args
    file = mom0_file
    error_map_file = emom0_file

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
    error_map_untrimmed = fits.getdata(error_map_file, memmap=True)

    mask_untrimmed = np.isnan(image_untrimmed) | np.isnan(error_map_untrimmed)

    # cut off wierd bit of NGC 3351

    if name == 'NGC3351':
            image_untrimmed = image_untrimmed[:, :1600]
            error_map_untrimmed = error_map_untrimmed[:, :1600]
            mask_untrimmed = mask_untrimmed[:, :1600]

    header = fits.getheader(file)
    D_Mpc = llamatab[llamatab['id'] == name]['D [Mpc]'][0]
    pixel_scale_arcsec = np.abs(header.get("CDELT1", 0)) * 3600
    pc_per_arcsec = (D_Mpc * 1e6) / 206265
    BMAJ = header.get("BMAJ", 0)
    BMIN = header.get("BMIN", 0)
    beam_scale_pc = np.sqrt(np.abs(BMAJ * BMIN)) * 3600 * pc_per_arcsec
    pixel_area_pc2 = (pixel_scale_arcsec * pc_per_arcsec)**2
    R_21, R_31, alpha_CO = 0.7, 0.31, 4.35
    R_pixel = int(R_kpc * (206.265 / D_Mpc) / pixel_scale_arcsec)

    ny, nx = image_untrimmed.shape
        # --- Convert RA/DEC galaxy center → pixel coordinates ---
    max_retries = 3
    for attempt in range(max_retries):
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

    x_end = min(target_size, nx_full)
    y_end = min(target_size, ny_full)

    target_region = image_untrimmed[:y_end, :x_end]

    if np.isnan(target_region).any():
        print(f"WARNING: Image for {name} contains NaN values within the "
            f"{target_size}×{target_size} pixel target region.")
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

    # print(f"\n--- MASK DEBUG for {name} ---")
    # print("image finite pixels:", np.sum(np.isfinite(image)))
    # print("mask false (kept):", np.sum(~mask))
    # print("mask true (masked out):", np.sum(mask))
    # print("fraction kept:", np.sum(~mask) / image.size)
    # print("--- END ---\n")


    # Update WCS for cutout
    wcs_trimmed = wcs_full.deepcopy()
    wcs_trimmed.wcs.crpix[0] -= x1
    wcs_trimmed.wcs.crpix[1] -= y1

    image_nd = NDData(data=image, wcs=wcs_trimmed)

    if isolate == None or 'plot' in isolate:
        plot_moment_map(image_nd, output_dir, name, BMAJ, BMIN, R_kpc, rebin, PHANGS_mask)

    if isolate == None or any(m in isolate for m in ['gini','asym','smooth','conc','tmass','LCO','mw','aw','clump']):

        # Generate Monte-Carlo images (full set)
        N_MC = 1000
        images_mc = generate_random_images(image, error_map, n_iter=N_MC)

        # ---- PARALLEL MC PROCESSING HERE ----
        cpu = multiprocessing.cpu_count()
        chunk_size = N_MC // cpu
        chunks = [images_mc[i:i+chunk_size] for i in range(0, N_MC, chunk_size)]

        metric_kwargs = dict(
            name=name,
            co32=co32,
            pixel_area_pc2=pixel_area_pc2,
            R_21=R_21,
            R_31=R_31,
            alpha_CO=alpha_CO,
            pc_per_arcsec=pc_per_arcsec,
            pixel_scale_arcsec=pixel_scale_arcsec
        )

        with ProcessPoolExecutor(max_workers=cpu) as ex:
            results = list(ex.map(
                process_mc_chunk,
                chunks,
                [mask] * len(chunks),
                [metric_kwargs] * len(chunks),
                [isolate] * len(chunks)  # if isolate is in scope
            ))

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
        LCO, LCO_err = merge_global("LCO")
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
        LCO, LCO_err = np.nan, np.nan
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
            plt.savefig(plot_path,dpi=200)
            plt.close()

        except:
            sigma0, rs = "fit failed", "fit failed"
    else:
        sigma0, rs = np.nan, np.nan

    # print({
    #     "Galaxy": name,
    #     "Gini": round(gini, 3), "Gini_err": round(gini_err, 3),
    #     "Asymmetry": round(asym, 3), "Asymmetry_err": round(asym_err, 3),
    #     "Smoothness": round(smooth, 3), "Smoothness_err": round(smooth_err, 3),
    #     "Concentration": round(conc, 3), "Concentration_err": round(conc_err, 3),
    #     "Sigma0 (Jy/beam km/s)": sigma0,
    #     "rs (pc)": rs,
    #     "Resolution (pc)": round(beam_scale_pc, 2),
    #     "clumping_factor": round(clump, 3), "clumping_factor_err": round(clump_err, 3),
    #     "pc_per_arcsec": round(pc_per_arcsec, 1),
    #     "total_mass (M_sun)": round(total_mass, 2), "total_mass_err (M_sun)": round(total_mass_err, 2),
    #     "L'CO (K km_s pc2)": round(LCO, 3), "L'CO_err (K km_s pc2)": round(LCO_err, 3),
    #     "mass_weighted_sd": round(mw_sd, 1), "mass_weighted_sd_err": round(mw_sd_err, 1),
    #     "area_weighted_sd": round(aw_sd, 1), "area_weighted_sd_err": round(aw_sd_err, 1)
    # })

    return {
        "Galaxy": name,
        "Gini": round(gini, 3), "Gini_err": round(gini_err, 3),
        "Asymmetry": round(asym, 3), "Asymmetry_err": round(asym_err, 3),
        "Smoothness": round(smooth, 3), "Smoothness_err": round(smooth_err, 3),
        "Concentration": round(conc, 3), "Concentration_err": round(conc_err, 3),
        "Sigma0 (Jy/beam km/s)": sigma0,
        "rs (pc)": rs,
        "Resolution (pc)": round(beam_scale_pc, 2),
        "clumping_factor": round(clump, 3), "clumping_factor_err": round(clump_err, 3),
        "pc_per_arcsec": round(pc_per_arcsec, 1),
        "total_mass (M_sun)": round(total_mass, 2), "total_mass_err (M_sun)": round(total_mass_err, 2),
        "L'CO (K km_s pc2)": round(LCO, 3), "L'CO_err (K km_s pc2)": round(LCO_err, 3),
        "mass_weighted_sd": round(mw_sd, 1), "mass_weighted_sd_err": round(mw_sd_err, 1),
        "area_weighted_sd": round(aw_sd, 1), "area_weighted_sd_err": round(aw_sd_err, 1)
    }

# ------------------ Parallel Directory Processing ------------------

def process_directory(outer_dir, llamatab, base_output_dir, co32, rebin=None, mask='broad', R_kpc=1,isolate=None):
    print(f"Processing directory: {outer_dir} (CO32={co32}, rebin={rebin}, mask={mask}, R_kpc={R_kpc}), isolate={isolate})")
    valid_names = set(llamatab['id'])
    subdirs = [d for d in os.listdir(outer_dir)
                if os.path.isdir(os.path.join(outer_dir, d)) and d in valid_names]

    args_list, meta_info = [], []
    for name in subdirs:
        subdir = os.path.join(outer_dir, name)
        if rebin is not None and not co32:
            mom0_file = os.path.join(subdir, f"{name}_12m_co21_{rebin}pc_{mask}_mom0.fits")
            emom0_file = os.path.join(subdir, f"{name}_12m_co21_{rebin}pc_{mask}_emom0.fits")
        elif rebin is not None and co32:
            mom0_file = os.path.join(subdir, f"{name}_12m_co32_{rebin}pc_{mask}_mom0.fits")
            emom0_file = os.path.join(subdir, f"{name}_12m_co32_{rebin}pc_{mask}_emom0.fits")
        elif not rebin and not co32:
            mom0_file = os.path.join(subdir, f"{name}_12m_co21_{mask}_mom0.fits")
            emom0_file = os.path.join(subdir, f"{name}_12m_co21_{mask}_emom0.fits")
        else:
            mom0_file = os.path.join(subdir, f"{name}_12m_co32_{mask}_mom0.fits")
            emom0_file = os.path.join(subdir, f"{name}_12m_co32_{mask}_emom0.fits")

        type_val = llamatab[llamatab['id'] == name]['type'][0]
        output_dir = os.path.join(base_output_dir, "inactive" if type_val=="i" else "AGN")
        group = "inactive" if type_val=="i" else "AGN"
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(mom0_file) and os.path.exists(emom0_file):
            args_list.append((mom0_file, emom0_file, subdir, output_dir, co32,rebin,mask,R_kpc))
            meta_info.append((name, group, output_dir))
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
        res = process_file(args, images_too_small, isolate=isolate)
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
            "gini":  ["Gini", "Gini_err"],
            "asym":  ["Asymmetry", "Asymmetry_err"],
            "smooth":["Smoothness", "Smoothness_err"],
            "conc":  ["Concentration", "Concentration_err"],
            "tmass": ["total_mass (M_sun)", "total_mass_err (M_sun)"],
            "LCO":   ["L'CO (K km_s pc2)", "L'CO_err (K km_s pc2)"],
            "mw":    ["mass_weighted_sd", "mass_weighted_sd_err"],
            "aw":    ["area_weighted_sd", "area_weighted_sd_err"],
            "clump": ["clumping_factor", "clumping_factor_err"],
            "expfit":["Sigma0 (Jy/beam km/s)", "rs (pc)"],
            "plot":  []  # plot-only: update nothing in CSV
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
        def _outfile_path(outdir, rebin, mask, R_kpc):
            if rebin is not None:
                return os.path.join(outdir, f"gas_analysis_summary_{rebin}pc_{mask}_{R_kpc}kpc.csv")
            else:
                return os.path.join(outdir, f"gas_analysis_summary_{mask}_{R_kpc}kpc.csv")

        for group in ["AGN", "inactive"]:
            group_df = df[df["group"] == group].copy()
            if group_df.empty:
                continue

            outdir = os.path.join(base_output_dir, group)
            os.makedirs(outdir, exist_ok=True)
            outfile = _outfile_path(outdir, rebin, mask, R_kpc)

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
    base_output_dir = '/data/c3040163/llama/alma/gas_analysis_results'
    isolate = None

    # CO(2-1)
    outer_dir_co21 = '/data/c3040163/llama/alma/phangs_imaging_scripts-master/full_run_newkeys_all_arrays/reduction/derived'
    print("Starting CO(2-1) analysis...")
    process_directory(outer_dir_co21, llamatab, base_output_dir, co32=False,rebin=120,mask='broad',R_kpc=1.5)
    process_directory(outer_dir_co21, llamatab, base_output_dir, co32=False,rebin=None,mask='broad',R_kpc=1.5)
    process_directory(outer_dir_co21, llamatab, base_output_dir, co32=False,rebin=None,mask='strict',R_kpc=1.5)

    process_directory(outer_dir_co21, llamatab, base_output_dir, co32=False,rebin=None,mask='broad',R_kpc=1)
    process_directory(outer_dir_co21, llamatab, base_output_dir, co32=False,rebin=None,mask='strict',R_kpc=1)

    process_directory(outer_dir_co21, llamatab, base_output_dir, co32=False,rebin=None,mask='broad',R_kpc=0.3,isolate=isolate)
    process_directory(outer_dir_co21, llamatab, base_output_dir, co32=False,rebin=None,mask='strict',R_kpc=0.3,isolate=isolate)


    # CO(3-2)
    co32 = True
    outer_dir_co32 = '/data/c3040163/llama/alma/phangs_imaging_scripts-master/CO32_all_arrays/reduction/derived/'
    print("Starting CO(3-2) analysis...")
    process_directory(outer_dir_co32, llamatab, base_output_dir, co32=True,rebin=120,mask='broad',R_kpc=1.5,isolate=isolate)
    process_directory(outer_dir_co32, llamatab, base_output_dir, co32=True,rebin=None,mask='broad',R_kpc=1.5,isolate=isolate)
    process_directory(outer_dir_co32, llamatab, base_output_dir, co32=True,rebin=None,mask='strict',R_kpc=1.5,isolate=isolate)

    process_directory(outer_dir_co32, llamatab, base_output_dir, co32=True,rebin=None,mask='broad',R_kpc=1,isolate=isolate)
    process_directory(outer_dir_co32, llamatab, base_output_dir, co32=True,rebin=None,mask='strict',R_kpc=1,isolate=isolate)

    process_directory(outer_dir_co32, llamatab, base_output_dir, co32=True,rebin=None,mask='broad',R_kpc=0.3,isolate=isolate)
    process_directory(outer_dir_co32, llamatab, base_output_dir, co32=True,rebin=None,mask='strict',R_kpc=0.3,isolate=isolate)

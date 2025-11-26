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

def monte_carlo_metric(func, images, mask, **kwargs):
    values = []
    for img in images:
        try:
            val = func(img, mask, **kwargs)
        except Exception:
            val = np.nan
        values.append(val)
    values = np.array(values)
    return np.nanmedian(values), np.nanstd(values)

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

def process_file(args):
    mom0_file, emom0_file, subdir, output_dir, co32 = args
    file = mom0_file
    error_map_file = emom0_file
    llamatab = Table.read('/data/c3040163/llama/llama_main_properties.fits', format='fits')
    name = os.path.basename(file).split("_12m_co21_broad_mom0.fits")[0]
    if co32:
        name = os.path.basename(file).split("_12m_co32_broad_mom0.fits")[0]
    # Skip incompatible galaxies
    if not co32 and name in ['NGC4388','NGC6814','NGC5728']:
        return None
    if co32 and name not in ['NGC4388','NGC6814','NGC5728']:
        return None

    print(f"Processing {name}...")

    # Load data
    image = fits.getdata(file, memmap=True)
    error_map = fits.getdata(error_map_file, memmap=True)
    mask = np.isnan(image) | np.isnan(error_map)
    mask = mask.astype(bool)

    header = fits.getheader(file)
    D_Mpc = llamatab[llamatab['id'] == name]['D [Mpc]'][0]
    pixel_scale_arcsec = np.abs(header.get("CDELT1", 0)) * 3600
    pc_per_arcsec = (D_Mpc * 1e6) / 206265
    beam_scale_pc = np.sqrt(np.abs(header.get("BMAJ", 0) * header.get("BMIN", 0))) * 3600 * pc_per_arcsec
    pixel_area_pc2 = (pixel_scale_arcsec * pc_per_arcsec)**2
    R_21, R_31, alpha_CO = 0.7, 0.31, 4.35

    # Monte Carlo
    images_mc = generate_random_images(image, error_map, n_iter=1000)
    gini, gini_err = monte_carlo_metric(gini_single, images_mc, mask)
    asym, asym_err = monte_carlo_metric(asymmetry_single, images_mc, mask)
    smooth, smooth_err = monte_carlo_metric(smoothness_single, images_mc, mask,
                                            pc_per_arcsec=pc_per_arcsec, pixel_scale_arcsec=pixel_scale_arcsec)
    conc, conc_err = monte_carlo_metric(concentration_single, images_mc, mask,
                                        pixel_scale_arcsec=pixel_scale_arcsec, pc_per_arcsec=pc_per_arcsec)
    total_mass, total_mass_err = monte_carlo_metric(total_mass_single, images_mc, mask,
                                                    pixel_area_pc2=pixel_area_pc2,
                                                    R_21=R_21, R_31=R_31, alpha_CO=alpha_CO,
                                                    name=name, co32=co32)
    mw_sd, mw_sd_err = monte_carlo_metric(mass_weighted_sd_single, images_mc, mask,
                                            pixel_area_pc2=pixel_area_pc2,
                                            R_21=R_21, R_31=R_31, alpha_CO=alpha_CO,
                                            name=name, co32=co32)
    aw_sd, aw_sd_err = monte_carlo_metric(area_weighted_sd_single, images_mc, mask,
                                            pixel_area_pc2=pixel_area_pc2,
                                            R_21=R_21, R_31=R_31, alpha_CO=alpha_CO,
                                            name=name, co32=co32)
    clump, clump_err = monte_carlo_metric(clumping_factor_single, images_mc, mask,
                                            pixel_area_pc2=pixel_area_pc2,
                                            R_21=R_21, R_31=R_31, alpha_CO=alpha_CO,
                                            name=name, co32=co32)

    # Radial profile
    radii, profile, profile_err = radial_profile_with_errors(image, error_map, mask, nbins=10)
    valid = np.isfinite(profile) & np.isfinite(profile_err)
    radii, profile, profile_err = radii[valid], profile[valid], profile_err[valid]
    if profile.size == 0:
        sigma0, rs = "fit failed", "fit failed"
    else:
        try:
            popt, pcov = curve_fit(exp_profile, radii, profile, sigma=profile_err,
                                    absolute_sigma=True, p0=[np.max(profile), 20], maxfev=2000)
            perr = np.sqrt(np.diag(pcov))
            sigma0 = f"{popt[0]:.2e} ± {perr[0]:.2e}"
            rs_pc = popt[1] * pc_per_arcsec * pixel_scale_arcsec
            rs_pc_err = perr[1] * pc_per_arcsec * pixel_scale_arcsec
            rs = f"{rs_pc:.2f} ± {rs_pc_err:.2f}"
        except:
            sigma0, rs = "fit failed", "fit failed"

    del image, images_mc, error_map
    gc.collect()

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
        "mass_weighted_sd": round(mw_sd, 1), "mass_weighted_sd_err": round(mw_sd_err, 1),
        "area_weighted_sd": round(aw_sd, 1), "area_weighted_sd_err": round(aw_sd_err, 1)
    }

# ------------------ Parallel Directory Processing ------------------

def process_directory_parallel(outer_dir, llamatab, base_output_dir, co32):
    valid_names = set(llamatab['id'])
    subdirs = [d for d in os.listdir(outer_dir)
                if os.path.isdir(os.path.join(outer_dir, d)) and d in valid_names]

    args_list, meta_info = [], []
    for name in subdirs:
        if name != 'ESO021':
            continue  # Handle ESO021 separately
        subdir = os.path.join(outer_dir, name)
        mom0_file = os.path.join(subdir, f"{name}_12m_co21_broad_mom0.fits")
        emom0_file = os.path.join(subdir, f"{name}_12m_co21_broad_emom0.fits")
        if co32:
            mom0_file = os.path.join(subdir, f"{name}_12m_co32_broad_mom0.fits")
            emom0_file = os.path.join(subdir, f"{name}_12m_co32_broad_emom0.fits")
        type_val = llamatab[llamatab['id'] == name]['type'][0]
        output_dir = os.path.join(base_output_dir, "inactive" if type_val=="i" else "AGN")
        group = "inactive" if type_val=="i" else "AGN"
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(mom0_file) and os.path.exists(emom0_file):
            args_list.append((mom0_file, emom0_file, subdir, output_dir, co32))
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

    for args in parallel_args:
        res = process_file(args)
        results_raw.append(res)

    # Save CSV
    results, meta_clean = [], []
    for res, meta in zip(results_raw, parallel_meta):
        if res is not None:
            results.append(res)
            meta_clean.append(meta)
    df = pd.DataFrame(results)
    df["id"] = [mi[0] for mi in meta_clean]
    df["group"] = [mi[1] for mi in meta_clean]

    for group in ["AGN", "inactive"]:
        group_df = df[df["group"] == group]
        if not group_df.empty:
            outdir = os.path.join(base_output_dir, group)
            outfile = os.path.join(outdir, "gas_analysis_summary.csv")
            if os.path.exists(outfile):
                existing_df = pd.read_csv(outfile)
                group_df = pd.concat([existing_df, group_df], ignore_index=True)
            group_df.to_csv(outfile, index=False)
            print(f"Results for {group} saved to {outfile}")


# ------------------ Main ------------------

if __name__ == '__main__':
    llamatab = Table.read('/data/c3040163/llama/llama_main_properties.fits', format='fits')
    base_output_dir = '/data/c3040163/llama/alma/gas_analysis_results'

    # CO(2-1)
    outer_dir_co21 = '/data/c3040163/llama/alma/phangs_imaging_scripts-master/full_run_newkeys_all_arrays/reduction/derived'
    print("Starting CO(2-1) analysis...")
    process_directory_parallel(outer_dir_co21, llamatab, base_output_dir, co32=False)

    # CO(3-2)
    co32 = True
    outer_dir_co32 = '/data/c3040163/llama/alma/phangs_imaging_scripts-master/CO32_all_arrays/reduction/derived/'
    print("Starting CO(3-2) analysis...")
    process_directory_parallel(outer_dir_co32, llamatab, base_output_dir, co32=True)

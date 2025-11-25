from curses import meta
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
np.seterr(all='ignore')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import multiprocessing
import gc

import traceback, time
try:
    import psutil
except Exception:
    psutil = None

co32 = False

# ---------- Monte Carlo helpers ----------

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

LLAMATAB = None

def init_worker(table):
    global LLAMATAB
    LLAMATAB = table

def safe_process(args):
    """Run process_file but never let the worker die silently. Return either dict or a tuple marker with traceback."""
    try:
        # call actual processing function
        return process_file(args)
    except BaseException as e:
        tb = traceback.format_exc()
        # capture memory info (optional)
        try:
            mem = psutil.Process().memory_info().rss
        except Exception:
            mem = None
        # return an error marker (serializable)
        return ("__ERROR__", args, str(e), tb, mem, time.time())



# ---------- Metrics ----------

def gini_single(image, mask):
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

def asymmetry_single(image, mask):
    image_rot = np.rot90(image, 2)
    mask_rot = np.rot90(mask, 2)
    valid_mask = (~mask) & (~mask_rot) & np.isfinite(image) & np.isfinite(image_rot)
    if np.sum(valid_mask) == 0:
        return np.nan
    diff = np.abs(image[valid_mask] - image_rot[valid_mask])
    total = np.abs(image[valid_mask])
    return np.sum(diff) / np.sum(total) if np.sum(total) > 0 else np.nan

def smoothness_single(image, mask, pc_per_arcsec, pixel_scale_arcsec):
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

def concentration_single(image, mask, pixel_scale_arcsec, pc_per_arcsec):
    y, x = np.indices(image.shape)
    center = (x.max() / 2, y.max() / 2)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r_pc = r * pixel_scale_arcsec * pc_per_arcsec
    valid = (~mask) & np.isfinite(image)
    flux_50 = np.sum(image[(r_pc < 50) & valid]) / 50**2
    flux_200 = np.sum(image[(r_pc < 200) & valid]) / 200**2
    if flux_200 <= 0 or flux_50 <= 0:
        return np.nan
    return np.log10(flux_50 / flux_200)

def total_mass_single(image, mask, pixel_area_pc2, R_21, R_31, alpha_CO, name, co32):
    map_LprimeCO = image * pixel_area_pc2
    if co32:
        map_LprimeCO10 = map_LprimeCO / R_31
    else:
        map_LprimeCO10 = map_LprimeCO / R_21
    map_MH2 = alpha_CO * map_LprimeCO10
    return np.nansum(map_MH2[~mask])

def mass_weighted_sd_single(image, mask, pixel_area_pc2, R_21, R_31, alpha_CO, name, co32):
    map_LprimeCO = image * pixel_area_pc2
    if co32:
        map_LprimeCO10 = map_LprimeCO / R_31
    else:
        map_LprimeCO10 = map_LprimeCO / R_21
    map_MH2 = alpha_CO * map_LprimeCO10
    Sigma = map_MH2 / pixel_area_pc2
    Sigma = Sigma[~mask]
    if Sigma.size == 0:
        return np.nan
    numerator = np.sum(Sigma**2 * pixel_area_pc2)
    denominator = np.sum(Sigma * pixel_area_pc2)
    return numerator / denominator if denominator > 0 else np.nan

def area_weighted_sd_single(image, mask, pixel_area_pc2, R_21, R_31, alpha_CO, name, co32):
    map_LprimeCO = image * pixel_area_pc2
    if co32:
        map_LprimeCO10 = map_LprimeCO / R_31
    else:
        map_LprimeCO10 = map_LprimeCO / R_21
    map_MH2 = alpha_CO * map_LprimeCO10
    Sigma = map_MH2 / pixel_area_pc2
    Sigma = Sigma[~mask]
    if Sigma.size == 0:
        return np.nan
    total_area = Sigma.size * pixel_area_pc2
    return np.sum(Sigma * pixel_area_pc2) / total_area if total_area > 0 else np.nan

def clumping_factor_single(image, mask, pixel_area_pc2, R_21, R_31, alpha_CO, name):
    mw = mass_weighted_sd_single(image, mask, pixel_area_pc2, R_21, R_31, alpha_CO, name)
    aw = area_weighted_sd_single(image, mask, pixel_area_pc2, R_21, R_31, alpha_CO, name)
    if aw and aw > 0:
        return mw / aw
    return np.nan

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


# ---------- Process file ---------- #

def process_file(args):
    mom0_file, emom0_file, subdir, output_dir, co32 = args
    file = mom0_file
    error_map = emom0_file
    input_dir = subdir
    global LLAMATAB
    llamatab = LLAMATAB                   # get the table from global
    try:
        name = os.path.basename(file).split("_12m_co21_broad_mom0.fits")[0]
        if co32:
            name = os.path.basename(file).split("_12m_co32_broad_mom0.fits")[0]
        if not co32 and name in ['NGC4388','NGC6814','NGC5728']:
            return None
        elif co32 and name not in ['NGC4388','NGC6814','NGC5728']:
            return None
        print(f"Processing {name}...")
        image = fits.getdata(file, memmap=True)
        header = fits.getheader(file)
        error_map = fits.getdata(emom0_file, memmap=True)

        mask = np.isnan(image) | np.isnan(error_map)
        mask = np.array(mask, dtype=bool)

        # sn_threshold = 1.0
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     sn_map = np.where(error_map > 0, image / error_map, 0)
        # mask |= (sn_map < sn_threshold)

        D_Mpc = llamatab[llamatab['id'] == name]['D [Mpc]'][0]
        pixel_scale_deg = np.abs(header.get("CDELT1", 0))
        pixel_scale_arcsec = pixel_scale_deg * 3600
        pc_per_arcsec = (D_Mpc * 1e6) / 206265
        beam_scale_deg = np.sqrt(np.abs(header.get("BMAJ", 0) * header.get("BMIN", 0)))
        beam_scale_arcsec = beam_scale_deg * 3600
        beam_scale_pc = beam_scale_arcsec * pc_per_arcsec

        pixel_area_pc2 = (pixel_scale_arcsec * pc_per_arcsec)**2

        R_21 = 0.7
        R_31 = 0.31
        alpha_CO = 4.35

        # Monte Carlo images
        n_iter = 1000
        images_mc = generate_random_images(image, error_map, n_iter=n_iter)

        # --- Compute monte carlo metrics ---
        gini, gini_err = monte_carlo_metric(gini_single, images_mc, mask)
        asym, asym_err = monte_carlo_metric(asymmetry_single, images_mc, mask)
        smooth, smooth_err = monte_carlo_metric(smoothness_single, images_mc, mask,
                                                pc_per_arcsec=pc_per_arcsec, pixel_scale_arcsec=pixel_scale_arcsec)
        conc, conc_err = monte_carlo_metric(concentration_single, images_mc, mask,
                                            pixel_scale_arcsec=pixel_scale_arcsec, pc_per_arcsec=pc_per_arcsec)
        total_mass, total_mass_err = monte_carlo_metric(total_mass_single, images_mc, mask,
                                                        pixel_area_pc2=pixel_area_pc2,
                                                        R_21=R_21, R_31=R_31, alpha_CO=alpha_CO, name=name, co32=co32)
        mw_sd, mw_sd_err = monte_carlo_metric(mass_weighted_sd_single, images_mc, mask,
                                              pixel_area_pc2=pixel_area_pc2,
                                              R_21=R_21, R_31=R_31, alpha_CO=alpha_CO, name=name, co32=co32)
        aw_sd, aw_sd_err = monte_carlo_metric(area_weighted_sd_single, images_mc, mask,
                                              pixel_area_pc2=pixel_area_pc2,
                                              R_21=R_21, R_31=R_31, alpha_CO=alpha_CO, name=name, co32=co32)
        clump, clump_err = monte_carlo_metric(clumping_factor_single, images_mc, mask,
                                              pixel_area_pc2=pixel_area_pc2,
                                              R_21=R_21, R_31=R_31, alpha_CO=alpha_CO, name=name, co32=co32)

        # Radial profile fit
        radii, profile, profile_err = radial_profile_with_errors(image, error_map, mask, nbins=10)
        valid = np.isfinite(profile) & np.isfinite(profile_err)
        radii, profile, profile_err = radii[valid], profile[valid], profile_err[valid]
        if profile.size == 0:
            print(f"Error processing {file}: no valid data for radial profile fitting")
            sigma0 = "fit failed"
            rs = "fit failed"
        else:
            try:
                popt, pcov = curve_fit(exp_profile, radii, profile, sigma=profile_err,
                                    absolute_sigma=True, p0=[np.max(profile), 20], maxfev=2000)
                perr = np.sqrt(np.diag(pcov))
                sigma0 = f"{popt[0]:.2e} ± {perr[0]:.2e}"
                rs_arcsec = popt[1] * pixel_scale_arcsec
                rs_arcsec_err = perr[1] * pixel_scale_arcsec
                rs_pc = rs_arcsec * pc_per_arcsec
                rs_pc_err = rs_arcsec_err * pc_per_arcsec
                rs = f"{rs_pc:.2f} ± {rs_pc_err:.2f}"
            except Exception as e:
                print(f"Error processing {file}: {e}")
                sigma0 = "fit failed"
                rs = "fit failed"

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

    except Exception as e:
        print(f"Fatal error processing {file}: {e}")
        return None


# ---------- Parallel directory processing ----------


def process_directory_parallel(outer_dir, llamatab, base_output_dir, co32):
    try:
        valid_names = set(llamatab['id'])
        subdirs = [d for d in os.listdir(outer_dir)
                if os.path.isdir(os.path.join(outer_dir, d)) and d in valid_names]

        # Build args_list and meta_info as usual
        args_list, meta_info = [], []

        for name in subdirs:
            subdir = os.path.join(outer_dir, name)
            mom0_file = os.path.join(subdir, f"{name}_12m_co21_broad_mom0.fits")
            emom0_file = os.path.join(subdir, f"{name}_12m_co21_broad_emom0.fits")
            if co32:
                mom0_file = os.path.join(subdir, f"{name}_12m_co32_broad_mom0.fits")
                emom0_file = os.path.join(subdir, f"{name}_12m_co32_broad_emom0.fits")
            type_val = llamatab[llamatab['id'] == name]['type'][0]
            if type_val == "i":
                output_dir = os.path.join(base_output_dir, "inactive")
                group = "inactive"
            else:
                output_dir = os.path.join(base_output_dir, "AGN")
                group = "AGN"
            os.makedirs(output_dir, exist_ok=True)
            if os.path.exists(mom0_file) and os.path.exists(emom0_file):
                args_list.append((mom0_file, emom0_file, subdir, output_dir, co32))
                meta_info.append((name, group, output_dir))
            else:
                print(f"Skipping {name}: required files not found")

        # Separate ESO021 from the parallel batch
        parallel_args = []
        parallel_meta = []
        eso021_args = None
        eso021_meta = None

        for args, meta in zip(args_list, meta_info):
            if meta[0] == "ESO021":
                eso021_args = args
                eso021_meta = meta
            else:
                parallel_args.append(args)
                parallel_meta.append(meta)

        # --- Run parallel for everything except ESO021 ---
        results_raw = []
        if parallel_args:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count(),
                                    initializer=init_worker,
                                    initargs=(llamatab,), mp_context=ctx) as executor:
                results_raw = list(executor.map(safe_process, parallel_args))

        # --- Run ESO021 serially ---
        if eso021_args is not None:
            print("Running ESO021 serially...")
            r = process_file(eso021_args)
            if r is not None:
                results_raw.append(r)
                parallel_meta.append(eso021_meta)  # keep metadata for CSV

        # --- Combine results and write CSV ---
        results = []
        meta_info_clean = []

        for res, meta in zip(results_raw, parallel_meta):
            if res is not None:
                results.append(res)
                meta_info_clean.append(meta)

        df = pd.DataFrame(results)
        df["id"] = [mi[0] for mi in meta_info_clean]
        df["group"] = [mi[1] for mi in meta_info_clean]

        for group in ["AGN", "inactive"]:
            group_df = df[df["group"] == group]
            if not group_df.empty:
                outdir = os.path.join(base_output_dir, group)
                outfile = os.path.join(outdir, "gas_analysis_summary.csv")
                if co32 and os.path.exists(outfile):
                    existing_df = pd.read_csv(outfile)
                    group_df = pd.concat([existing_df, group_df], ignore_index=True)
                group_df.to_csv(outfile, index=False)
                print(f"Results for {group} saved to {outfile}")
    except Exception as e:
        print(f"Fatal error in processing directory {name}: {e}")


if __name__ == '__main__':
    llamatab = Table.read('/data/c3040163/llama/llama_main_properties.fits', format='fits')
    outer_dir = '/data/c3040163/llama/alma/phangs_imaging_scripts-master/full_run_newkeys_all_arrays/reduction/derived'
    base_output_dir = '/data/c3040163/llama/alma/gas_analysis_results'
    try:
        process_directory_parallel(outer_dir, llamatab, base_output_dir,co32)
    except Exception as e:
        print('multiprocessing error is ignored:', e)
    co32 = True
    print("Starting CO(3-2) analysis...")
    outer_dir = '/data/c3040163/llama/alma/phangs_imaging_scripts-master/CO32_all_arrays/reduction/derived/'
    process_directory_parallel(outer_dir, llamatab, base_output_dir,co32)
    

    # name, group, out = 'ESO021', 'AGN', '/data/c3040163/llama/alma/gas_analysis_results/AGN'
    # mom0 = os.path.join(outer_dir, name, f"{name}_12m_co21_broad_mom0.fits")
    # emom0 = os.path.join(outer_dir, name, f"{name}_12m_co21_broad_emom0.fits")
    # args = (mom0, os.path.join(outer_dir, name), out, llamatab)
    # print(process_file(args))

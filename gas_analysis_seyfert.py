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

def process_mc_chunk(chunk, mask, metric_kwargs):
    """Worker: compute metrics over a chunk of MC images."""
    gini_vals = []
    asym_vals = []
    smooth_vals = []
    conc_vals = []
    tm_vals = []
    mw_vals = []
    aw_vals = []
    clump_vals = []

    for img in chunk:
        try:
            gini_vals.append(gini_single(img, mask))
            asym_vals.append(asymmetry_single(img, mask))
            smooth_vals.append(smoothness_single(img, mask,
                                                 pc_per_arcsec=metric_kwargs["pc_per_arcsec"],
                                                 pixel_scale_arcsec=metric_kwargs["pixel_scale_arcsec"]))
            conc_vals.append(concentration_single(img, mask,
                                                  pixel_scale_arcsec=metric_kwargs["pixel_scale_arcsec"],
                                                  pc_per_arcsec=metric_kwargs["pc_per_arcsec"]))
            tm_vals.append(total_mass_single(img, mask,
                                             metric_kwargs["pixel_area_pc2"],
                                             metric_kwargs["R_21"], metric_kwargs["R_31"],
                                             metric_kwargs["alpha_CO"],
                                             metric_kwargs["name"],
                                             co32=metric_kwargs["co32"]))
            mw_vals.append(mass_weighted_sd_single(img, mask,
                                                   metric_kwargs["pixel_area_pc2"],
                                                   metric_kwargs["R_21"], metric_kwargs["R_31"],
                                                   metric_kwargs["alpha_CO"],
                                                   metric_kwargs["name"],
                                                   co32=metric_kwargs["co32"]))
            aw_vals.append(area_weighted_sd_single(img, mask,
                                                   metric_kwargs["pixel_area_pc2"],
                                                   metric_kwargs["R_21"], metric_kwargs["R_31"],
                                                   metric_kwargs["alpha_CO"],
                                                   metric_kwargs["name"],
                                                   co32=metric_kwargs["co32"]))
            clump_vals.append(clumping_factor_single(img, mask,
                                                     metric_kwargs["pixel_area_pc2"],
                                                     metric_kwargs["R_21"], metric_kwargs["R_31"],
                                                     metric_kwargs["alpha_CO"],
                                                     metric_kwargs["name"],
                                                     co32=metric_kwargs["co32"]))
        except:
            pass

    import numpy as np

    # Convert to medians + std for this chunk
    return {
        "gini":   (np.nanmedian(gini_vals), np.nanstd(gini_vals)),
        "asym":   (np.nanmedian(asym_vals), np.nanstd(asym_vals)),
        "smooth": (np.nanmedian(smooth_vals), np.nanstd(smooth_vals)),
        "conc":   (np.nanmedian(conc_vals), np.nanstd(conc_vals)),
        "tmass":  (np.nanmedian(tm_vals), np.nanstd(tm_vals)),
        "mw":     (np.nanmedian(mw_vals), np.nanstd(mw_vals)),
        "aw":     (np.nanmedian(aw_vals), np.nanstd(aw_vals)),
        "clump":  (np.nanmedian(clump_vals), np.nanstd(clump_vals)),
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
    mom0_file, emom0_file, subdir, output_dir, co32, rebin, mask = args
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
    image = fits.getdata(file, memmap=True)
    error_map = fits.getdata(error_map_file, memmap=True)

    mask = np.isnan(image) | np.isnan(error_map)

    header = fits.getheader(file)
    D_Mpc = llamatab[llamatab['id'] == name]['D [Mpc]'][0]
    pixel_scale_arcsec = np.abs(header.get("CDELT1", 0)) * 3600
    pc_per_arcsec = (D_Mpc * 1e6) / 206265
    beam_scale_pc = np.sqrt(np.abs(header.get("BMAJ", 0) * header.get("BMIN", 0))) * 3600 * pc_per_arcsec
    pixel_area_pc2 = (pixel_scale_arcsec * pc_per_arcsec)**2
    R_21, R_31, alpha_CO = 0.7, 0.31, 4.35

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
        results = list(ex.map(process_mc_chunk,
                              chunks,
                              [mask]*len(chunks),
                              [metric_kwargs]*len(chunks)))

    # Aggregate results across chunks
    def merge(metric):
        meds = [r[metric][0] for r in results]
        stds = [r[metric][1] for r in results]
        return float(np.nanmean(meds)), float(np.nanmean(stds))

    gini, gini_err = merge("gini")
    asym, asym_err = merge("asym")
    smooth, smooth_err = merge("smooth")
    conc, conc_err = merge("conc")
    total_mass, total_mass_err = merge("tmass")
    mw_sd, mw_sd_err = merge("mw")
    aw_sd, aw_sd_err = merge("aw")
    clump, clump_err = merge("clump")

    # Radial profile unchanged
    radii, profile, profile_err = radial_profile_with_errors(image, error_map, mask, nbins=10)
    valid = np.isfinite(profile) & np.isfinite(profile_err)
    radii, profile, profile_err = radii[valid], profile[valid], profile_err[valid]

    try:
        popt, pcov = curve_fit(exp_profile, radii, profile, sigma=profile_err,
                               absolute_sigma=True, p0=[np.max(profile), 20], maxfev=2000)
        perr = np.sqrt(np.diag(pcov))
        sigma0 = f"{popt[0]:.2e} ± {perr[0]:.2e}"
        rs_pc = popt[1] * pc_per_arcsec * pixel_scale_arcsec
        rs_pc_err = perr[1] * pc_per_arcsec * pixel_scale_arcsec
        rs = f"{rs_pc:.2f} ± {rs_pc_err:.2f}"

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
        plot_path = os.path.join(output_dir, f"{name}_{mask}_{rebin}_expfit.png")
        plt.savefig(plot_path)
        plt.close()

    except:
        sigma0, rs = "fit failed", "fit failed"

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

def process_directory(outer_dir, llamatab, base_output_dir, co32, rebin=None, mask='broad'):
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
            args_list.append((mom0_file, emom0_file, subdir, output_dir, co32,rebin,mask))
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
            if rebin is not None:
                outfile = os.path.join(outdir, f"gas_analysis_summary_{rebin}pc_{mask}.csv")
            else:
                outfile = os.path.join(outdir, f"gas_analysis_summary_{mask}.csv")
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
    process_directory(outer_dir_co21, llamatab, base_output_dir, co32=False,rebin=120,mask='broad')

    # CO(3-2)
    co32 = True
    outer_dir_co32 = '/data/c3040163/llama/alma/phangs_imaging_scripts-master/CO32_all_arrays/reduction/derived/'
    print("Starting CO(3-2) analysis...")
    process_directory(outer_dir_co32, llamatab, base_output_dir, co32=True,rebin=120,mask='broad')

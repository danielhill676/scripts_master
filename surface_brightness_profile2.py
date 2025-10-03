import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend suitable for scripts
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy.ndimage import convolve
from glob import glob
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import gc
from scipy.ndimage import gaussian_filter
from astropy.convolution import convolve, Box2DKernel
from scipy.ndimage import uniform_filter

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

def gini_coefficient(image, mask):
    valid_data = image[~mask & np.isfinite(image)].flatten()
    
    # Remove negative values (Gini requires non-negative domain)
    valid_data = valid_data[valid_data >= 0]
    
    if len(valid_data) == 0:
        return np.nan

    sorted_vals = np.sort(valid_data)
    n = len(sorted_vals)
    
    # Prevent division by zero
    total = np.sum(sorted_vals)
    if total == 0:
        return 0.0

    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_vals) / (n * total)) - (n + 1) / n
    return gini

def process_file(args):
    file, input_dir, output_dir, llamatab = args
    try:
        name = os.path.basename(file).split("_moment_0")[0]
        print(f"Processing {name}...")
        image = fits.getdata(file,memmap=True)
        header = fits.getheader(file)
        error_map = fits.getdata(os.path.join(input_dir, f"{name}.sigma.fits"),memmap=True)
        mask = fits.getdata(os.path.join(input_dir, f"{name}.bad_pixels.fits"),memmap=True).astype(bool)

        sn_threshold = 1.0
        with np.errstate(divide='ignore', invalid='ignore'):  # avoid warnings from /0
            sn_map = np.where(error_map > 0, image / error_map, 0)

        original_masked_pixels = np.count_nonzero(mask)
        mask |= (sn_map < sn_threshold)  # extend mask where S/N < threshold

        total_pixels = image.size
        masked_pixels = np.count_nonzero(mask)
        added_masked_pixels = masked_pixels - original_masked_pixels

        print(f"Masking applied: {masked_pixels}/{total_pixels} pixels masked "
              f"({masked_pixels/total_pixels:.2%}) ")
              #f" | Added due to S/N: {added_masked_pixels}")

        D_Mpc = llamatab[llamatab['id'] == name]['D [Mpc]'][0]
        z = llamatab[llamatab['id'] == name]['redshift'][0]

        pixel_scale_deg = np.abs(header.get("CDELT1", 0))
        pixel_scale_arcsec = pixel_scale_deg * 3600
        pc_per_arcsec = (D_Mpc * 1e6) / 206265

        beam_scale_deg = np.sqrt(np.abs(header.get("BMAJ", 0) * header.get("BMIN", 0)))
        beam_scale_arcsec = beam_scale_deg * 3600
        beam_scale_pc = beam_scale_arcsec * pc_per_arcsec


        ##### converting map into physical units #####

        beam = np.pi * header['BMAJ'] * header['BMIN'] / (4 * np.log(2))  # in deg^2
        beam_arcsec2 = beam * 3600**2  # in arcsec^
        pixel_area_arcsec2 = pixel_scale_arcsec**2  # in arcsec^2
        pixel_area_pc2 = (pixel_scale_arcsec * pc_per_arcsec)**2  # in pc^2

        map_jy_kms = image * pixel_area_arcsec2/beam_arcsec2# in Jy km/s

        vobs = 230.538 / (1 + z)  # in GHz
        map_LprimeCO = 3.25e7 * map_jy_kms * vobs**(-2) * (1+z)**(-3) * D_Mpc**2  # in K km/s pc^2  

        R_21 = 0.7 # CO(2-1)/CO(1-0) ratio
        R_31 = 0.31 # CO(3-2)/CO(1-0) ratio

        if name == 'NGC4388':
            map_LprimeCO10 = map_LprimeCO / R_31
        else:
            map_LprimeCO10 = map_LprimeCO / R_21  # in K km/s pc^2

        alpha_CO = 4.35  # in M_sun/(K km/s pc^2)
        map_MH2 = alpha_CO * map_LprimeCO10  # in M_sun


        # Asymmetry
        print("Calculating asymmetry...")
        image_rot = np.rot90(image, 2)
        mask_rot = np.rot90(mask, 2)
        valid_mask = (~mask) & (~mask_rot) & np.isfinite(image) & np.isfinite(image_rot)
        diff = np.abs(image[valid_mask] - image_rot[valid_mask])
        total = np.abs(image[valid_mask])
        asymmetry = np.sum(diff) / np.sum(total)

        # Smoothness
        print("Calculating smoothness...")
        smoothing_sigma_pc = 500
        smoothing_sigma = (smoothing_sigma_pc / pc_per_arcsec) / pixel_scale_arcsec  # in pixels

        # Make sure smoothing window size is integer
        size = max(1, int(round(smoothing_sigma)))

        # Replace NaNs with 0 temporarily to avoid propagation
        image_filled = np.nan_to_num(image, nan=0.0)

        # Create mask for valid pixels (1 where valid, 0 where NaN or masked)
        valid_mask = (~mask) & np.isfinite(image)

        # Smooth the image and the mask separately with the same boxcar filter
        smooth_image = uniform_filter(image_filled, size=size, mode='reflect')
        smooth_mask = uniform_filter(valid_mask.astype(float), size=size, mode='reflect')

        # Avoid division by zero
        with np.errstate(invalid='ignore', divide='ignore'):
            image_smooth = smooth_image / smooth_mask
        image_smooth[smooth_mask == 0] = np.nan  # Mark pixels with no data in window as NaN

        # Now calculate smoothness as before:
        valid_smooth = (~mask) & np.isfinite(image) & np.isfinite(image_smooth)

        if np.sum(valid_smooth) > 0:
            diff_smooth = np.abs(image[valid_smooth] - image_smooth[valid_smooth])
            total_flux = np.abs(image[valid_smooth])
            smoothness = np.sum(diff_smooth) / np.sum(total_flux)
        else:
            smoothness = np.nan

        gc.collect()

        # Alternative clumpiness factor from Leroy et al. 2013

        map_Sigma = map_MH2 / pixel_area_pc2  # M_sun / pc^2
        Sigma = map_Sigma[~mask]

        map_Sigma_masked = np.where(mask, np.nan, map_Sigma)  # keep 2D shape
        hdu = fits.PrimaryHDU(map_Sigma_masked)
        hdu.writeto(f'{name}_surface_density.fits', overwrite=True)

        if Sigma.size == 0:
            mass_weighted_sd = np.nan
            area_weighted_sd = np.nan
            total_mass = 0.0
        else:
            # per-pixel mass in M_sun
            mass_per_pixel = Sigma * pixel_area_pc2   # M_sun

            # mass-weighted mean: (sum Σ^2 A) / (sum Σ A)
            numerator = np.sum(Sigma**2 * pixel_area_pc2)
            denominator = np.sum(Sigma * pixel_area_pc2)
            mass_weighted_sd = numerator / denominator

            print(f"Mass-weighted SD: {mass_weighted_sd}")

            # area-weighted mean: (sum Σ A) / (sum A) = sum Σ / N  if A same
            total_area_pc2 = Sigma.size * pixel_area_pc2
            area_weighted_sd = np.sum(Sigma * pixel_area_pc2) / total_area_pc2

            print(f"Area-weighted SD: {area_weighted_sd}")

            total_mass = np.sum(mass_per_pixel)
            clumping_factor = mass_weighted_sd / area_weighted_sd if area_weighted_sd != 0 else np.nan

        # Gini coefficient

        gini = gini_coefficient(image, mask)

        # Binned Radial profile
        print("Calculating radial profile...")
        radii, profile, profile_err = radial_profile_with_errors(image, error_map, mask, nbins=10)
        valid = np.isfinite(profile) & np.isfinite(profile_err)
        radii, profile, profile_err = radii[valid], profile[valid], profile_err[valid]

        # Unbinned Radial profile and concentration index
        y, x = np.indices(image.shape)
        center = (x.max() / 2, y.max() / 2)
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)

        r_arcsec = r * pixel_scale_arcsec
        r_pc = r_arcsec * pc_per_arcsec

        valid_conc = (~mask) & np.isfinite(image)

        flux_50 = np.sum(image[(r_pc < 50) & valid_conc]) / 50**2
        print(f"Flux within 50 pc: {flux_50}")
        flux_200 = np.sum(image[(r_pc < 200) & valid_conc]) / 200**2
        print(f"Flux within 200 pc: {flux_200}")

        if flux_200 > 0:
            concentration_index = np.log10(flux_50 / flux_200)
        else:
            concentration_index = np.nan

        print('Concentration index =', concentration_index)

        try:
            popt, pcov = curve_fit(exp_profile, radii, profile, sigma=profile_err, absolute_sigma=True, p0=[np.max(profile), 20], maxfev=2000)
            perr = np.sqrt(np.diag(pcov))
            sigma0 = f"{popt[0]:.2e} ± {perr[0]:.2e}"
            rs_arcsec = popt[1] * pixel_scale_arcsec
            rs_arcsec_err = perr[1] * pixel_scale_arcsec
            rs_pc = rs_arcsec * pc_per_arcsec
            rs_pc_err = rs_arcsec_err * pc_per_arcsec
            rs = f"{rs_pc:.2f} ± {rs_pc_err:.2f}"
        except Exception:
            sigma0 = "fit failed"
            rs = "fit failed"

        # Plot
        print("Plotting results...")
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
        plot_path = os.path.join(output_dir, f"{name}_expfit.png")
        plt.savefig(plot_path)
        plt.close()

        del image, image_smooth, image_rot, mask_rot, error_map
        gc.collect()

        return {
            "Galaxy": name,
            "Gini": round(gini, 3),
            "Asymmetry": round(asymmetry, 3),
            "Smoothness": round(smoothness, 3),
            "Concentration": round(concentration_index, 3),
            "Sigma0 (Jy/beam km/s)": sigma0,
            "rs (pc)": rs,
            'Resolution (pc)': round(beam_scale_pc, 2),
            'clumping_factor': round(clumping_factor, 3),
            'pc_per_arcsec': round(pc_per_arcsec, 1),
            'total_mass (M_sun)': round(total_mass, 2),
            'mass_weighted_sd': round(mass_weighted_sd, 1),
            'area_weighted_sd': round(area_weighted_sd, 1)
        }

    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None

def process_directory_parallel(input_dir, output_dir, llamatab):
    os.makedirs(output_dir, exist_ok=True)
    files = glob(os.path.join(input_dir, "*_moment_0.fits"))
    args_list = [(f, input_dir, output_dir, llamatab) for f in files]

    with ProcessPoolExecutor(max_workers=1) as executor:
        results = list(executor.map(process_file, args_list))

    results = [res for res in results if res is not None]
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "gas_analysis_summary.csv"), index=False)
    print(f"Results saved to {os.path.join(output_dir, 'gas_analysis_summary.csv')}")

if __name__ == '__main__':
    from astropy.table import Table

    # Load galaxy properties
    llamatab = Table.read('/Users/administrator/Astro/LLAMA/llama_main_properties.fits', format='fits')

    # file = "/Users/administrator/Astro/LLAMA/ALMA/AGN_images/NGC4388_moment_0.fits"
    # result = process_file((file, "/Users/administrator/Astro/LLAMA/ALMA/AGN_images", "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/AGN", llamatab))
    # print(result)

    # Run for AGN
    process_directory_parallel(
        input_dir="/Users/administrator/Astro/LLAMA/ALMA/AGN_images",
        output_dir="/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/AGN",
        llamatab=llamatab
    )

    # Run for inactive
    process_directory_parallel(
        input_dir="/Users/administrator/Astro/LLAMA/ALMA/inactive_images",
        output_dir="/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/inactive",
        llamatab=llamatab
    )

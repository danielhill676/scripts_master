from astropy.convolution import convolve_fft
from radio_beam import Beam
from radio_beam.utils import BeamError
from scipy.ndimage import gaussian_filter
from astropy.io import fits
import numpy as np
from astropy.table import Table
import astropy.units as u
from astropy.convolution import convolve, Gaussian2DKernel
import time


############################################################################################################
fits_file = '/Users/administrator/Astro/LLAMA/ALMA/pipeline_m0/NGC4254/NGC4254_12m_co21_120pc_strict_mom0.fits'
base_name = 'NGC4254'
target_res = 200
pixel_sigma = 11.39
############################################################################################################

image = fits.getdata(fits_file)
header = fits.getheader(fits_file)
llamatab = Table.read('/Users/administrator/Astro/LLAMA/llama_main_properties.fits', format='fits')
pixel_scale_arcsec = np.abs(header.get("CDELT1", 0)) * 3600
pixel_area_arcsec2 = pixel_scale_arcsec**2
BMAJ = header.get("BMAJ", np.nan)
BMIN = header.get("BMIN", np.nan)
beam_arcsec = np.sqrt(np.abs(BMAJ * BMIN)) * 3600
beam_area_arcsec2 = (np.pi/(4*np.log(2)))*(BMAJ*3600)*(BMIN*3600)
D_Mpc = llamatab[llamatab['id'] == base_name]['D [Mpc]'][0]
pc_per_arcsec = (D_Mpc * 1e6) / 206265
beam_scale_pc = beam_arcsec * pc_per_arcsec
beam_area_pc2 = beam_area_arcsec2 * pc_per_arcsec**2
pixel_scale_pc = pixel_scale_arcsec * pc_per_arcsec
pixel_area_pc2 = pixel_scale_pc**2
native_res = float(beam_scale_pc)

image_copy = image.copy()
# ---------- NaN handling ----------
nan_pixels = np.isnan(image_copy)
if nan_pixels.any():
    image_copy[nan_pixels] = 0.0


smooth_factor = target_res / beam_scale_pc

if target_res is not None and smooth_factor > 1:
    start = time.perf_counter()
    pixel_scale_pc = pixel_scale_arcsec * pc_per_arcsec
    sigma_kernel_pc = np.sqrt(target_res**2 - beam_scale_pc**2) / 2.355
    sigma_kernel_pix = sigma_kernel_pc / pixel_scale_pc 
    image_gauss = gaussian_filter(image_copy, sigma=sigma_kernel_pix, mode='constant', cval=0.0)
    gauss_time = time.perf_counter() - start
    print(f"Gaussian filter time: {gauss_time:.4f} seconds")
    
    print('Gaussian smoothing kernel:')
    print(f'  sigma = {sigma_kernel_pc:.2f} pc')
    print(f'  sigma = {sigma_kernel_pc / pc_per_arcsec:.2f} arcsec')
    print(f'  sigma = {sigma_kernel_pix:.2f} pixels')
    print(f'  FWHM  = {sigma_kernel_pc * 2.355:.2f} pc')
    print(f'  FWHM  = {(sigma_kernel_pc / pc_per_arcsec) * 2.355:.2f} arcsec')
    print(f'  FWHM  = {sigma_kernel_pix * 2.355:.2f} pixels')


##################################################################################################

    start = time.perf_counter()
    beam = Beam( major=BMAJ * u.deg, minor=BMIN * u.deg, pa= 0 * u.deg )
    # Target circular beam corresponding to 120 pc
    target_fwhm_arcsec = target_res / pc_per_arcsec
    target_beam = Beam( major=target_fwhm_arcsec * u.arcsec, minor=target_fwhm_arcsec * u.arcsec, pa=0 * u.deg)

    try:
        kernel = target_beam.deconvolve(beam).as_kernel(
            pixel_scale_arcsec * u.arcsec
        )

        image_fft = convolve_fft(
            image_copy,
            kernel,
            boundary="fill",
            fill_value=0.0,
            normalize_kernel=True,
            preserve_nan=True,
        )
        fft_time = time.perf_counter() - start
        print(f"fft time: {fft_time:.4f} seconds")
        # Print FFT kernel size
        kernel_beam = target_beam.deconvolve(beam)

        kernel_fwhm_arcsec = kernel_beam.major.to(u.arcsec).value
        kernel_fwhm_pc = kernel_fwhm_arcsec * pc_per_arcsec
        kernel_fwhm_pix = kernel_fwhm_arcsec / pixel_scale_arcsec

        print('')
        print('FFT convolution kernel:')
        print(f'  FWHM = {kernel_fwhm_pc:.2f} pc')
        print(f'  FWHM = {kernel_fwhm_arcsec:.2f} arcsec')
        print(f'  FWHM = {kernel_fwhm_pix:.2f} pixels')

    except BeamError:
        print(f"{base_name}: {beam_scale_pc} native beam is already larger than or incompatible with a {target_res} pc circular beam.")


import matplotlib.pyplot as plt
import numpy as np

# Residual map
residual = image_fft - image_gauss

# Shared colour scale for images
all_pixels = np.concatenate([
    image_gauss[np.isfinite(image_gauss)].ravel(),
    image_fft[np.isfinite(image_fft)].ravel()
])

vmin = np.nanpercentile(all_pixels, 0.5)
vmax = np.nanpercentile(all_pixels, 99.5)

# Symmetric residual scale around zero
res_lim = np.nanpercentile(np.abs(residual[np.isfinite(residual)]), 99.5)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

# Gaussian
im1 = axes[0].imshow(
    image_gauss,
    origin='lower',
    cmap='viridis',
    vmin=vmin,
    vmax=vmax
)
axes[0].set_title("Gaussian filter")

# FFT
im2 = axes[1].imshow(
    image_fft,
    origin='lower',
    cmap='viridis',
    vmin=vmin,
    vmax=vmax
)
axes[1].set_title("FFT convolution")

# Residual
im3 = axes[2].imshow(
    residual,
    origin='lower',
    cmap='RdBu_r',
    vmin=-res_lim,
    vmax=res_lim
)
axes[2].set_title("FFT - Gaussian")

# Colourbars
cbar1 = fig.colorbar(im2, ax=axes[:2], fraction=0.046, pad=0.04)
cbar1.set_label("Moment 0")

cbar2 = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
cbar2.set_label("Residual")

plt.show()

################################                        ########################################
################################ simple, pixel only mode ###################################
################################                        ########################################

image_copy = image.copy()
# ---------- NaN handling ----------
nan_pixels = np.isnan(image_copy)
if nan_pixels.any():
    image_copy[nan_pixels] = 0.0


start = time.perf_counter()
sigma_kernel_pix = pixel_sigma
image_gauss = gaussian_filter(image_copy, sigma=sigma_kernel_pix, mode='constant', cval=0.0)
gauss_time = time.perf_counter() - start
print(f"Gaussian filter time: {gauss_time:.4f} seconds")

##################################################################################################

try:
    start = time.perf_counter()
    kernel = Gaussian2DKernel(pixel_sigma)

    image_fft = convolve_fft(
        image_copy,
        kernel,
        boundary="fill",
        fill_value=0.0,
        normalize_kernel=True,
        preserve_nan=True,
    )
    fft_time = time.perf_counter() - start
    print(f"fft time: {fft_time:.4f} seconds")

except BeamError:
    print(f"{base_name}: {beam_scale_pc} native beam is already larger than or incompatible with a {target_res} pc circular beam.")

# Residual map
residual = image_fft - image_gauss

# Shared colour scale for images
all_pixels = np.concatenate([
    image_gauss[np.isfinite(image_gauss)].ravel(),
    image_fft[np.isfinite(image_fft)].ravel()
])

vmin = np.nanpercentile(all_pixels, 0.5)
vmax = np.nanpercentile(all_pixels, 99.5)

# Symmetric residual scale around zero
res_lim = np.nanpercentile(np.abs(residual[np.isfinite(residual)]), 99.5)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

# Gaussian
im1 = axes[0].imshow(
    image_gauss,
    origin='lower',
    cmap='viridis',
    vmin=vmin,
    vmax=vmax
)
axes[0].set_title("Gaussian filter")

# FFT
im2 = axes[1].imshow(
    image_fft,
    origin='lower',
    cmap='viridis',
    vmin=vmin,
    vmax=vmax
)
axes[1].set_title("FFT convolution")

# Residual
im3 = axes[2].imshow(
    residual,
    origin='lower',
    cmap='RdBu_r',
    vmin=-res_lim,
    vmax=res_lim
)
axes[2].set_title("FFT - Gaussian")

# Colourbars
cbar1 = fig.colorbar(im2, ax=axes[:2], fraction=0.046, pad=0.04)
cbar1.set_label("Moment 0")

cbar2 = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
cbar2.set_label("Residual")

plt.show()
import numpy as np
from astropy.io import fits
import os
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import NDData
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
from photutils.aperture import aperture_photometry

from astropy.table import Table


_original_read = Table.read

def patched_read(path, *args, **kwargs):

    if path == '/data/c3040163/llama/llama_main_properties.fits':
        path = '/Users/administrator/Astro/LLAMA/llama_main_properties.fits'

    return _original_read(path, *args, **kwargs)

# Apply patch
Table.read = patched_read

# Now import the module
import gas_analysis_seyfert

llamatab = Table.read('/Users/administrator/Astro/LLAMA/llama_main_properties.fits', format='fits')
colourbar_list = []
gas_analysis_seyfert.llamatab = llamatab
gas_analysis_seyfert.colourbar_list = colourbar_list

import cas_molecules

from metric_debug_utils import plot_moment_map_debug, make_projected_region_mask


def smoothness_debug(image, mask, pc_per_arcsec, pixel_scale_arcsec, flux_mask=False, **kwargs):
    plot_moment_map_debug(image, 'original image', flux_mask=flux_mask)
    smoothing_sigma_pc = 500 
    smoothing_sigma = (smoothing_sigma_pc / pc_per_arcsec) / pixel_scale_arcsec 
    size = max(1, int(round(smoothing_sigma))) 
    image_filled = np.nan_to_num(image, nan=0.0) 
    valid_mask = (~mask) & np.isfinite(image) 
    smooth_image = uniform_filter(image_filled, size=size, mode='constant',cval=0.0) # Nans replaced with 0 reduce the smoothed values of emission pixels
    plot_moment_map_debug(smooth_image, 'smoothed image', flux_mask=flux_mask,R_kpc = R_kpc, snmask=snmask, rebin=rebin, output_dir=output_dir, name=name)
    # if not flux_mask:
        # smooth_mask = uniform_filter(valid_mask.astype(float), size=size, mode='constant',cval=0.0) # Smoothing the mask will cancel out this effect
        # plot_moment_map_debug(smooth_mask, 'smoothed mask', flux_mask=flux_mask)
        # with np.errstate(invalid='ignore', divide='ignore'): smooth_image = smooth_image / smooth_mask # this normalises back up those pixels, avoiding divide-by-zero errs
    valid_smooth = (~mask) & np.isfinite(image) & np.isfinite(smooth_image)
    plot_moment_map_debug(valid_smooth, 'valid smooth', flux_mask=flux_mask,R_kpc = R_kpc, snmask=snmask, rebin=rebin, output_dir=output_dir, name=name)
    if np.sum(valid_smooth) == 0: return np.nan 
    diff_smooth = image[valid_smooth] - smooth_image[valid_smooth] # original image and smoothed image use orginal mask
    full = np.full(image.shape, np.nan)   # empty image
    full[valid_smooth] = diff_smooth              # place values back
    plot_moment_map_debug(full, 'smooth difference image', flux_mask=flux_mask,R_kpc = R_kpc, snmask=snmask, rebin=rebin, output_dir=output_dir, name=name)
    total_flux = np.sum(image[valid_smooth])
    print('S=',np.sum(diff_smooth[(diff_smooth>0)]) / total_flux if total_flux > 0 else np.nan)

def smoothness_debug_maskfirst(image, mask, pc_per_arcsec, pixel_scale_arcsec, flux_mask=False, **kwargs):
    plot_moment_map_debug(image, 'original image', flux_mask=flux_mask,R_kpc = R_kpc, snmask=snmask, rebin=rebin, output_dir=output_dir, name=name)
    smoothing_sigma_pc = 500 
    smoothing_sigma = (smoothing_sigma_pc / pc_per_arcsec) / pixel_scale_arcsec 
    size = max(1, int(round(smoothing_sigma))) 
    valid_mask = (~mask) & np.isfinite(image)
    plot_moment_map_debug(valid_mask, 'valid mask', flux_mask=flux_mask,R_kpc = R_kpc, snmask=snmask, rebin=rebin, output_dir=output_dir, name=name)
    valid_image = np.full_like(image, np.nan)
    valid_image[valid_mask] = image[valid_mask]
    plot_moment_map_debug(valid_image, 'valid image', flux_mask=flux_mask,R_kpc = R_kpc, snmask=snmask, rebin=rebin, output_dir=output_dir, name=name)
    smooth_image = uniform_filter(valid_image, size=size, mode='constant',cval=0.0) 

    plot_moment_map_debug(smooth_image, 'smoothed image', flux_mask=flux_mask,R_kpc = R_kpc, snmask=snmask, rebin=rebin, output_dir=output_dir, name=name)
    # if not flux_mask:
    #     smooth_mask = uniform_filter(valid_mask.astype(float), size=size, mode='constant',cval=0.0) # Smoothing the mask will cancel out this effect
    #     plot_moment_map_debug(smooth_mask, 'smoothed mask', flux_mask=flux_mask)
    #     with np.errstate(invalid='ignore', divide='ignore'): smooth_image = smooth_image / smooth_mask # this normalises back up those pixels, avoiding divide-by-zero errs
    valid_smooth = (~mask) & np.isfinite(image) & np.isfinite(smooth_image)
    plot_moment_map_debug(valid_smooth, 'valid smooth', flux_mask=flux_mask,R_kpc = R_kpc, snmask=snmask, rebin=rebin, output_dir=output_dir, name=name)
    if np.sum(valid_smooth) == 0: return np.nan 
    diff_smooth = image[valid_smooth] - smooth_image[valid_smooth] # original image and smoothed image use orginal mask
    full = np.full(image.shape, np.nan)   # empty image
    full[valid_smooth] = diff_smooth              # place values back
    plot_moment_map_debug(full, 'smooth difference image', flux_mask=flux_mask,R_kpc = R_kpc, snmask=snmask, rebin=rebin, output_dir=output_dir, name=name)
    total_flux = np.sum(image[valid_smooth])
    print('S=',np.sum(diff_smooth) / total_flux if total_flux > 0 else np.nan)
    


def asymmetry_debug(image, mask, **kwargs):
    plot_moment_map_debug(image, 'original image',flux_mask=flux_mask,R_kpc = R_kpc, snmask=snmask, rebin=rebin, output_dir=output_dir, name=name)
    image_rot = np.rot90(image, 2)
    plot_moment_map_debug(image_rot, 'rotated image',flux_mask=flux_mask,R_kpc = R_kpc, snmask=snmask, rebin=rebin, output_dir=output_dir, name=name)
    mask_rot = np.rot90(mask, 2)
    plot_moment_map_debug(mask, 'valid mask assym', flux_mask=flux_mask,R_kpc = R_kpc, snmask=snmask, rebin=rebin, output_dir=output_dir, name=name)
    if np.sum(~mask) == 0 or np.sum(~mask_rot) == 0:
        return np.nan
    diff = abs(image[~mask] - image_rot[~mask_rot])
    valid = (~mask) & (~mask_rot)
    full = np.full(image.shape, np.nan)   # empty image
    full[valid] = np.abs(image[valid] - image_rot[valid])
    plot_moment_map_debug(full, 'assym difference image', flux_mask=flux_mask,R_kpc = R_kpc, snmask=snmask, rebin=rebin, output_dir=output_dir, name=name)
    total = image[~mask]
    print('A=',np.sum(diff) / np.sum(total) if np.sum(total) > 0 else np.nan)

def asymmetry_debug_mask_after(image, mask, **kwargs):
    plot_moment_map_debug(image, 'original image',flux_mask=flux_mask,R_kpc = R_kpc, snmask=snmask, rebin=rebin, output_dir=output_dir, name=name)
    image_rot = np.rot90(image, 2)
    plot_moment_map_debug(image_rot, 'rotated image',flux_mask=flux_mask,R_kpc = R_kpc, snmask=snmask, rebin=rebin, output_dir=output_dir, name=name)
    #mask_rot = np.rot90(mask, 2)
    plot_moment_map_debug(mask, 'valid mask assym', flux_mask=flux_mask,R_kpc = R_kpc, snmask=snmask, rebin=rebin, output_dir=output_dir, name=name)
    if np.sum(~mask) == 0:
        return np.nan
    diff = abs(image- image_rot)
    full = np.full(image.shape, np.nan)   # empty image
    full[~mask] = diff[~mask]            # place values back
    plot_moment_map_debug(full, 'assym difference image alt', flux_mask=flux_mask,R_kpc = R_kpc, snmask=snmask, rebin=rebin, output_dir=output_dir, name=name)
    total = image[~mask]
    print('A=',np.sum(diff[~mask]) / np.sum(total) if np.sum(total) > 0 else np.nan)



def gini_debug(image, mask, **kwargs):
    valid_data = image[~mask & np.isfinite(image)].flatten()
    if len(valid_data) == 0:
        return np.nan
    sorted_vals = np.sort(valid_data)
    n = len(sorted_vals)
    total = np.sum(sorted_vals)
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    mean = total / (n-1) # changed from total/n to match formula from cas_molecules
    print('G=', 1/(mean*n*(n-1)) * np.sum((2*index - n - 1) * sorted_vals))

def gini_debug_alt(image, mask):
    valid_data = image[~mask & np.isfinite(image)].flatten()
    if len(valid_data) == 0:
        return np.nan
    sorted_vals = np.sort(valid_data)
    n = len(sorted_vals)
    total = np.sum(sorted_vals)
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1)

    G = np.sum((2*index - n - 1) * sorted_vals) / (n * total)

    print('G=', G)


def concentration_single_new(image, mask, pixel_scale_arcsec, pc_per_arcsec, PA, I, **kwargs):

    valid = (~mask) & np.isfinite(image)

    # Apply mask directly to image (important for photutils)
    data = np.where(valid, image, np.nan)

    # --- Build apertures ---
    _, aperture_50pc_region, _ = make_projected_region_mask(
        image.shape, 50/1000, pc_per_arcsec, pixel_scale_arcsec, PA, I)

    _, aperture_200pc_region, _ = make_projected_region_mask(
        image.shape, 200/1000, pc_per_arcsec, pixel_scale_arcsec, PA, I)
    
    plot_moment_map_debug(data, 'conc apertures with I', flux_mask=flux_mask,R_kpc = R_kpc, snmask=snmask, rebin=rebin, output_dir=output_dir, name=name,aperture=[aperture_50pc_region, aperture_200pc_region])

    # Convert regions → photutils apertures
    aperture_50pc = aperture_50pc_region
    aperture_200pc = aperture_200pc_region

    # --- Sub-pixel photometry ---
    phot_50 = aperture_photometry(data, aperture_50pc, method='exact')
    phot_200 = aperture_photometry(data, aperture_200pc, method='exact')

    flux_50 = phot_50['aperture_sum'][0]
    flux_200 = phot_200['aperture_sum'][0]

    # --- Normalisation (surface density-like) ---
    area_50 = aperture_50pc.area
    area_200 = aperture_200pc.area

    flux_50 /= area_50
    flux_200 /= area_200

    if flux_50 <= 0 or flux_200 <= 0:
        return np.nan

    return np.log10(flux_50 / flux_200)


def concentration_single(image, mask, pixel_scale_arcsec, pc_per_arcsec, PA, I, **kwargs):
    y, x = np.indices(image.shape)
    center = (x.max() / 2, y.max() / 2)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r_pc = r * pixel_scale_arcsec * pc_per_arcsec
    valid = (~mask) & np.isfinite(image)
    flux_50 = np.sum(image[(r_pc < 50) & valid]) / 50**2
    flux_200 = np.sum(image[(r_pc < 200) & valid]) / 200**2
    _, aperture_50pc_region, _ = make_projected_region_mask(
        image.shape, 50/1000, pc_per_arcsec, pixel_scale_arcsec, PA, 0)

    _, aperture_200pc_region, _ = make_projected_region_mask(
        image.shape, 200/1000, pc_per_arcsec, pixel_scale_arcsec, PA, 0)
    
    plot_moment_map_debug(image, 'conc apertures circular', flux_mask=flux_mask,R_kpc = R_kpc, snmask=snmask, rebin=rebin, output_dir=output_dir, name=name,aperture=[aperture_50pc_region, aperture_200pc_region])

    
    if flux_50 <= 0 or flux_200 <= 0:
        return np.nan
    return np.log10(flux_50 / flux_200)


#################### arguments ####################
R_kpc = 1.5
co32 = False
flux_mask = False
snmask = False
normalise_norm = False
output_dir = '/Users/administrator/Astro/LLAMA/ALMA/AGN/PHANGS_m0_for_test/outputs'
res_src = 'native'
rebin = None
PHANGS_mask = 'strict'
##################################################
#INPUT FILE #


subdir = '/Users/administrator/Astro/LLAMA/ALMA/AGN/PHANGS_m0_for_test/'

file = '/Users/administrator/Astro/LLAMA/ALMA/AGN/PHANGS_m0_for_test/NGC5506_12m_co21_strict_mom0.fits'
#file = '/Users/administrator/Astro/LLAMA/ALMA/comp_samples/m0/NGC7172_mom0.fits'
file = '/Users/administrator/Astro/LLAMA/ALMA/AGN/PHANGS_m0_for_test/NGC2110_12m_co21_120pc_strict_mom0.fits'
file = '/Users/administrator/Astro/LLAMA/ALMA/AGN/PHANGS_m0_for_test/ESO208_12m_co21_120pc_strict_mom0.fits'

##################################################


if rebin == None:
    file = file
elif rebin == 120:
    file = file.replace('_strict_mom0.fits', '_120pc_strict_mom0.fits')
else:    raise ValueError("Unsupported rebin value. Use None or 120.")

base = os.path.basename(file)
print(base)
extension = os.path.splitext(base)[1]

name = base.split("_12m")[0]
#name = 'NGC7172'
name = 'ESO208'

print(f'running {name} with R={R_kpc}kpc, rebin={rebin}, flux_mask={flux_mask}')

#print(os.path.exists(file))

image_untrimmed = fits.getdata(file, memmap=True)
#print(np.nanmax(image_untrimmed))
mask_untrimmed = np.isnan(image_untrimmed)


main_meta = gas_analysis_seyfert.resolve_galaxy_beam_scale(
name=name,
fits_file=file,
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

print(D_Mpc)
print('I=', I)
print('PA=', PA)

R_pixel = int(R_kpc * (206.265 / D_Mpc) / pixel_scale_arcsec)
jy_per_K = float(header.get("JYTOK", 0))
pixel_scale_pc = pixel_scale_arcsec * pc_per_arcsec
pixel_area_pc2 = pixel_scale_pc**2

beam_scales_pc = [beam_scale_pc]
beam_scale_labels = [name]


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
mask = np.ones_like(image, dtype=bool)

x1i, x2i = max(0, x1), min(nx_full, x2)
y1i, y2i = max(0, y1), min(ny_full, y2)

xp1, xp2 = x1i - x1, x1i - x1 + (x2i - x1i)
yp1, yp2 = y1i - y1, y1i - y1 + (y2i - y1i)

# print("cx, cy:", cx, cy)
# print("x1, x2:", x1, x2)
# print("y1, y2:", y1, y2)

# print("x1i, x2i:", x1i, x2i, "width:", x2i - x1i)
# print("y1i, y2i:", y1i, y2i, "height:", y2i - y1i)

# print("source shape:", image_untrimmed[y1i:y2i, x1i:x2i].shape)
# print("dest shape:", image[yp1:yp2, xp1:xp2].shape)

image[yp1:yp2, xp1:xp2] = image_untrimmed[y1i:y2i, x1i:x2i]
mask[yp1:yp2, xp1:xp2] = mask_untrimmed[y1i:y2i, x1i:x2i]

        # ---------- NaN handling ----------
nan_pixels = np.isnan(image)
if nan_pixels.any():
    image[nan_pixels] = 0.0
    mask[nan_pixels] = False


# ---------- Update WCS ----------
wcs_trimmed = wcs_full.deepcopy()
wcs_trimmed.wcs.crpix[0] -= x1
wcs_trimmed.wcs.crpix[1] -= y1

image_nd = NDData(data=image, wcs=wcs_trimmed)

# ---------- Flux mask ----------
if flux_mask == True:
    total_flux = np.nansum(image[~mask])
    f = 2.0
    ratio = 1.0
    while ratio > 0.9:
        flux_mask_90, flux_aperture_90, R_90 = gas_analysis_seyfert.make_projected_region_mask(
            image.shape, R_kpc * f * 1.42 , pc_per_arcsec, pixel_scale_arcsec, PA, I
        )
        flux_90 = np.nansum(image[~flux_mask_90 & ~mask])
        ratio = flux_90 / total_flux if total_flux > 0 else 0.0
        f -= 0.01

    print(
        f"Flux in aperture = {ratio} of total. R_90 (kpc): ",
        round(R_90, 2)
    )
    mask = flux_mask_90 | mask
    outpath = output_dir + f'/{name}' + f'/{name}_flux90mask.fits'
    if not os.path.exists(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath))
    fits.writeto(outpath, mask.astype(int), overwrite=True)
    aperture_to_plot = flux_aperture_90
else:
    aperture_to_plot = None


# ------------------ Signal to noise mask ------------------

if snmask:

    # try:
    #     sn_mask = fits.getdata(f'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/AGN/masks/{name}_snmask.fits', memmap=True).astype(bool)
    # except:
    #     sn_mask = fits.getdata(f'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/inactive/masks/{name}_snmask.fits', memmap=True).astype(bool)

    # mask = sn_mask

    sn_mask = image != float(0)
    mask = ~sn_mask | mask  # combine with existing mask



# ---------- Plot ----------

# norm_type = 'sqrt' if normalise_norm else 'linear'
# gas_analysis_seyfert.plot_moment_map(
#     image_nd, output_dir, name,
#     BMAJ, BMIN, R_kpc, rebin,
#     PHANGS_mask, flux_mask,
#     aperture=aperture_to_plot, res_src=res_src, norm_type=norm_type, normalise_norm=normalise_norm
# )


###### metrics debugging ######

# smoothness_debug(image, mask, pc_per_arcsec=pc_per_arcsec, pixel_scale_arcsec=pixel_scale_arcsec, flux_mask=flux_mask)
# # asymmetry_debug(image, mask)
# asymmetry_debug_mask_after(image, mask)
# gini_debug(image, mask)
 
# images = [image]
# inc = [I]
# pa = [PA]
# xc = [cx]
# yc = [cy]
# cellsize = [pixel_scale_pc]
# scale = [(500 / pc_per_arcsec) / pixel_scale_arcsec]

# for i in range(len(images)):


#     cas = cas_molecules.cas_molecules(
#         image=images[i],
#         inc=inc[i],
#         pa=pa[i],
#         xc=xc[i],
#         yc=yc[i],
#         cellsize=cellsize[i],
#         scale=scale[i]
#     )

#     results = cas.run_all(clip=None)



print('circular C:',concentration_single(image, mask, pixel_scale_arcsec, pc_per_arcsec, PA, I))
print('elliptical C (new):',concentration_single_new(image, mask, pixel_scale_arcsec, pc_per_arcsec, PA, I))
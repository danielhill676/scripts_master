import numpy as np
from astropy.io import fits
import os
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import NDData
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
from regions import EllipsePixelRegion, PixCoord
from matplotlib.patches import Ellipse
from astropy.table import Table

def plot_moment_map_debug(image, title,flux_mask=False,R_kpc=1.0, snmask=False, rebin=None, output_dir='debug_plots', name='debug', aperture=None,): 
    # Initialise plot
    fontsize = 35 * R_kpc
    plt.rcParams.update({'font.size': fontsize})
    figsize = 18 * R_kpc
    fig = plt.figure(figsize=(figsize , figsize),constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.margins(x=0,y=0)
    ax.set_axis_off()

    im=plt.imshow(image.data,origin='lower',cmap='RdBu_r')
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)

    if aperture is not None:
        for i in range(len(aperture)):
            # Regions EllipsePixelRegion parameters:
            cx = aperture[i].center.x
            cy = aperture[i].center.y
            width = aperture[i].width     # = 2*a in pixels
            height = aperture[i].height   # = 2*b in pixels
            angle_deg = aperture[i].angle.to_value("deg")

            ellipse_patch = Ellipse(
                (cx, cy), width=width, height=height,
                angle=angle_deg,
                edgecolor='lime',       # visible bright color
                facecolor='none',
                linewidth=3,
            )
            ax.add_patch(ellipse_patch)


    plt.colorbar(im, fraction=0.046, pad=0.04, label=None)
    plt.title(title)
    if flux_mask:
        title += '_flux_mask'
    if snmask:
        title += '_snmask'
    if rebin is not None:
        title += f'_{rebin}pc'
    path = output_dir + f'/{name}' + f'/{name}_{title}.png'
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


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
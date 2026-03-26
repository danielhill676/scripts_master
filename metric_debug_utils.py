import numpy as np
from astropy.io import fits
import os
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import NDData
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt

from astropy.table import Table

def plot_moment_map_debug(image, title,flux_mask=False,R_kpc=1.0, snmask=False, rebin=None, output_dir='debug_plots', name='debug'): 
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
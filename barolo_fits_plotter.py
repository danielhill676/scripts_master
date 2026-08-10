
import os
import gc
import time
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
import astropy
from astropy.visualization import simple_norm
from astropy.visualization.wcsaxes import add_scalebar
from astropy.visualization.wcsaxes import add_beam
from astropy.nddata import NDData
from astropy.wcs import WCS
from matplotlib.patches import Ellipse
from reproject import reproject_interp

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
})


def plot_moment_map(
    image,
    outfolder,
    name_short,
    type,
    R_kpc,
    norm_type='sqrt',
    res_src='native',
    normalise_norm=False,
    noise=None,
    mom=0
):

    global colourbar_list

    # ------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------

    fontsize = 35 * R_kpc
    plt.rcParams.update({'font.size': fontsize})

    figsize = 18 * R_kpc

    fig = plt.figure(
        figsize=(figsize, figsize),
        frameon=False
    )

    ax = fig.add_axes(
        [0, 0, 1, 1],
        projection=image.wcs.celestial
    )
    ax.margins(x=0, y=0)
    ax.set_axis_off()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.coords.frame.patch.set_visible(False)


    # ------------------------------------------------------------
    # Determine colormap, vmin, vmax and normalisation
    # ------------------------------------------------------------

    finite_data = image.data[np.isfinite(image.data)]

    if finite_data.size == 0:
        print(f"{name_short}: image contains no finite pixels.")
        plt.close(fig)
        return


    if mom == 0:

        # Initial limits
        vmin = (
            2 * noise
            if noise is not None and np.isfinite(noise)
            else 0
        )

        vmax = np.nanpercentile(finite_data, 99.5)

        # --------------------------------------------------------
        # Store limits for common normalisation
        # --------------------------------------------------------

        if not normalise_norm and res_src in ['native', 'rebin']:
            colourbar_list.append(vmin)
            colourbar_list.append(vmax)

        # Use common limits when normalising between maps
        if normalise_norm and res_src in ['native', 'rebin']:
            vmin = np.nanmin(colourbar_list)
            vmax = np.nanmax(colourbar_list)

        if vmin >= vmax:
            vmin = 0

        # Original behaviour when not normalising
        if not normalise_norm:
            vmin = 0
            vmax = np.nanmax(finite_data)

        # --------------------------------------------------------
        # Normalisation
        # --------------------------------------------------------

        if norm_type == 'sqrt':
            norm = simple_norm(
                image.data,
                'sqrt',
                vmin=vmin,
                vmax=vmax
            )

        elif norm_type == 'linear':
            norm = simple_norm(
                image.data,
                'linear',
                vmin=vmin,
                vmax=vmax
            )

        else:
            raise ValueError(
                f"Unknown norm_type: {norm_type}"
            )

        cmap = plt.cm.inferno.copy()
        cmap.set_bad('lightgrey',alpha=1)


    elif mom == 1:

        vmax = np.nanpercentile(finite_data, 97.5)
        vmin = np.nanpercentile(finite_data, 2.5)

        norm = simple_norm(
            image.data,
            'linear',
            vmin=vmin,
            vmax=vmax
        )


        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad('black',alpha=1)


    elif mom == 2:

        vmax = np.nanpercentile(finite_data, 97.5)
        vmin = np.nanpercentile(finite_data, 2.5)

        norm = simple_norm(
            image.data,
            'linear',
            vmin=vmin,
            vmax=vmax
        )

        cmap = 'jet'



    else:
        raise ValueError(f"Unknown moment: {mom}")


    # ------------------------------------------------------------
    # Plot image
    # ------------------------------------------------------------

    im = ax.imshow(
        image.data,
        origin='lower',
        norm=norm,
        cmap=cmap,
        interpolation='nearest'
    )

    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)


    # ------------------------------------------------------------
    # Colourbar
    #
    # normalise_norm=False:
    #     small colourbar INSIDE the figure
    #
    # normalise_norm=True:
    #     colourbar saved separately
    # ------------------------------------------------------------

    if not normalise_norm:

        # Manually positioned colourbar
        cbar_ax = fig.add_axes([
            0.82,
            0.12,
            0.035,
            0.30
        ])

        cb = fig.colorbar(
            im,
            cax=cbar_ax,
            orientation='vertical'
        )

        colour = 'white' if mom == 1 else 'black'
        cb.ax.tick_params(
            axis='y',
            which='major',
            labelsize=fontsize * 0.65,
            length=3,
            width=0.8,
            direction='out',
            labelcolor=colour
            
        )
        cb.outline.set_edgecolor(colour)

        if mom == 0:
            cb.set_label(
                r'Intensity (K km/s)',
                fontsize=fontsize * 0.7,
                labelpad=5,color=colour
            )

        elif mom == 1:
            cb.set_label(
                r'Velocity (km s$^{-1}$)',
                fontsize=fontsize * 0.7,
                labelpad=5,color=colour
            )

        elif mom == 2:
            cb.set_label(
                r'Dispersion (km s$^{-1}$)',
                fontsize=fontsize * 0.7,
                labelpad=5,color=colour
            )

    # ------------------------------------------------------------
    # Save main figure
    # ------------------------------------------------------------

    plot_dir = os.path.join(
        outfolder,
        f'plots/{name_short}'
    )

    os.makedirs(
        plot_dir,
        exist_ok=True
    )

    path = os.path.join(
        plot_dir,
        f'{name_short}_{type}_mom{mom}.pdf'
    )
    if normalise_norm:
        path = os.path.join(
        plot_dir,
        f'{name_short}_{type}_mom{mom}_norm.pdf'
    )

    plt.savefig(
        path,
        pad_inches=0.0
    )

    plt.close(fig)


    # ------------------------------------------------------------
    # If normalise_norm=True, save colourbar separately
    # ------------------------------------------------------------

    if normalise_norm:

        cbar_fig, cbar_ax = plt.subplots(
            figsize=(4, figsize * 7.5)
        )

        # Use EXACTLY the same norm and cmap as the image
        cb = plt.colorbar(
            plt.cm.ScalarMappable(
                norm=norm,
                cmap=cmap
            ),
            cax=cbar_ax,
            orientation='vertical'
        )

        # --------------------------------------------------------
        # Five evenly spaced ticks
        # --------------------------------------------------------

        if (
            np.isfinite(vmin)
            and np.isfinite(vmax)
            and vmax > vmin
        ):

            import matplotlib.ticker as mticker

            cb.set_ticks(
                np.linspace(
                    vmin,
                    vmax,
                    5
                )
            )

            cb.ax.yaxis.set_major_formatter(
                mticker.ScalarFormatter()
            )

            cb.ax.tick_params(
                axis='y',
                which='major',
                labelsize=fontsize * 3.5,
                length=8,
                width=1.5,
                direction='out'
            )

        # --------------------------------------------------------
        # Colourbar label
        # --------------------------------------------------------

        if mom == 0:

            cb.set_label(
                r'Surface density ($M_{\odot}\,\mathrm{pc}^{-2}$)',
                fontsize=fontsize * 5,
                labelpad=20
            )

        elif mom == 1:

            cb.set_label(
                r'Velocity (km s$^{-1}$)',
                fontsize=fontsize * 5,
                labelpad=20
            )

        elif mom == 2:

            cb.set_label(
                r'Dispersion (km s$^{-1}$)',
                fontsize=fontsize * 5,
                labelpad=20
            )

        # --------------------------------------------------------
        # Colourbar filename
        # --------------------------------------------------------

        colourbar_dir = os.path.join(
            outfolder,
            'colourbars'
        )

        os.makedirs(
            colourbar_dir,
            exist_ok=True
        )

        colourbar_path = os.path.join(
            colourbar_dir,
            f'{name}_colourbar_mom{mom}.pdf'
        )

        plt.savefig(
            colourbar_path,
            bbox_inches='tight',
            pad_inches=0
        )

        plt.close(cbar_fig)




import os
import gc
import time
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import NDData

from reproject import reproject_interp


outerdir = "/Users/administrator/Astro/LLAMA/ALMA/barolo/phangsmask"
outerdir_phangs = "/Users/administrator/Astro/LLAMA/ALMA/pipeline_m0"
outputdir = outerdir
R_kpc = 1.5
PHANGS_mask = 'strict'

maps = {}

for name in sorted(os.listdir(outerdir)):

    maps_dir = os.path.join(outerdir, name, "maps")

    if not os.path.isdir(maps_dir):
        continue

    maps[name] = {}

    # ------------------------------------------------------------
    # Load BAROLO moment maps
    # ------------------------------------------------------------

    for n in [0, 1, 2]:

        for map_type, filename in [
            ("true", f"{name}_{n}mom.fits"),
            ("local", f"{name}_local_{n}mom.fits"),
        ]:

            filepath = os.path.join(maps_dir, filename)

            if not os.path.exists(filepath):
                print(f"Missing: {filepath}")
                continue

            with fits.open(filepath) as hdul:

                data = hdul[0].data.copy()
                header = hdul[0].header.copy()

                wcs = WCS(header)

                nddata = NDData(
                    data=data,
                    wcs=wcs
                )

            maps[name][(map_type, n)] = nddata

            print(f"Loaded: {filepath}")


    # ------------------------------------------------------------
    # Check that all six BAROLO maps exist
    # ------------------------------------------------------------

    required_maps = [
        ("true", 0),
        ("true", 1),
        ("true", 2),
        ("local", 0),
        ("local", 1),
        ("local", 2),
    ]

    if not all(key in maps[name] for key in required_maps):
        print(f"Skipping {name}: not all BAROLO moment maps found.")
        continue


    # ------------------------------------------------------------
    # Load PHANGS strict CO(2-1) moment-0 map
    # ------------------------------------------------------------

    co21_file = os.path.join(
        outerdir_phangs,
        name,
        f"{name}_12m_co21_strict_mom0.fits"
    )

    if not os.path.exists(co21_file):
        print(f"Missing CO(2-1) mask for {name}: {co21_file}")
        continue

    with fits.open(co21_file) as hdul:

        co21_data = hdul[0].data.copy()
        co21_header = hdul[0].header.copy()
        co21_wcs = WCS(co21_header)
        BMAJ = co21_header.get("BMAJ", np.nan)
        BMIN= co21_header.get("BMIN", np.nan)

    print(f"Loaded CO(2-1) mask: {co21_file}")


    # ------------------------------------------------------------
    # Apply CO(2-1) mask to every BAROLO map
    # ------------------------------------------------------------

    for key, image in maps[name].items():

        map_type, n = key

        target_data = image.data
        target_wcs = image.wcs

        # Reproject the CO(2-1) mask onto the target map's
        # exact WCS/pixel grid.
        co21_reprojected, footprint = reproject_interp(
            (co21_data, co21_wcs),
            target_wcs,
            shape_out=target_data.shape
        )

        # Pixels are masked if:
        #   1. CO(2-1) is NaN
        #   2. CO(2-1) is exactly zero
        #   3. There is no valid reprojection footprint
        mask = (
            ~np.isfinite(co21_reprojected)
            | (co21_reprojected == 0)
            | (footprint == 0)
        )

        # Copy the BAROLO map so the original remains untouched
        masked_data = target_data.copy()

        # Apply mask
        masked_data[mask] = np.nan

        # Replace NDData object with masked version
        maps[name][key] = NDData(
            data=masked_data,
            wcs=target_wcs
        )

        print(
            f"Masked {name} {map_type} moment {n}: "
            f"{np.sum(mask)} pixels masked"
        )


    # ------------------------------------------------------------
    # Retrieve the masked maps
    # ------------------------------------------------------------

    mom0_true = maps[name][("true", 0)]
    mom1_true = maps[name][("true", 1)]
    mom2_true = maps[name][("true", 2)]

    mom0_fit = maps[name][("local", 0)]
    mom1_fit = maps[name][("local", 1)]
    mom2_fit = maps[name][("local", 2)]


    print(f"masked maps for {name}")

    mom0_res = NDData(
        data=mom0_fit.data - mom0_true.data,
        wcs=mom0_fit.wcs
    )

    mom1_res = NDData(
        data=mom1_fit.data - mom1_true.data,
        wcs=mom1_fit.wcs
    )

    mom2_res = NDData(
        data=mom2_fit.data - mom2_true.data,
        wcs=mom2_fit.wcs
    )
    colourbar_list = []

    for map, type in zip([mom0_true,mom0_fit, mom0_res],['true','fit','res']):

        plot_moment_map(
            map, outputdir, name, type, R_kpc, norm_type='sqrt',mom = 0
        )
        plot_moment_map(
            map, outputdir, name, type, R_kpc, norm_type='sqrt',mom = 0, normalise_norm=True
        )

    colourbar_list = []
    for map, type in zip([mom1_true,mom1_fit, mom1_res],['true','fit','res']):

        plot_moment_map(
            map, outputdir, name, type, R_kpc, norm_type='linear',mom = 1
        )
        plot_moment_map(
            map, outputdir, name, type, R_kpc, norm_type='linear',mom = 1, normalise_norm=True
        )

    for map, type in zip([mom2_true,mom2_fit, mom2_res],['true','fit','res']):

        plot_moment_map(
            map, outputdir, name, type, R_kpc, norm_type='linear',mom = 2
        )
        plot_moment_map(
            map, outputdir, name, type, R_kpc, norm_type='linear',mom = 2, normalise_norm=True
        )


import os
import gc
import time
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import NDData
from reproject import reproject_interp

import matplotlib.pyplot as plt
from astropy.visualization import simple_norm


# ==========================================================================================
# RUN NAME
runname = 'phangsmask_cenfroz_axisfree'
# runname = 'phangsmask_cenfroz_axisfree_zfree'
# =========================================================================================
# RUN NUMBER
runn = 3
# runn = 4

# MACHINE
seyfert = True


# ==========================================================================================
# COMPARISON RUN
# Set runname2 = None to use the standard true/fit/res workflow.
# Set runname2 to another BAROLO run to compare the two fits.
# ==========================================================================================
runname2 = 'phangsmask_cenfroz_axisfree_zfree'
# runname2 = None
runn2 = 4


# ================================================================
# Configuration
# ================================================================

outerdir = f"/Users/administrator/Astro/LLAMA/ALMA/barolo/{runname}" if not seyfert else f"/data/c3040163/llama/alma/barolo/{runname}"

outerdir2 = (
    None
    if runname2 is None
    else (
        f"/Users/administrator/Astro/LLAMA/ALMA/barolo/{runname2}"
        if not seyfert
        else
        f"/data/c3040163/llama/alma/barolo/{runname2}"
    )
)

outputdir = outerdir


R_kpc = 1.5
PHANGS_mask = "strict"


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
})


# ================================================================
# BAROLO ring parameters
# ================================================================

def read_barolo_parameters(outfolder):
    """
    Read the final BAROLO tilted-ring parameters.

    Uses the same columns as BAROLO's plotting script:

        column 1  = radius
        column 4  = inclination
        column 5  = PA
        column 9  = xpos
        column 10 = ypos
        column 11 = vsys

    Returns
    -------
    xcen, ycen, pa, inc, vsys
    """

    rings_file = os.path.join(
        outfolder,
        "rings_final2.txt"
    )

    if not os.path.exists(rings_file):
        raise FileNotFoundError(
            f"Missing BAROLO rings file: {rings_file}"
        )

    rad, inc, pa, xpos, ypos, vsys, vrot, disp, z, vrad = np.genfromtxt(
        rings_file,
        usecols=(1, 4, 5, 9, 10, 11, 3, 3, 6, 12),
        unpack=True
    )

    xcen = np.nanmean(xpos)
    ycen = np.nanmean(ypos)
    pa_mean = np.nanmean(pa)
    inc_mean = np.nanmean(inc)
    vsys_mean = np.nanmean(vsys)
    vrot_mean = np.nanmean(vrot)
    disp_mean = np.nanmean(disp)
    z_mean = np.nanmean(z)
    vrad_mean = np.nanmean(vrad)

    return (
        xcen,
        ycen,
        pa_mean,
        inc_mean,
        vsys_mean,
        vrot_mean,
        disp_mean,
        z_mean,
        vrad_mean
    )


# ================================================================
# Plotting function
# ================================================================

def plot_moment_map(
    image,
    outfolder,
    name_short,
    type,
    R_kpc,
    norm_type="sqrt",
    res_src="native",
    normalise_norm=False,
    noise=None,
    mom=0,
    barolo_params=None, cbar = False
):
    """
    Plot one BAROLO moment map.

    Parameters
    ----------
    image : NDData
        Moment map to plot.

    outfolder : str
        Main BAROLO output directory.

    name_short : str
        Galaxy name.

    type : str
        'true', 'fit', or 'res'.

    R_kpc : float
        Retained for compatibility with the existing script.

    norm_type : str
        'sqrt' or 'linear'.

    normalise_norm : bool
        Whether to use the common normalisation.

    noise : float or None
        Noise level for moment 0.

    mom : int
        Moment number.

    barolo_params : tuple or None
        (xcen, ycen, pa, inc, vsys)
    """

    global colourbar_list

    # ------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------

    fontsize = 35 * R_kpc
    plt.rcParams.update({
        "font.size": fontsize
    })

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
    # Determine image dimensions
    # ------------------------------------------------------------

    data = image.data

    ny, nx = data.shape

    extent = [
        0,
        nx,
        0,
        ny
    ]

    # ------------------------------------------------------------
    # Determine colormap, vmin, vmax and normalisation
    # ------------------------------------------------------------

    finite_data = data[np.isfinite(data)]

    if finite_data.size == 0:

        print(
            f"{name_short}: image contains no finite pixels."
        )

        plt.close(fig)

        return

    # ------------------------------------------------------------
    # Default limits
    # ------------------------------------------------------------
    if mom == 0:
        vmin = 0
        vmax = np.nanpercentile(finite_data,99.5)
    else:
        vmin = np.nanpercentile(finite_data,2.5)
        vmax = np.nanpercentile(finite_data,97.5)

    # --------------------------------------------------------
    # Store limits for common normalisation
    # --------------------------------------------------------

    if not normalise_norm:

        colourbar_list.append(vmin)
        colourbar_list.append(vmax)

    # --------------------------------------------------------
    # Use common limits when normalising
    # --------------------------------------------------------

    if normalise_norm:

        vmin = np.nanmin(colourbar_list)
        vmax = np.nanmax(colourbar_list)


    if runname2 is None or normalise_norm or type == f"{runname2}_res":

        # --------------------------------------------------------
        # Normalisation
        # --------------------------------------------------------


        if norm_type == "sqrt":

            norm = simple_norm(
                data,
                "sqrt",
                vmin=vmin,
                vmax=vmax
            )

        elif norm_type == "linear":

            norm = simple_norm(
                data,
                "linear",
                vmin=vmin,
                vmax=vmax
            )

        else:

            raise ValueError(
                f"Unknown norm_type: {norm_type}"
            )

        # --------------------------------------------------------
        # Colourmaps (order dependent)
        # --------------------------------------------------------

        if mom == 0:

            cmap = plt.cm.inferno.copy()

            cmap.set_bad(
                "black",
                alpha=1
            )

        elif mom == 1:

            cmap = plt.cm.RdBu_r.copy()

            cmap.set_bad(
                "black",
                alpha=1
            )

        elif mom == 2:

            cmap = plt.cm.jet.copy()

            cmap.set_bad(
                "black",
                alpha=1
            )

        else:

            raise ValueError(
                f"Unknown moment: {mom}"
            )

        # ============================================================
        # Plot image
        # ============================================================


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
        # Preserve the exact image limits
        # ------------------------------------------------------------

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # ------------------------------------------------------------
        # BAROLO annotations
        # ------------------------------------------------------------

        if barolo_params is not None:

            xcen, ycen, pa, inc, vsys, vrot, disp, z, vrad  = barolo_params

            # Annotation colour
            annotation_colour = (
                "white"
                if mom == 2
                else "lime"
            )

            # --------------------------------------------------------
            # Centre
            # --------------------------------------------------------

            ax.plot(
                xcen,
                ycen,
                marker='*',
                color=annotation_colour,
                markersize=50,
                mew=1.5,
                linestyle='None',
                zorder=20
            )

            # --------------------------------------------------------
            # PA line
            # --------------------------------------------------------

            theta = np.radians(pa - 90.0)

            dx = np.cos(theta)
            dy = np.sin(theta)

            xmin, xmax = xlim
            ymin, ymax = ylim

            t = np.linspace(
                -2 * max(nx, ny),
                2 * max(nx, ny),
                1000
            )

            x_line = xcen + t * dx
            y_line = ycen + t * dy

            valid = (
                (x_line >= xmin)
                & (x_line <= xmax)
                & (y_line >= ymin)
                & (y_line <= ymax)
            )

            ax.plot(
                x_line[valid],
                y_line[valid],
                '--',
                color=annotation_colour,
                linewidth=6,
                zorder=19
            )


        # ------------------------------------------------------------
        # Restore exact image limits
        # ------------------------------------------------------------

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # ============================================================
        # Colourbar
        # ============================================================

        if not normalise_norm:

            cbar_ax = fig.add_axes([
                0.82,
                0.12,
                0.035,
                0.30
            ])

            cb = fig.colorbar(
                im,
                cax=cbar_ax,
                orientation="vertical"
            )

            colour = (
                "white"
            )

            cb.ax.tick_params(
                axis="y",
                which="major",
                labelsize=fontsize * 0.65,
                length=3,
                width=0.8,
                direction="out",
                labelcolor=colour
            )

            cb.outline.set_edgecolor(
                colour
            )

            if mom == 0:

                cb.set_label(
                    r"Intensity (K km/s)",
                    fontsize=fontsize * 0.7,
                    labelpad=5,
                    color=colour
                )

            elif mom == 1:

                cb.set_label(
                    r"Velocity (km s$^{-1}$)",
                    fontsize=fontsize * 0.7,
                    labelpad=5,
                    color=colour
                )

            elif mom == 2:

                cb.set_label(
                    r"Dispersion (km s$^{-1}$)",
                    fontsize=fontsize * 0.7,
                    labelpad=5,
                    color=colour
                )

        # ============================================================
        # Output directory
        # ============================================================

        plot_dir = os.path.join(
            outfolder,
            "plots",
            name_short
        )

        os.makedirs(
            plot_dir,
            exist_ok=True
        )

        # ============================================================
        # Save main figure
        # ============================================================


        filename = (
            f"{name_short}_{type}_mom{mom}"
        )

        if normalise_norm:
            filename += "_norm"

        path = os.path.join(
            plot_dir,
            filename + ".pdf"
        )

        plt.savefig(
            path,
            pad_inches=0.0
        )

        plt.close(fig)

        print(
            f"Saved: {path}"
        )

    # ============================================================
    # Save separate colourbar for normalised maps
    # ============================================================

    if cbar:

        cbar_fig, cbar_ax = plt.subplots(
            figsize=(
                4,
                figsize * 7.5
            )
        )

        cb = plt.colorbar(
            plt.cm.ScalarMappable(
                norm=norm,
                cmap=cmap
            ),
            cax=cbar_ax,
            orientation="vertical"
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
                axis="y",
                which="major",
                labelsize=fontsize * 3.5,
                length=8,
                width=1.5,
                direction="out"
            )

        # --------------------------------------------------------
        # Colourbar label
        # --------------------------------------------------------

        if mom == 0:

            cb.set_label(
                r"Surface density ($M_{\odot}\,\mathrm{pc}^{-2}$)",
                fontsize=fontsize * 5,
                labelpad=20
            )

        elif mom == 1:

            cb.set_label(
                r"Velocity (km s$^{-1}$)",
                fontsize=fontsize * 5,
                labelpad=20
            )

        elif mom == 2:

            cb.set_label(
                r"Dispersion (km s$^{-1}$)",
                fontsize=fontsize * 5,
                labelpad=20
            )

        # --------------------------------------------------------
        # Colourbar filename
        # --------------------------------------------------------

        colourbar_dir = os.path.join(
            outfolder,
            "colourbars"
        )

        os.makedirs(
            colourbar_dir,
            exist_ok=True
        )

        colourbar_path = os.path.join(
            colourbar_dir,
            f"{name_short}_colourbar_mom{mom}.pdf"
        )

        plt.savefig(
            colourbar_path,
            bbox_inches="tight",
            pad_inches=0
        )

        plt.close(cbar_fig)

        print(
            f"Saved colourbar: {colourbar_path}"
        )


# =================================================================
# Loop through galaxies
# =================================================================

maps = {}

for name in sorted(os.listdir(outerdir)):

    maps_dir = os.path.join(
        outerdir,
        name,
        "maps"
    )

    if not os.path.isdir(maps_dir):
        continue

    # if name not in ['NGC4388', 'NGC5728', 'NGC6814']:
    #     continue

    print("\n" + "=" * 70)
    print(f"Processing {name}")
    print("=" * 70)

    maps[name] = {}

    # ------------------------------------------------------------
    # BAROLO parameters
    # ------------------------------------------------------------

    galaxy_outfolder = os.path.join(
        outerdir,
        name
    )

    try:

        barolo_params = read_barolo_parameters(
            galaxy_outfolder
        )

        xcen, ycen, pa, inc, vsys, vrot, disp, z, vrad = barolo_params


        print(
            f"BAROLO centre: ({xcen:.2f}, {ycen:.2f})"
        )

        print(
            f"BAROLO PA: {pa:.2f} deg"
        )

        print(
            f"BAROLO inclination: {inc:.2f} deg"
        )

        print(
            f"BAROLO Vsys: {vsys:.2f} km/s"
        )

    except Exception as e:

        print(
            f"Skipping {name}: could not read "
            f"rings_final2.txt"
        )

        print(
            f"Reason: {e}"
        )

        continue

    # ------------------------------------------------------------
    # Load BAROLO moment maps
    # ------------------------------------------------------------

    for n in [0, 1, 2]:

        for map_type, filename in [
            (
                "true",
                f"{name}_{n}mom.fits"
            ),
            (
                "local",
                f"{name}_local_{n}mom.fits"
            ),
        ]:

            filepath = os.path.join(
                maps_dir,
                filename
            )

            if not os.path.exists(filepath):

                print(
                    f"Missing: {filepath}"
                )

                continue

            with fits.open(filepath) as hdul:

                data = hdul[0].data.copy()
                header = hdul[0].header.copy()

                wcs = WCS(header)

                nddata = NDData(
                    data=data,
                    wcs=wcs
                )

            maps[name][
                (map_type, n)
            ] = nddata

            print(
                f"Loaded: {filepath}"
            )

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

    if not all(
        key in maps[name]
        for key in required_maps
    ):

        print(
            f"Skipping {name}: "
            f"not all BAROLO moment maps found."
        )

        continue

    # ------------------------------------------------------------
    # Load fits for second model if it is specified
    # ------------------------------------------------------------

    if runname2 is not None:

        maps_dir2 = os.path.join(
            outerdir2,
            name,
            "maps"
        )

        fit2_maps = {}

        for n in [0, 1, 2]:

            filepath = os.path.join(
                maps_dir2,
                f"{name}_local_{n}mom.fits"
            )

            if not os.path.exists(filepath):

                print(
                    f"Missing fit2 map: {filepath}"
                )

                fit2_maps = None
                break

            with fits.open(filepath) as hdul:

                data = hdul[0].data.copy()
                header = hdul[0].header.copy()

                wcs = WCS(header)

                fit2_maps[n] = NDData(
                    data=data,
                    wcs=wcs
                )

            print(
                f"Loaded fit2: {filepath}"
            )

        if fit2_maps is None:
            print(
                f"Skipping {name}: "
                f"not all fit2 maps found."
            )
            continue


    # ------------------------------------------------------------
    # Retrieve maps
    # ------------------------------------------------------------

    mom0_true = maps[name][
        ("true", 0)
    ]

    mom1_true = maps[name][
        ("true", 1)
    ]

    mom2_true = maps[name][
        ("true", 2)
    ]

    mom0_fit = maps[name][
        ("local", 0)
    ]

    mom1_fit = maps[name][
        ("local", 1)
    ]

    mom2_fit = maps[name][
        ("local", 2)
    ]


    if runname2 is not None:

        mom0_fit2 = fit2_maps[0]
        mom1_fit2 = fit2_maps[1]
        mom2_fit2 = fit2_maps[2]


    # Correct mom1 velocity to be centred on 0

    mom1_true = NDData(
    data=mom1_true.data - vsys,
    wcs=mom1_true.wcs
)
    mom1_fit = NDData(
    data=mom1_fit.data - vsys,
    wcs=mom1_fit.wcs
)
    if runname2 is not None:
        mom1_fit2 = NDData(
        data=mom1_fit2.data - vsys,
        wcs=mom1_fit2.wcs
    )

    # ------------------------------------------------------------
    # Residual maps
    # ------------------------------------------------------------

    if runname2 is None:

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

    else:

        mom0_res = NDData(
            data=mom0_fit2.data - mom0_fit.data,
            wcs=mom0_fit.wcs
        )

        mom1_res = NDData(
            data=mom1_fit2.data - mom1_fit.data,
            wcs=mom1_fit.wcs
        )

        mom2_res = NDData(
            data=mom2_fit2.data - mom2_fit.data,
            wcs=mom2_fit.wcs
        )
    # ============================================================
    # Record mom1 residual info
    # ============================================================
    if runname2 == None:
        output_csv = outerdir + f"/{runname}_fit{runn}.csv"

        non_circ = np.nansum(np.abs(mom1_res.data))

        df = pd.read_csv(output_csv)

        df.loc[df["name"] == name, "abs_mom1_residual (km/s)"] = non_circ

        df.to_csv(output_csv, index=False)

    # ============================================================
    # Plotting logic
    # ============================================================

    if runname2 is None:

        plot_maps0 = [
            mom0_true,
            mom0_fit,
            mom0_res
        ]

        plot_maps1 = [
            mom1_true,
            mom1_fit,
            mom1_res
        ]

        plot_maps2 = [
            mom2_true,
            mom2_fit,
            mom2_res
        ]


        plot_labels = [
            "true",
            "fit",
            "res"
        ]

    else:

        plot_maps0 = [
            mom0_fit,
            mom0_fit2,
            mom0_res
        ]

        plot_maps1 = [
            mom1_fit,
            mom1_fit2,
            mom1_res
        ]

        plot_maps2 = [
                        mom2_fit,
            mom2_fit2,
            mom2_res
        ]


        plot_labels = [
            f"{runname}",
            f"{runname2}",
            f"{runname2}_res"
        ]

    # ============================================================
    # Moment 0
    # ============================================================

    colourbar_list = []

    for image, map_type in zip(
    plot_maps0,
    plot_labels
):

        plot_moment_map(
            image,
            outputdir,
            name,
            map_type,
            R_kpc,
            norm_type="sqrt",
            mom=0,
            barolo_params=barolo_params
        )
        cb = True if map_type in ('res', f"{runname2}_res") else False

        plot_moment_map(
            image,
            outputdir,
            name,
            map_type,
            R_kpc,
            norm_type="sqrt",
            mom=0,
            normalise_norm=True,
            barolo_params=barolo_params, cbar = cb
        )

    # ============================================================
    # Moment 1
    # ============================================================

    colourbar_list = []

    for image, map_type in zip(
    plot_maps1,
    plot_labels
):
        plot_moment_map(
            image,
            outputdir,
            name,
            map_type,
            R_kpc,
            norm_type="linear",
            mom=1,
            barolo_params=barolo_params
        )
        cb = True if map_type in ('res', f"{runname2}_res") else False
        plot_moment_map(
            image,
            outputdir,
            name,
            map_type,
            R_kpc,
            norm_type="linear",
            mom=1,
            normalise_norm=True,
            barolo_params=barolo_params, cbar = cb
        
        )

    # ============================================================
    # Moment 2
    # ============================================================

    colourbar_list = []

    for image, map_type in zip(
    plot_maps2,
    plot_labels
):

        plot_moment_map(
            image,
            outputdir,
            name,
            map_type,
            R_kpc,
            norm_type="linear",
            mom=2,
            barolo_params=barolo_params
        )
        cb = True if map_type in ('res', f"{runname2}_res") else False
        plot_moment_map(
            image,
            outputdir,
            name,
            map_type,
            R_kpc,
            norm_type="linear",
            mom=2,
            normalise_norm=True,
            barolo_params=barolo_params, cbar = cb
        )

    # ------------------------------------------------------------
    # Clean up
    # ------------------------------------------------------------

    gc.collect()

    print(
        f"Finished {name}"
    )
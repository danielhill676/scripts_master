import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import itertools
import os
from IPython.display import display
from collections import defaultdict

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
})

def strip_units(colname):
    return colname.split("[")[0].strip()

def normalize_name(col):
    s = pd.Series(col.astype(str))
    return (
        s.str.replace('–', '-', regex=False)
        .str.replace('−', '-', regex=False)
        .str.strip()
        .str.upper()
    )

def get_errorbars(df, colname):
    """
    Return symmetric error bars if a matching column exists, else None.
    Expects '<colname>_err' column (or matching after stripping units).
    """
    err_col = f"{colname}_err"
    if err_col in df.columns:
        err = pd.to_numeric(df[err_col], errors="coerce")
        return err.values

    base = strip_units(colname)
    possible_err = [
        c for c in df.columns
        if strip_units(c).lower() == f"{base}_err".lower()
    ]

    if possible_err:
        err = pd.to_numeric(df[possible_err[0]], errors="coerce")
        return err.values

    return None


def connect_shared_galaxy(ax,
                          x1, y1, name1,
                          x2, y2, name2,
                          line_color='black',
                          alpha=0.5,
                          lw=1.0,
                          zorder=1):
    """
    Draw a line between two galaxies if they have the same name.
    """

    if (
        name1 == name2
        and np.isfinite(x1) and np.isfinite(y1)
        and np.isfinite(x2) and np.isfinite(y2)
    ):
        ax.annotate(
            "",
            xy=(float(x2), float(y2)),      # arrow tip
            xytext=(float(x1), float(y1)),  # arrow tail
            arrowprops=dict(
                arrowstyle="->",
                color=line_color,
                alpha=alpha,
                lw=lw,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=zorder,
        )

def plot_llama_triptych(
    x_column1, y_column1,
    x_column2, y_column2,
    x_column3, y_column3,
    wis_df=None,
    phangs_df=None,
    xerr_cols=None,
    yerr_cols=None,
    log_axes=None,
    figsize=9,
    colours_list=None, markers_list = None, sizes_list=None, connect=False
):

    if xerr_cols is None:
        xerr_cols = {}
    if yerr_cols is None:
        yerr_cols = {}
    if log_axes is None:
        log_axes = {}




    # --------------------------------------------------
    # Use fit_data as primary dataset if df is None
    # --------------------------------------------------
    path_wis = f"/Users/administrator/Astro/LLAMA/ALMA/comp_samples/m0_metrics_wis.csv"
    path_phangs = f"/Users/administrator/Astro/LLAMA/ALMA/comp_samples/m0_metrics_phangs.csv"
    fit_data_wis = pd.read_csv(path_wis)
    fit_data_phangs = pd.read_csv(path_phangs)

    # --------------------------------------------------
    # Prepare optional datasets
    # --------------------------------------------------
    optional_datasets = {}
    if wis_df is not None:
        wis_df = wis_df.copy()
        if 'Galaxy' not in wis_df.columns and 'Name' in wis_df.columns:
            wis_df['Galaxy'] = wis_df['Name']
        optional_datasets['WISDOM'] = wis_df
    if phangs_df is not None:
        phangs_df = phangs_df.copy()
        if 'Galaxy' not in phangs_df.columns and 'Name' in phangs_df.columns:
            phangs_df['Galaxy'] = phangs_df['Name']
        optional_datasets['PHANGS'] = phangs_df



    # --------------------------------------------------
    # XY extraction helper
    # --------------------------------------------------
    def extract_xy(df, xcol, ycol):
        # Get numeric values

        x = pd.to_numeric(df[xcol], errors="coerce")
        y = pd.to_numeric(df[ycol], errors="coerce")

        # Combine into DataFrame to filter NaNs consistently
        name_col = "name" if "name" in df.columns else "Name"
        tmp = pd.DataFrame({"x": x, "y": y, "name": df[name_col]})  
        tmp = tmp.dropna(subset=["x", "y"])

        x = tmp["x"].values
        y = tmp["y"].values
        name = tmp["name"].values

        return x, y,normalize_name(name)


    # --------------------------------------------------
    # Prepare panels
    # --------------------------------------------------
    datasets_for_plotting = {"WIS metric": fit_data_wis, "PHANGS metric": fit_data_phangs, **optional_datasets}
    panels = {}
    for key, xcol, ycol in [
        ("left", x_column1, y_column1),
        ("top", x_column2, y_column2),
        ("bottom", x_column3, y_column3)
    ]:
        panels[key] = {}
        for label, df in datasets_for_plotting.items():
            panels[key][label] = extract_xy(df, xcol, ycol)

        panels[key]["xcol"] = xcol
        panels[key]["ycol"] = ycol


    # --------------------------------------------------
    # Build LLAMA lookup: {panel: {name: (x, y)}}
    # --------------------------------------------------
    metrics_lookup = {}

    for key in ["left", "top", "bottom"]:

        # Combine AGN + inactive for this panel
        x_w, y_w, n_w = panels[key]["WIS metric"]
        x_p, y_p, n_p = panels[key]["PHANGS metric"]

        names = np.concatenate([n_w, n_p])
        xs = np.concatenate([x_w, x_p])
        ys = np.concatenate([y_w, y_p])

        # Build dict (last occurrence wins, fine for your case)
        metrics_lookup[key] = {
            str(name).strip().upper(): (float(x), float(y))
            for name, x, y in zip(names, xs, ys)
        }


        

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------
    fig = plt.figure(figsize=(figsize*1.5, figsize))
    gs = gridspec.GridSpec(
        2, 3,
        width_ratios=[1, 1, 0.25],
        height_ratios=[1, 1],
        wspace=0,
        hspace=0
    )

    ax_left = fig.add_subplot(gs[0, 0])
    ax_top = fig.add_subplot(gs[0, 1], sharey=ax_left)
    ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_top)
    axes = {"left": ax_left, "top": ax_top, "bottom": ax_bottom}

    # --------------------------------------------------
    # Markers and colors
    # --------------------------------------------------


    default_label_styles = {
        "WIS metric":      ("s", "deepskyblue", 6),
        "PHANGS metric":  ("o", "firebrick",   6),
        "WISDOM":          ("H", "indigo",      7),
        "PHANGS":          ("D", "orange",      5),
    }

    # Build label_styles
    label_styles = {}


    if "WIS metric" in datasets_for_plotting:
        label_styles["WIS metric"] = default_label_styles["WIS metric"]
    if "PHANGS metric" in datasets_for_plotting:
        label_styles["PHANGS metric"] = default_label_styles["PHANGS metric"]

    # Always add optional datasets
    for opt in ['Comparison-control', 'WISDOM', 'PHANGS']:
        if opt in datasets_for_plotting:
            label_styles[opt] = default_label_styles[opt]

    # Colour overrides
    if colours_list is not None:
        for label in label_styles:
            if label in colours_list:
                marker, _, size = label_styles[label]
                label_styles[label] = (marker, colours_list[label], size)

    # Marker overrides
    if markers_list is not None:
        for label in label_styles:
            if label in markers_list:
                _, color, size = label_styles[label]
                label_styles[label] = (markers_list[label], color, size)

    # Optional size overrides
    if sizes_list is not None:
        for label in label_styles:
            if label in sizes_list:
                marker, color, _ = label_styles[label]
                label_styles[label] = (marker, color, sizes_list[label])

    # --------------------------------------------------
    # Scatter + errorbars
    # --------------------------------------------------
    for key in panels:
        ax = axes[key]
        pdata = panels[key]
        for label, (marker, color, size) in label_styles.items():
            x, y, names = pdata[label]
            ax.errorbar(
                x, y,
                xerr=None, yerr=None,
                fmt=marker,
                markersize=size,
                capsize=2,
                elinewidth=1,
                alpha=0.85,
                color=color,
                label=label if key == "left" else None
            )


            # --------------------------------------------------
            # Annotate shared galaxies (both datasets)
            # --------------------------------------------------
            if label in ["WISDOM", "PHANGS"]: #### edited to print out all names

                for xi, yi, name in zip(x, y, names):

                    key_name = name

                    if key_name in metrics_lookup[key]:

                        # # --- annotate optional dataset point ---
                        # ax.text(
                        #     float(xi),
                        #     float(yi),
                        #     str(name),
                        #     fontsize=7,
                        #     color=color,
                        #     zorder=10
                        # )
                        
                        if connect:
                        #--- annotate corresponding LLAMA point ---
                            xL, yL = metrics_lookup[key][key_name]

                            ax.text(
                                xL,
                                yL,
                                str(name),
                                fontsize=7,
                                color="black",   # distinguish LLAMA labels
                                zorder=10
                            )
                        # print(
                        #     f"{key}: {pdata['xcol']} vs {pdata['ycol']} | "
                        #     f"{name} | "
                        #     f"davis22=({xi:.3f}, {yi:.3f}) | "
                        #     f"llama=({xL:.3f}, {yL:.3f}) | "
                        #     f"ratio=({xL/xi:.3f}, {yL/yi:.3f})"
                        #)
                        if connect:
                            connect_shared_galaxy(
                            ax,
                            xi, yi, name,
                            xL, yL, name,
                            line_color='grey',
                            alpha=0.5,
                            lw=1
                        )


            ax.grid(False)
    # --------------------------------------------------
    # Shared limits
    # --------------------------------------------------
    def combined_limits(arrays, log=False, pad=0.05,min_range=1.2):
        data = np.concatenate([a for a in arrays if len(a) > 0])
        if log:
            data = data[data > 0]
        if len(data) == 0:
            return None
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        delta = vmax - vmin

        # if delta < min_range:
        #     center = 0.5 * (vmin + vmax)
        #     vmin = center - min_range / 2
        #     vmax = center + min_range / 2
        #     delta = min_range

        return vmin - pad*delta, vmax + pad*delta

    # y1 == y2
    y_shared = combined_limits([
        panels["left"][label][1] for label in label_styles
    ] + [
        panels["top"][label][1] for label in label_styles
    ], log_axes.get("y_shared", False))
    if y_shared:
        ax_left.set_ylim(y_shared)
        ax_top.set_ylim(y_shared)

    # x2 == x3
    x_shared = combined_limits([
        panels["top"][label][0] for label in label_styles
    ] + [
        panels["bottom"][label][0] for label in label_styles
    ], log_axes.get("x_shared", False))
    if x_shared:
        ax_top.set_xlim(x_shared)
        ax_bottom.set_xlim(x_shared)

    # Independent limits
    ax_left.set_xlim(combined_limits([
        panels["left"][label][0] for label in label_styles
    ]))
    ax_bottom.set_ylim(combined_limits([
        panels["bottom"][label][1] for label in label_styles
    ]))

    # --------------------------------------------------
    # Log scaling
    # --------------------------------------------------
    for key in panels:
        ax = axes[key]
        xcol = panels[key]["xcol"]
        ycol = panels[key]["ycol"]

        if log_axes.get(xcol, False):
            ax.set_xscale("log")
        if log_axes.get(ycol, False):
            ax.set_yscale("log")


    # --------------------------------------------------
    # Labels + legend
    # --------------------------------------------------
    fs = 14  # adjust as needed

    ax_left.set_xlabel(axis_label_lookup.get(x_column1, x_column1), fontsize=fs)
    ax_left.set_ylabel(axis_label_lookup.get(y_column1, y_column1), fontsize=fs)

    ax_bottom.set_xlabel(axis_label_lookup.get(x_column3, x_column3), fontsize=fs)
    ax_bottom.set_ylabel(axis_label_lookup.get(y_column3, y_column3), fontsize=fs)


    ax_top.set_xlabel("")
    ax_top.set_ylabel("")
    ax_top.tick_params(labelleft=False, labelbottom=False)

    ax_empty = fig.add_subplot(gs[1, 0])
    ax_empty.axis("off")
    from matplotlib.lines import Line2D

    # Create legend handles manually without error bars

    legend_marker_sizes = {
    "WIS metric": 10,
    "PHANGS metric": 10,
    "WISDOM": 12,
    "PHANGS": 9,
}
    
    legend_handles = []
    for label, (marker, color, _) in label_styles.items():
        msize = legend_marker_sizes.get(label, 8)
        legend_handles.append(
            Line2D([0], [0], marker=marker, color='w', label=label,
                markerfacecolor=color, markersize=msize)
        )

    ax_empty.legend(handles=legend_handles, loc="center", fontsize=16, frameon=False)


    #plt.tight_layout()
    outfolder = f"/Users/administrator/Astro/LLAMA/ALMA/comp_samples/plots/"
    if not os.path.exists(os.path.dirname(outfolder)):
        os.makedirs(os.path.dirname(outfolder))
    if not connect:
        plt.savefig(f"{outfolder}/wisdom_CAS_triptych.png", dpi=300)
    else:
        plt.savefig(f"{outfolder}/wisdom_CAS_triptych_connect_shared.png", dpi=300)



axis_label_lookup = {
    "Resolution (pc)": "Resolution (pc)",
    "log LH (L⊙)": "$\log{L_H}$ (L$_\odot$)",
    "Smoothness": "Clumpiness",
    "clumping_factor": "Clumping Factor",
    "Smoothness_davis": "Clumpiness",
    "log LX": "$\log{L_{2-10}}$ (erg s$^{-1}$)",
    "total_mass (M_sun)": "Total Molecular Gas Mass ($M_\odot$)",
    "avg_mass_dens": "Average molecular Mass Surface Density ($M_\odot$kpc$^{-2}$)",
    "L'CO_JCMT (K km s pc2)": "ALMA L$'$ CO (K km s pc$^2$)",
    "L'CO_APEX (K km s pc2)": "ALMA L$'$ CO (K km s pc$^2$)",
    'log L′ CO': "Single-dish L$'$ CO (K km s pc$^2$)",
    "smoothness_espocito50_sig100": r"Clumpiness$_{50\,\mathrm{pc}}^{\sigma=100}$"
}


base_AGN = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/AGN"
base_inactive = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/inactive"
base_aux = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/aux"

wis_df = pd.read_csv("/Users/administrator/Astro/LLAMA/ALMA/comp_samples"+"/wis_new.csv")
phangs_df = pd.read_csv("/Users/administrator/Astro/LLAMA/ALMA/comp_samples"+"/phangs_new.csv")

exclude = ['NGC1375','NGC1315','NGC2775','MCG630']
exclude1 = ['NGC1375','NGC1315','NGC2775']
excludewisphagns= ['NGC1375','NGC1315','NGC2775','NGC5064_WIS','NGC1387_WIS']

################################################################### AGN vs inactive CAS triptych ###################################################################

m = 'strict'
r = 1.5


# plot_llama_triptych(
#     x_column1='Gini_davis', y_column1='Smoothness',
#     x_column2='Asymmetry_davis', y_column2='Smoothness',
#     x_column3='Asymmetry_davis', y_column3='Gini_davis',
# base_AGN=base_AGN, base_inactive=base_inactive,
#     log_axes={'x_shared': False, 'y_shared': False},
#     bins=10,
#     figsize=9, m = m, r = r, native_res=False, hist=False, exclude_names=['NGC1375','NGC1315','NGC2775','NGC5845','MCG630']
# )




# plot_llama_triptych(
#     x_column1='Gini', y_column1='Smoothness_davis',
#     x_column2='Asymmetry', y_column2='Smoothness_davis',
#     x_column3='Asymmetry', y_column3='Gini',
# base_AGN=base_AGN, base_inactive=base_inactive,
#     log_axes={'x_shared': False, 'y_shared': False},
#     bins=10,
#     figsize=9, m = m, r = r, native_res=True, hist=False, exclude_names=exclude1
# )

# ################################################################ AGN vs inactive CAS triptych wis phangs comparison ###################################################################

m = '120pc_flux90_strict'
r = 1.5



plot_llama_triptych(
    x_column1='Gini', y_column1='Smoothness_davis',
    x_column2='Asymmetry', y_column2='Smoothness_davis',
    x_column3='Asymmetry', y_column3='Gini',
    wis_df=wis_df,
    phangs_df=phangs_df,
    log_axes={'x_shared': False, 'y_shared': False},
    figsize=9,connect=True)

################################################################ comparison of mask and apertures ###################################################################

# plot_llama_triptych(
#     x_column1='Gini', y_column1='Smoothness_davis',
#     x_column2='Asymmetry', y_column2='Smoothness_davis',
#     x_column3='Asymmetry', y_column3='Gini',
# base_AGN=base_AGN, base_inactive=base_inactive,
#     log_axes={'x_shared': False, 'y_shared': False},
#     bins=10,
#     figsize=9, comb_llama=True, which_compare=[['strict','flux90_strict'],[0.3,1,1.5]], native_res=True, colours_list={
#   "\'strict\' mask and 0.6x0.6kpc aperture": "#0F2FFF",
#   "\'strict\' mask and 2.0x2.0kpc aperture": "#AF0FFF",
#   "\'strict\' mask and 3.0x3.0kpc aperture": "#FF0F9B",
#   "\'flux90_strict\' mask and 3.0x3.0kpc aperture": "#00EDED"
# }, markers_list={
#   "\'strict\' mask and 0.6x0.6kpc aperture": "D",
#   "\'strict\' mask and 2.0x2.0kpc aperture": "s",
#   "\'strict\' mask and 3.0x3.0kpc aperture": "o",
# "\'flux90_strict\' mask and 3.0x3.0kpc aperture": "P"
# }, exclude_names=exclude1,hist=False)


# plot_llama_triptych(
#     x_column1='Gini', y_column1='Smoothness_davis',
#     x_column2='Asymmetry', y_column2='Smoothness_davis',
#     x_column3='Asymmetry', y_column3='Gini',
# base_AGN=base_AGN, base_inactive=base_inactive,
#     log_axes={'x_shared': False, 'y_shared': False},
#     bins=10,
#     figsize=9, comb_llama=True, which_compare=[['strict','broad'],[1.5]], native_res=True, 
#     colours_list=  
#     {"\'strict\' mask and 3.0x3.0kpc aperture": "#008891",
#   "\'broad\' mask and 3.0x3.0kpc aperture": "#CC6900"
# }, markers_list={
#   "\'strict\' mask and 3.0x3.0kpc aperture": "D",
#   "\'broad\' mask and 3.0x3.0kpc aperture": "s",
# }, exclude_names=exclude1,hist=False)


# plot_llama_triptych(
#     x_column1='Gini', y_column1='Smoothness_davis',
#     x_column2='Asymmetry', y_column2='Smoothness_davis',
#     x_column3='Asymmetry', y_column3='Gini',
# base_AGN=base_AGN, base_inactive=base_inactive,
#     log_axes={'x_shared': False, 'y_shared': False},
#     bins=10,
#     figsize=9, comb_llama=True, which_compare=[['strict','120pc_strict'],[1.5]], native_res=True, colours_list={
#   "\'strict\' mask and 3.0x3.0kpc aperture": "#83CC90",
#   "\'120pc_strict\' mask and 3.0x3.0kpc aperture": "#00470E"
# }, markers_list={
#   "\'strict\' mask and 3.0x3.0kpc aperture": "D",
#   "\'120pc_strict\' mask and 3.0x3.0kpc aperture": "s"
# }, exclude_names=exclude1,hist=False)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools


def plot_llama_triptych(
    x_column1, y_column1,
    x_column2, y_column2,
    x_column3, y_column3,
    base_AGN,
    base_inactive,
    compare_masks,
    compare_radii,
    wis_df=None,
    phangs_df=None,
    axis_limits=None,
    colours_list=None,
    markers_list=None,
    hist=False,
    bins=20,
    figsize=6
):

    # --------------------------------------------------
    # helpers
    # --------------------------------------------------

    def get_errorbars(df, col):
        lo = col + "_err_lo"
        hi = col + "_err_hi"
        if lo in df.columns and hi in df.columns:
            return np.vstack([df[lo].values, df[hi].values])
        return None


    def extract_xy(df, xcol, ycol):

        sub = df[[xcol, ycol]].copy()
        sub[xcol] = pd.to_numeric(sub[xcol], errors="coerce")
        sub[ycol] = pd.to_numeric(sub[ycol], errors="coerce")
        sub = sub.dropna(subset=[xcol, ycol])

        x = sub[xcol].values
        y = sub[ycol].values

        xerr_full = get_errorbars(df, xcol)
        yerr_full = get_errorbars(df, ycol)

        xerr = xerr_full[:, sub.index] if xerr_full is not None else None
        yerr = yerr_full[:, sub.index] if yerr_full is not None else None

        return x, y, xerr, yerr


    # --------------------------------------------------
    # panel structure
    # --------------------------------------------------

    panels = {"left": {}, "top": {}, "bottom": {}}

    label_styles = {}

    default_colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    default_markers = ["o", "s", "D", "^", "v", "P"]


    # --------------------------------------------------
    # load datasets
    # --------------------------------------------------

    for i, (m_i, r_i) in enumerate(itertools.product(compare_masks, compare_radii)):

        path_AGN = f"{base_AGN}/gas_analysis_summary_{m_i}_{r_i}kpc.csv"
        path_inactive = f"{base_inactive}/gas_analysis_summary_{m_i}_{r_i}kpc.csv"

        dfA = pd.read_csv(path_AGN)
        dfI = pd.read_csv(path_inactive)

        combined_df = pd.concat([dfA, dfI], ignore_index=True)

        label = f"{m_i} mask and {r_i}kpc aperture"

        if colours_list:
            colour = colours_list[label]
        else:
            colour = default_colours[i % len(default_colours)]

        if markers_list:
            marker = markers_list[label]
        else:
            marker = default_markers[i % len(default_markers)]

        label_styles[label] = (marker, colour)

        panels["left"][label] = extract_xy(combined_df, x_column1, y_column1)
        panels["top"][label] = extract_xy(combined_df, x_column2, y_column2)
        panels["bottom"][label] = extract_xy(combined_df, x_column3, y_column3)


    # --------------------------------------------------
    # optional external samples
    # --------------------------------------------------

    for opt_label, opt_df, marker, colour in [
        ("WISDOM", wis_df, "^", "indigo"),
        ("PHANGS", phangs_df, "D", "orange"),
    ]:

        if opt_df is not None:

            label_styles[opt_label] = (marker, colour)

            panels["left"][opt_label] = extract_xy(opt_df, x_column1, y_column1)
            panels["top"][opt_label] = extract_xy(opt_df, x_column2, y_column2)
            panels["bottom"][opt_label] = extract_xy(opt_df, x_column3, y_column3)


    labels = list(label_styles.keys())


    # --------------------------------------------------
    # determine panel limits (square)
    # --------------------------------------------------

    def compute_limits(key):

        x = np.concatenate([panels[key][l][0] for l in labels])
        y = np.concatenate([panels[key][l][1] for l in labels])

        xmin, xmax = np.nanmin(x), np.nanmax(x)
        ymin, ymax = np.nanmin(y), np.nanmax(y)

        if axis_limits and key in axis_limits:
            lim = axis_limits[key]
            if "x" in lim: xmin, xmax = lim["x"]
            if "y" in lim: ymin, ymax = lim["y"]

        dx = xmax - xmin
        dy = ymax - ymin

        if dx > dy:
            c = 0.5 * (ymin + ymax)
            ymin, ymax = c - dx/2, c + dx/2
        else:
            c = 0.5 * (xmin + xmax)
            xmin, xmax = c - dy/2, c + dy/2

        return xmin, xmax, ymin, ymax


    limits = {k: compute_limits(k) for k in panels}


    # --------------------------------------------------
    # geometry
    # --------------------------------------------------

    dx_left = limits["left"][1] - limits["left"][0]
    dx_top = limits["top"][1] - limits["top"][0]

    dy_top = limits["top"][3] - limits["top"][2]
    dy_bottom = limits["bottom"][3] - limits["bottom"][2]

    total_w = dx_left + dx_top
    total_h = dy_top + dy_bottom

    # relative panel sizes
    w_left = dx_left / total_w
    w_top = dx_top / total_w

    h_top = dy_top / total_h
    h_bottom = dy_bottom / total_h

    # --------------------------------------------------
    # create figure
    # --------------------------------------------------

    fig = plt.figure(figsize=(figsize, figsize))

    # small margin so nothing touches frame
    margin = 0.08

    usable_w = 1 - margin*2
    usable_h = 1 - margin*2

    # scaled panel sizes
    w_left *= usable_w
    w_top *= usable_w

    h_top *= usable_h
    h_bottom *= usable_h

    # panel positions
    x_left = margin
    x_top = margin + w_left

    y_bottom = margin
    y_top = margin + h_bottom

    # --------------------------------------------------
    # create axes
    # --------------------------------------------------

    ax_left = fig.add_axes([x_left, y_top, w_left, h_top])
    ax_top = fig.add_axes([x_top, y_top, w_top, h_top])
    ax_bottom = fig.add_axes([x_top, y_bottom, w_top, h_bottom])


    axes = {"left": ax_left, "top": ax_top, "bottom": ax_bottom}

    # --------------------------------------------------
    # apply limits
    # --------------------------------------------------

    for key, ax in axes.items():

        xmin, xmax, ymin, ymax = limits[key]

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.set_aspect("equal", adjustable="box")

        ax.grid(False)


    # --------------------------------------------------
    # scatter plotting
    # --------------------------------------------------

    for label, (marker, colour) in label_styles.items():

        for key, ax in axes.items():

            x, y, xerr, yerr = panels[key][label]

            ax.errorbar(
                x, y,
                xerr=xerr, yerr=yerr,
                fmt=marker,
                markersize=7,
                capsize=2,
                alpha=0.2,
                color=colour,
                label=label if key == "left" else None
            )

            if len(x) > 0:

                ax.scatter(
                    np.nanmean(x),
                    np.nanmean(y),
                    marker=marker,
                    s=175,
                    color=colour,
                    edgecolor="black",
                    zorder=5
                )


    # --------------------------------------------------
    # histograms
    # --------------------------------------------------

    if hist:

        ax_hist_x = ax_top.inset_axes([0, 1.02, 1, 0.25])
        ax_hist_y1 = ax_top.inset_axes([1.02, 0, 0.25, 1])
        ax_hist_y2 = ax_bottom.inset_axes([1.02, 0, 0.25, 1])

        x_all = np.concatenate([panels["top"][l][0] for l in labels])
        bins_x = np.histogram_bin_edges(x_all, bins=bins)

        y_top_all = np.concatenate([panels["top"][l][1] for l in labels])
        bins_y1 = np.histogram_bin_edges(y_top_all, bins=bins)

        y_bot_all = np.concatenate([panels["bottom"][l][1] for l in labels])
        bins_y2 = np.histogram_bin_edges(y_bot_all, bins=bins)

        for label, (_, colour) in label_styles.items():

            ax_hist_x.hist(panels["top"][label][0], bins=bins_x, color=colour, alpha=0.4)
            ax_hist_y1.hist(panels["top"][label][1], bins=bins_y1,
                            orientation="horizontal", color=colour, alpha=0.4)
            ax_hist_y2.hist(panels["bottom"][label][1], bins=bins_y2,
                            orientation="horizontal", color=colour, alpha=0.4)

        ax_hist_x.axis("off")
        ax_hist_y1.axis("off")
        ax_hist_y2.axis("off")


    ax_left.legend(frameon=False)

    return fig, axes




base_AGN = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/AGN"
base_inactive = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/inactive"
base_aux = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/aux"

wis_df = pd.read_csv("/Users/administrator/Astro/LLAMA/ALMA/comp_samples"+"/wis_df.csv")
phangs_df = pd.read_csv("/Users/administrator/Astro/LLAMA/ALMA/comp_samples"+"/phangs_df.csv")

axis_limits={
    "left": {"x": (0.1,1.0), "y": (-0.1,3.5)},
    "top": {"x": (-0.1,2.1)},
    "bottom": {"y": (0.1,1)}
}

################################################################### AGN vs inactive CAS triptych ###################################################################

m = 'strict'
r = 1.5



# plot_llama_triptych(
#     x_column1='Gini', y_column1='Smoothness',
#     x_column2='Asymmetry', y_column2='Smoothness',
#     x_column3='Asymmetry', y_column3='Gini',
# base_AGN=base_AGN, base_inactive=base_inactive,
#     log_axes={'x_shared': False, 'y_shared': False},
#     bins=10,
#     figsize=9, m = m, r = r, native_res=True
# )

# plot_llama_triptych(
#     x_column1='Gini', y_column1='Smoothness',
#     x_column2='Asymmetry', y_column2='Smoothness',
#     x_column3='Asymmetry', y_column3='Gini',
# base_AGN=base_AGN, base_inactive=base_inactive,
#     log_axes={'x_shared': False, 'y_shared': False},
#     bins=10,
#     figsize=9, m = m, r = r, native_res=False
# )


# ################################################################ AGN vs inactive CAS triptych wis phangs comparison ###################################################################

# m = '120pc_flux90_strict'
# r = 1.5



# plot_llama_triptych(
#     x_column1='Gini', y_column1='Smoothness',
#     x_column2='Asymmetry', y_column2='Smoothness',
#     x_column3='Asymmetry', y_column3='Gini',
# base_AGN=base_AGN, base_inactive=base_inactive, base_aux=base_aux,
#     wis_df=wis_df,
#     phangs_df=phangs_df,
#     log_axes={'x_shared': False, 'y_shared': False},
#     bins=10,
#     figsize=9, m = m, r = r, comb_llama=True, native_res=True
#)

################################################################ comparison of mask and apertures ###################################################################

fig, axes = plot_llama_triptych(
    x_column1='Gini', y_column1='Smoothness',
    x_column2='Asymmetry', y_column2='Smoothness',
    x_column3='Asymmetry', y_column3='Gini',
    base_AGN=base_AGN,
    base_inactive=base_inactive,
    compare_masks=['strict'],
    compare_radii=[0.3,1,1.5],
    wis_df=None,
    phangs_df=None,
    axis_limits=axis_limits,
    colours_list={
  "strict mask and 0.3kpc aperture": "midnightblue",
  "strict mask and 1kpc aperture": "lime",
  "strict mask and 1.5kpc aperture": "orangered"
},  markers_list={
  "strict mask and 0.3kpc aperture": "D",
  "strict mask and 1kpc aperture": "s",
  "strict mask and 1.5kpc aperture": "p"
}, hist=False, bins=10, figsize=30
)

plt.show()

# plot_llama_triptych(
#     x_column1='Gini', y_column1='Smoothness',
#     x_column2='Asymmetry', y_column2='Smoothness',
#     x_column3='Asymmetry', y_column3='Gini',
# base_AGN=base_AGN, base_inactive=base_inactive,
#     log_axes={'x_shared': False, 'y_shared': False},
#     bins=10,
#     figsize=9, comb_llama=True, which_compare=[['strict','broad'],[1.5]], native_res=True, colours_list=  {"strict mask and 1.5kpc aperture": "darkcyan",
#   "broad mask and 1.5kpc aperture": "darkorange"
# }, markers_list={
#   "strict mask and 1.5kpc aperture": "D",
#   "broad mask and 1.5kpc aperture": "s",
# }, hist=False )

# plot_llama_triptych(
#     x_column1='Gini', y_column1='Smoothness',
#     x_column2='Asymmetry', y_column2='Smoothness',
#     x_column3='Asymmetry', y_column3='Gini',
# base_AGN=base_AGN, base_inactive=base_inactive,
#     log_axes={'x_shared': False, 'y_shared': False},
#     bins=10,
#     figsize=9, comb_llama=True, which_compare=[['strict','120pc_strict','120pc_flux90_strict'],[1.5]], native_res=True, colours_list={
#   "strict mask and 1.5kpc aperture": "seagreen",
#   "120pc_strict mask and 1.5kpc aperture": "mediumblue",
#   "120pc_flux90_strict mask and 1.5kpc aperture": "firebrick"
# }, markers_list={
#   "strict mask and 1.5kpc aperture": "D",
#   "120pc_strict mask and 1.5kpc aperture": "s",
#   "120pc_flux90_strict mask and 1.5kpc aperture": "p"
# }, hist=False )
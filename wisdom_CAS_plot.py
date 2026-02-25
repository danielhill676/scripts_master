import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

def strip_units(colname):
    return colname.split("[")[0].strip()

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

def plot_llama_triptych(
    x_column1, y_column1,
    x_column2, y_column2,
    x_column3, y_column3,
    df_AGN=None, fit_data_AGN=None,
    df_inactive=None, fit_data_inactive=None,
    fit_data_aux=None,
    wis_df=None,
    phangs_df=None,
    xerr_cols=None,
    yerr_cols=None,
    log_axes=None,
    bins=8,
    figsize=9,
    exclude_names=None,
    isolate_names=None,
    m='strict',
    r=1.5, comb_llama=False
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
    dfA = fit_data_AGN if fit_data_AGN is not None else df_AGN
    dfI = fit_data_inactive if fit_data_inactive is not None else df_inactive

    # --------------------------------------------------
    # Prepare optional datasets
    # --------------------------------------------------
    optional_datasets = {}
    if fit_data_aux is not None:
        optional_datasets['Comparison-pipeline'] = fit_data_aux
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
    # Name filtering (acts on Galaxy column)
    # --------------------------------------------------
    excluded = {"left": {"x": [], "y": []},
                "bottom": {"x": [], "y": []}}

    datasets_for_filtering = {"LLAMA AGN": dfA, "LLAMA inactive": dfI, **optional_datasets}

    if exclude_names is not None and isolate_names is None:
        exclude_norm = [n.strip().upper() for n in exclude_names]

        for label, df in datasets_for_filtering.items():
            mask = df["Galaxy"].str.strip().str.upper().isin(exclude_norm)

            # Store excluded values for left & bottom panels
            for panel_key, (xcol, ycol) in {
                "left": (x_column1, y_column1),
                "bottom": (x_column3, y_column3)
            }.items():
                excluded[panel_key]["x"].extend(
                    pd.to_numeric(df.loc[mask, xcol], errors="coerce").dropna().values
                )
                excluded[panel_key]["y"].extend(
                    pd.to_numeric(df.loc[mask, ycol], errors="coerce").dropna().values
                )

            # Remove excluded rows
            datasets_for_filtering[label] = df[~mask]

        dfA = datasets_for_filtering["LLAMA AGN"]
        dfI = datasets_for_filtering["LLAMA inactive"]
        for k in optional_datasets:
            optional_datasets[k] = datasets_for_filtering[k]

    # --------------------------------------------------
    # XY extraction helper
    # --------------------------------------------------
    def extract_xy(df, xcol, ycol):
        sub = df[[xcol, ycol]].copy()
        sub[xcol] = pd.to_numeric(sub[xcol], errors="coerce")
        sub[ycol] = pd.to_numeric(sub[ycol], errors="coerce")
        sub = sub.dropna(subset=[xcol, ycol])

        x = sub[xcol].values
        y = sub[ycol].values

        # Automatic errorbar extraction
        xerr_full = get_errorbars(df, xcol)
        yerr_full = get_errorbars(df, ycol)

        xerr = xerr_full[sub.index] if xerr_full is not None else None
        yerr = yerr_full[sub.index] if yerr_full is not None else None

        return x, y, xerr, yerr

    # --------------------------------------------------
    # Prepare panels
    # --------------------------------------------------
    datasets_for_plotting = {"LLAMA AGN": dfA, "LLAMA inactive": dfI, **optional_datasets}
    panels = {}
    for key, xcol, ycol in [
        ("left", x_column1, y_column1),
        ("top", x_column2, y_column2),
        ("bottom", x_column3, y_column3)
    ]:
        panels[key] = {}
        for label, df in datasets_for_plotting.items():
            panels[key][label] = extract_xy(df, xcol, ycol)

        # --- Add LLAMA combined if needed ---
        if comb_llama:
            combined_df = pd.concat([dfA, dfI], ignore_index=True)
            panels[key]['LLAMA combined'] = extract_xy(combined_df, xcol, ycol)

        panels[key]["xcol"] = xcol
        panels[key]["ycol"] = ycol

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

    ax_hist_y1 = fig.add_subplot(gs[0, 2], sharey=ax_top)
    ax_hist_y3 = fig.add_subplot(gs[1, 2], sharey=ax_bottom)
    ax_hist_x2 = ax_top.inset_axes([0, 1.02, 1, 0.25], sharex=ax_top)

    axes = {"left": ax_left, "top": ax_top, "bottom": ax_bottom}

    # --------------------------------------------------
    # Markers and colors
    # --------------------------------------------------


    default_label_styles = {
        "LLAMA AGN": ("s", 'red'),
        "LLAMA inactive": ("v", 'blue'),
        "Comparison-pipeline": ("*", "cyan"),
        "WISDOM": ("^", "indigo"),
        "PHANGS": ("D", "orange")
    }

    # Build label_styles
    label_styles = {}

    if comb_llama:
        label_styles['LLAMA combined'] = ('o', 'black')  # Combined AGN+inactive
    else:
        if 'LLAMA AGN' in datasets_for_plotting:
            label_styles['LLAMA AGN'] = default_label_styles['LLAMA AGN']
        if 'LLAMA inactive' in datasets_for_plotting:
            label_styles['LLAMA inactive'] = default_label_styles['LLAMA inactive']

    # Always add optional datasets
    for opt in ['Comparison-pipeline', 'WISDOM', 'PHANGS']:
        if opt in datasets_for_plotting:
            label_styles[opt] = default_label_styles[opt]



    # --------------------------------------------------
    # Scatter + errorbars
    # --------------------------------------------------
    for key in panels:
        ax = axes[key]
        pdata = panels[key]
        for label, (marker, color) in label_styles.items():
            x, y, xerr, yerr = pdata[label]
            ax.errorbar(
                x, y,
                xerr=xerr, yerr=yerr,
                fmt=marker, markersize=6,
                capsize=2, elinewidth=1,
                alpha=0.85, color=color,
                label=label if key == "left" else None
            )
        ax.grid(True)

    # --------------------------------------------------
    # Shared limits
    # --------------------------------------------------
    def combined_limits(arrays, log=False, pad=0.05):
        data = np.concatenate([a for a in arrays if len(a) > 0])
        if log:
            data = data[data > 0]
        if len(data) == 0:
            return None
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        delta = vmax - vmin
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
    # Excluded tick marks
    # --------------------------------------------------
    if exclude_names is not None:
        for panel_key, ax in [("left", ax_left), ("bottom", ax_bottom)]:
            xlower, xupper = ax.get_xlim()
            ylower, yupper = ax.get_ylim()

            for x_val in excluded[panel_key]["x"]:
                ax.plot([x_val], [ylower], marker='|', color='gray', markersize=10,
                        linestyle='None', alpha=0.7, clip_on=False)
            for y_val in excluded[panel_key]["y"]:
                ax.plot([xlower], [y_val], marker='_', color='gray', markersize=10,
                        linestyle='None', alpha=0.7, clip_on=False)

    # --------------------------------------------------
    # Histograms
    # --------------------------------------------------
    # Determine which datasets to include in histograms
    if comb_llama:
        hist_labels = ['LLAMA combined']
    else:
        hist_labels = ['LLAMA AGN', 'LLAMA inactive']
    
    include_opts = ['WISDOM', 'PHANGS']

    for opt in include_opts:
        if opt in panels['top']:
            hist_labels.append(opt)

    # y1 histogram
    y_all_top = np.concatenate([panels["top"][label][1] for label in hist_labels])
    bins_y1 = np.histogram_bin_edges(y_all_top, bins=bins)
    for label in hist_labels:
        ax_hist_y1.hist(panels["top"][label][1], bins=bins_y1, orientation='horizontal',
                        alpha=0.4, color=label_styles[label][1])
    ax_hist_y1.axis("off")

    # y3 histogram
    y_all_bottom = np.concatenate([panels["bottom"][label][1] for label in hist_labels])
    bins_y3 = np.histogram_bin_edges(y_all_bottom, bins=bins)
    for label in hist_labels:
        ax_hist_y3.hist(panels["bottom"][label][1], bins=bins_y3, orientation='horizontal',
                        alpha=0.4, color=label_styles[label][1])
    ax_hist_y3.axis("off")

    # x2 histogram
    x_all_top = np.concatenate([panels["top"][label][0] for label in hist_labels])
    bins_x2 = np.histogram_bin_edges(x_all_top, bins=bins)
    for label in hist_labels:
        ax_hist_x2.hist(panels["top"][label][0], bins=bins_x2, alpha=0.4, color=label_styles[label][1])
    ax_hist_x2.axis("off")


    # --------------------------------------------------
    # Labels + legend
    # --------------------------------------------------
    ax_left.set_xlabel(x_column1)
    ax_left.set_ylabel(y_column1)
    ax_bottom.set_xlabel(x_column3)
    ax_bottom.set_ylabel(y_column3)

    ax_top.set_xlabel("")
    ax_top.set_ylabel("")
    ax_top.tick_params(labelleft=False, labelbottom=False)

    ax_empty = fig.add_subplot(gs[1, 0])
    ax_empty.axis("off")
    from matplotlib.lines import Line2D

    # Create legend handles manually without error bars

    legend_marker_sizes = {
    "LLAMA AGN": 8,
    "LLAMA inactive": 12,
    "LLAMA combined": 8,
    "WISDOM": 8,
    "PHANGS": 8,
    "Comparison-pipeline": 14   # make it bigger
}
    
    legend_handles = []
    for label, (marker, color) in label_styles.items():
        msize = legend_marker_sizes.get(label, 8)
        legend_handles.append(
            Line2D([0], [0], marker=marker, color='w', label=label,
                markerfacecolor=color, markersize=msize)
        )

    ax_empty.legend(handles=legend_handles, loc="center", fontsize=18, frameon=False)


    plt.tight_layout()
    outfolder = f"/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/plots/{m}_{r}kpc"
    if m == '120pc_flux90_strict':
        outfolder = f"/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/plots/flux90_strict_{r}kpc"
    plt.savefig(f"{outfolder}/wisdom_CAS_triptych_{m}_{r}kpc.png", dpi=300)
    plt.show()

################################################################### AGN vs inactive CAS triptych ###################################################################

m = 'strict'
r = 1.5

base_AGN = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/AGN"
base_inactive = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/inactive"
base_aux = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/aux"

path_AGN = f"{base_AGN}/gas_analysis_summary_{m}_{r}kpc.csv"
path_inactive = f"{base_inactive}/gas_analysis_summary_{m}_{r}kpc.csv"
aux_path = f"{base_aux}/gas_analysis_summary_{m}_{r}kpc.csv"

fit_data_AGN = pd.read_csv(path_AGN)
fit_data_inactive = pd.read_csv(path_inactive)
fit_data_aux = pd.read_csv(aux_path)

wis_df = pd.read_csv("/Users/administrator/Astro/LLAMA/ALMA/comp_samples"+"/wis_df.csv")
phangs_df = pd.read_csv("/Users/administrator/Astro/LLAMA/ALMA/comp_samples"+"/phangs_df.csv")

plot_llama_triptych(
    x_column1='Gini', y_column1='Smoothness',
    x_column2='Asymmetry', y_column2='Smoothness',
    x_column3='Asymmetry', y_column3='Gini',
fit_data_AGN=fit_data_AGN,
fit_data_inactive=fit_data_inactive, 
    log_axes={'x_shared': False, 'y_shared': False},
    bins=10,
    figsize=9, m = m, r = r
)


################################################################ AGN vs inactive CAS triptych wis phangs comparison ###################################################################

m = '120pc_flux90_strict'
r = 1.5

base_AGN = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/AGN"
base_inactive = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/inactive"
base_aux = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/aux"

path_AGN = f"{base_AGN}/gas_analysis_summary_{m}_{r}kpc.csv"
path_inactive = f"{base_inactive}/gas_analysis_summary_{m}_{r}kpc.csv"
aux_path = f"{base_aux}/gas_analysis_summary_{m}_{r}kpc.csv"

fit_data_AGN = pd.read_csv(path_AGN)
fit_data_inactive = pd.read_csv(path_inactive)
fit_data_aux = pd.read_csv(aux_path)

wis_df = pd.read_csv("/Users/administrator/Astro/LLAMA/ALMA/comp_samples"+"/wis_df.csv")
phangs_df = pd.read_csv("/Users/administrator/Astro/LLAMA/ALMA/comp_samples"+"/phangs_df.csv")

plot_llama_triptych(
    x_column1='Gini', y_column1='Smoothness',
    x_column2='Asymmetry', y_column2='Smoothness',
    x_column3='Asymmetry', y_column3='Gini',
fit_data_AGN=fit_data_AGN,
fit_data_inactive=fit_data_inactive, fit_data_aux=fit_data_aux,
    wis_df=wis_df,
    phangs_df=phangs_df,
    log_axes={'x_shared': False, 'y_shared': False},
    bins=10,
    figsize=9, m = m, r = r, comb_llama=True
)
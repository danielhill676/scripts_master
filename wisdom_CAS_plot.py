import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import itertools
import os

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

def plot_llama_triptych(
    x_column1, y_column1,
    x_column2, y_column2,
    x_column3, y_column3,
    fit_data_AGN=None,
    base_AGN=None,
    base_inactive=None, base_aux=None,
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
    r=1.5, comb_llama=False, 
    which_compare = None, native_res=False, colours_list=None, markers_list = None, hist = True
):

    if xerr_cols is None:
        xerr_cols = {}
    if yerr_cols is None:
        yerr_cols = {}
    if log_axes is None:
        log_axes = {}


    if which_compare is None:

        # --------------------------------------------------
        # Use fit_data as primary dataset if df is None
        # --------------------------------------------------
        path_AGN = f"{base_AGN}/gas_analysis_summary_{m}_{r}kpc.csv"
        path_inactive = f"{base_inactive}/gas_analysis_summary_{m}_{r}kpc.csv"
        aux_path = f"{base_aux}/gas_analysis_summary_{m}_{r}kpc.csv"

        fit_data_AGN = pd.read_csv(path_AGN)
        fit_data_inactive = pd.read_csv(path_inactive)
        try:
            fit_data_aux = pd.read_csv(aux_path)
        except:
            fit_data_aux = None

        fit_data_AGN_res = fit_data_AGN.sort_values("Resolution (pc)", ascending=True)
        fit_data_inactive_res = fit_data_inactive.sort_values("Resolution (pc)", ascending=True)
        fit_data_aux_res = fit_data_aux.sort_values("Resolution (pc)", ascending=True) if fit_data_aux is not None else None

        fit_data_AGN_maxres = fit_data_AGN_res.drop_duplicates(subset="Galaxy", keep="last")
        fit_data_inactive_maxres = fit_data_inactive_res.drop_duplicates(subset="Galaxy", keep="last")
        fit_data_aux_maxres = fit_data_aux_res.drop_duplicates(subset="Galaxy", keep="last") if fit_data_aux_res is not None else None

        fit_data_AGN_minres = fit_data_AGN_res.drop_duplicates(subset="Galaxy", keep="first")
        fit_data_inactive_minres = fit_data_inactive_res.drop_duplicates(subset="Galaxy", keep="first")
        fit_data_aux_minres = fit_data_aux_res.drop_duplicates(subset="Galaxy", keep="first") if fit_data_aux_res is not None else None

        if native_res:

            dfA = fit_data_AGN_minres
            dfI = fit_data_inactive_minres
            df_aux = fit_data_aux_minres if fit_data_aux is not None else None
        
        else:

            dfA = fit_data_AGN_maxres
            dfI = fit_data_inactive_maxres
            df_aux = fit_data_aux_maxres if fit_data_aux is not None else None

        # --------------------------------------------------
        # Prepare optional datasets
        # --------------------------------------------------
        optional_datasets = {}
        if df_aux is not None:
            optional_datasets['Comparison-pipeline'] = df_aux
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
            # Build combined LLAMA name set
            # --------------------------------------------------
            llama_names = set(pd.concat([dfA, dfI])["Galaxy"].dropna().astype(str))

        # --------------------------------------------------
        # XY extraction helper
        # --------------------------------------------------
        def extract_xy(df, xcol, ycol):
            # Get numeric values
            x = pd.to_numeric(df[xcol], errors="coerce")
            y = pd.to_numeric(df[ycol], errors="coerce")

            # Get errorbars
            xerr = get_errorbars(df, xcol)
            yerr = get_errorbars(df, ycol)

            # Combine into DataFrame to filter NaNs consistently
            name_col = "Galaxy" if "Galaxy" in df.columns else "Name"
            tmp = pd.DataFrame({"x": x, "y": y, "name": df[name_col]})  
            if xerr is not None:
                tmp["xerr"] = xerr
            if yerr is not None:
                tmp["yerr"] = yerr

            tmp = tmp.dropna(subset=["x", "y"])

            x = tmp["x"].values
            y = tmp["y"].values
            xerr = tmp["xerr"].values if "xerr" in tmp.columns else None
            yerr = tmp["yerr"].values if "yerr" in tmp.columns else None
            name = tmp["name"].values

            return x, y, xerr, yerr, normalize_name(name)


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
        # Build LLAMA lookup: {panel: {name: (x, y)}}
        # --------------------------------------------------
        llama_lookup = {}

        for key in ["left", "top", "bottom"]:

            # Combine AGN + inactive for this panel
            xA, yA, _, _, nA = panels[key]["LLAMA AGN"]
            xI, yI, _, _, nI = panels[key]["LLAMA inactive"]

            names = np.concatenate([nA, nI])
            xs = np.concatenate([xA, xI])
            ys = np.concatenate([yA, yI])

            # Build dict (last occurrence wins, fine for your case)
            llama_lookup[key] = {
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

        if colours_list is not None:
            for label in label_styles:
                if label in colours_list:
                    label_styles[label] = (label_styles[label][0], colours_list[label])
        if markers_list is not None:
            for label in label_styles:
                if label in markers_list:
                    label_styles[label] = (markers_list[label], label_styles[label][1])

        # --------------------------------------------------
        # Scatter + errorbars
        # --------------------------------------------------
        for key in panels:
            ax = axes[key]
            pdata = panels[key]
            for label, (marker, color) in label_styles.items():
                x, y, xerr, yerr, names = pdata[label]
                ax.errorbar(
                    x, y,
                    xerr=xerr, yerr=yerr,
                    fmt=marker, markersize=6,
                    capsize=2, elinewidth=1,
                    alpha=0.85, color=color,
                    label=label if key == "left" else None
                )


                # --------------------------------------------------
                # Annotate shared galaxies (both datasets)
                # --------------------------------------------------
                if label in ["WISDOM", "PHANGS"]:

                    for xi, yi, name in zip(x, y, names):

                        key_name = name

                        if key_name in llama_lookup[key]:

                            # --- annotate optional dataset point ---
                            ax.text(
                                float(xi),
                                float(yi),
                                str(name),
                                fontsize=7,
                                color=color,
                                zorder=10
                            )

                            # --- annotate corresponding LLAMA point ---
                            xL, yL = llama_lookup[key][key_name]

                            ax.text(
                                xL,
                                yL,
                                str(name),
                                fontsize=7,
                                color="black",   # distinguish LLAMA labels
                                zorder=10
                            )

                ax.grid(False)
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

        ax_empty.legend(handles=legend_handles, loc="center", fontsize=16, frameon=False)


        plt.tight_layout()
        outfolder = f"/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/plots/{m}_{r}kpc"
        if m == '120pc_flux90_strict':
            outfolder = f"/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/plots/flux90_strict_{r}kpc"
        plt.savefig(f"{outfolder}/wisdom_CAS_triptych_{m}_{r}kpc_native{native_res}.png", dpi=300)
        plt.show()

###################################################################################################################################################################################################################################################################
# COMPARISON OF MASK/APERTURE
###################################################################################################################################################################################################################################################################
    else:

        compare_masks = which_compare[0]
        compare_radii = which_compare[1]

        if not comb_llama:
            raise ValueError("comb_llama must be True when using which_compare.")

        # Default pools (only used if not overridden)
        default_colours = ['red', 'blue', 'green', 'orange', 'purple', 'brown',
                        'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow']
        default_markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']

        # if colours_list is not None:
        #     colours = [colours_list.get(label, default) for label, default in zip(compare_masks, colours)]
        # if markers_list is not None:
        #     markers = [markers_list.get(label, default) for label, default in zip(compare_masks, markers)]

        # n = len(compare_masks) * len(compare_radii)
        # colours = colours[:n]
        # markers = markers[:n]

        label_styles = {}

        # --------------------------------------------------
        # Layout (create once)
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

        if hist:

            ax_hist_y1 = fig.add_subplot(gs[0, 2], sharey=ax_top)
            ax_hist_y3 = fig.add_subplot(gs[1, 2], sharey=ax_bottom)
            ax_hist_x2 = ax_top.inset_axes([0, 1.02, 1, 0.25], sharex=ax_top)

        axes = {"left": ax_left, "top": ax_top, "bottom": ax_bottom}

        # --------------------------------------------------
        # Prepare panels structure ONCE
        # --------------------------------------------------
        panels = {
            "left": {},
            "top": {},
            "bottom": {}
        }

        def extract_xy(df, xcol, ycol):
            sub = df[[xcol, ycol]].copy()
            sub[xcol] = pd.to_numeric(sub[xcol], errors="coerce")
            sub[ycol] = pd.to_numeric(sub[ycol], errors="coerce")
            sub = sub.dropna(subset=[xcol, ycol])

            x = sub[xcol].values
            y = sub[ycol].values

            xerr_full = get_errorbars(df, xcol)
            yerr_full = get_errorbars(df, ycol)

            xerr = xerr_full[sub.index] if xerr_full is not None else None
            yerr = yerr_full[sub.index] if yerr_full is not None else None

            return x, y, xerr, yerr

        # --------------------------------------------------
        # Load each mask/radius and store as its own label
        # --------------------------------------------------
        label_styles = {}

        for i, (m_i, r_i) in enumerate(itertools.product(compare_masks, compare_radii)):

            path_AGN = f"{base_AGN}/gas_analysis_summary_{m_i}_{r_i}kpc.csv"
            path_inactive = f"{base_inactive}/gas_analysis_summary_{m_i}_{r_i}kpc.csv"

            if not os.path.exists(path_AGN) or not os.path.exists(path_inactive):
                print(f"Warning: Missing data for mask '{m_i}' and radius '{r_i}kpc'. Skipping.")
                continue

            fit_data_AGN = pd.read_csv(path_AGN)
            fit_data_inactive = pd.read_csv(path_inactive)
            try:
                fit_data_aux = pd.read_csv(f"{base_aux}/gas_analysis_summary_{m_i}_{r_i}kpc.csv")
            except:
                fit_data_aux = None


            fit_data_AGN_res = fit_data_AGN.sort_values("Resolution (pc)", ascending=True)
            fit_data_inactive_res = fit_data_inactive.sort_values("Resolution (pc)", ascending=True)
            fit_data_aux_res = fit_data_aux.sort_values("Resolution (pc)", ascending=True) if fit_data_aux is not None else None

            fit_data_AGN_maxres = fit_data_AGN_res.drop_duplicates(subset="Galaxy", keep="last")
            fit_data_inactive_maxres = fit_data_inactive_res.drop_duplicates(subset="Galaxy", keep="last")
            fit_data_aux_maxres = fit_data_aux_res.drop_duplicates(subset="Galaxy", keep="last") if fit_data_aux_res is not None else None

            fit_data_AGN_minres = fit_data_AGN_res.drop_duplicates(subset="Galaxy", keep="first")
            fit_data_inactive_minres = fit_data_inactive_res.drop_duplicates(subset="Galaxy", keep="first")
            fit_data_aux_minres = fit_data_aux_res.drop_duplicates(subset="Galaxy", keep="first") if fit_data_aux_res is not None else None

            if native_res:

                dfA = fit_data_AGN_minres
                dfI = fit_data_inactive_minres
                df_aux = fit_data_aux_minres if fit_data_aux is not None else None
            
            else:

                dfA = fit_data_AGN_maxres
                dfI = fit_data_inactive_maxres
                df_aux = fit_data_aux_maxres if fit_data_aux is not None else None


            combined_df = pd.concat([dfA, dfI], ignore_index=True)
            label = f"\'{m_i}\' mask and {float(2*r_i)}x{float(2*r_i)}kpc aperture"

            # ---------------------------
            # Colour selection
            # ---------------------------
            if colours_list is not None:
                if label not in colours_list:
                    raise ValueError(
                        f"Missing colour for label '{label}' in colours_list"
                    )
                colour = colours_list[label]
            else:
                colour = default_colours[i % len(default_colours)]

            # ---------------------------
            # Marker selection
            # ---------------------------
            if markers_list is not None:
                if label not in markers_list:
                    raise ValueError(
                        f"Missing marker for label '{label}' in markers_list"
                    )
                marker = markers_list[label]
            else:
                marker = default_markers[i % len(default_markers)]

            label_styles[label] = (marker, colour)
            

            for key, xcol, ycol in [
                ("left", x_column1, y_column1),
                ("top", x_column2, y_column2),
                ("bottom", x_column3, y_column3)
            ]:
                panels[key][label] = extract_xy(combined_df, xcol, ycol)
                panels[key]["xcol"] = xcol
                panels[key]["ycol"] = ycol

        # --------------------------------------------------
        # Add PHANGS / WISDOM once
        # --------------------------------------------------
        for opt_label, opt_df, marker, color in [
            ("WISDOM", wis_df, "^", "indigo"),
            ("PHANGS", phangs_df, "D", "orange")
        ]:
            if opt_df is not None:
                label_styles[opt_label] = (marker, color)
                for key in panels:
                    xcol = panels[key]["xcol"]
                    ycol = panels[key]["ycol"]
                    panels[key][opt_label] = extract_xy(opt_df, xcol, ycol)

        # --------------------------------------------------
        # Scatter plotting
        # --------------------------------------------------
        for key in panels:
            ax = axes[key]
            pdata = panels[key]

            for label, (marker, color) in label_styles.items():
                x, y, xerr, yerr = pdata[label]
                ax.errorbar(
                    x, y,
                    xerr=xerr, yerr=yerr,
                    fmt=marker, markersize=8,
                    capsize=2, elinewidth=1,
                    alpha=0.15, color=color,
                    label=label if key == "left" else None
                )

                # mean marker
                if len(x) > 0:
                    ax.scatter(
                        np.nanmean(x),
                        np.nanmean(y),
                        marker=marker,
                        s=250,
                        color=color,
                        edgecolor='black',
                        zorder=5
                    )

            ax.grid(False)

        # --------------------------------------------------
        # Histograms (color matched)
        # --------------------------------------------------

        if hist:

            hist_labels = list(label_styles.keys())

            y_all_top = np.concatenate([panels["top"][l][1] for l in hist_labels])
            bins_y1 = np.histogram_bin_edges(y_all_top, bins=bins)

            for label in hist_labels:
                ax_hist_y1.hist(panels["top"][label][1], bins=bins_y1,
                            orientation='horizontal', alpha=0.4,
                            color=label_styles[label][1])
            ax_hist_y1.axis("off")

            y_all_bottom = np.concatenate([panels["bottom"][l][1] for l in hist_labels])
            bins_y3 = np.histogram_bin_edges(y_all_bottom, bins=bins)

            for label in hist_labels:
                ax_hist_y3.hist(panels["bottom"][label][1], bins=bins_y3,
                            orientation='horizontal', alpha=0.4,
                            color=label_styles[label][1])
            ax_hist_y3.axis("off")

            x_all_top = np.concatenate([panels["top"][l][0] for l in hist_labels])
            bins_x2 = np.histogram_bin_edges(x_all_top, bins=bins)

            for label in hist_labels:
                ax_hist_x2.hist(panels["top"][label][0], bins=bins_x2,
                            alpha=0.4, color=label_styles[label][1])
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
            "WISDOM": 8,
            "PHANGS": 8
        }

        # All other labels (mask/radius combinations) use a larger size
        for label in label_styles:
            if label not in legend_marker_sizes:
                legend_marker_sizes[label] = 18
        import textwrap
        wrap_width = 20 
        legend_handles = []
        for label, (marker, color) in label_styles.items():
            msize = legend_marker_sizes.get(label, 8)
            wrapped_label = "\n".join(textwrap.wrap(label, wrap_width))
            legend_handles.append(
                Line2D([0], [0], marker=marker, color='w', label=wrapped_label,
                    markerfacecolor=color, markersize=msize)
            )

        ax_empty.legend(handles=legend_handles, loc="center", fontsize=16, frameon=False)


        plt.tight_layout()
        outfolder = f"/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/plots/"
        plt.savefig(f"{outfolder}/wisdom_CAS_triptych_comparison{which_compare}native{native_res}.png", dpi=300)



base_AGN = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/AGN"
base_inactive = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/inactive"
base_aux = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/aux"

wis_df = pd.read_csv("/Users/administrator/Astro/LLAMA/ALMA/comp_samples"+"/wis_df.csv")
phangs_df = pd.read_csv("/Users/administrator/Astro/LLAMA/ALMA/comp_samples"+"/phangs_df.csv")

################################################################### AGN vs inactive CAS triptych ###################################################################

# m = 'strict'
# r = 1.5



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

m = '120pc_flux90_strict'
r = 1.5



plot_llama_triptych(
    x_column1='Gini', y_column1='Smoothness',
    x_column2='Asymmetry_altmask', y_column2='Smoothness',
    x_column3='Asymmetry_altmask', y_column3='Gini',
base_AGN=base_AGN, base_inactive=base_inactive, base_aux=base_aux,
    wis_df=wis_df,
    phangs_df=phangs_df,
    log_axes={'x_shared': False, 'y_shared': False},
    bins=10,
    figsize=9, m = m, r = r, comb_llama=True, native_res=True, hist=False
)

################################################################ comparison of mask and apertures ###################################################################

# plot_llama_triptych(
#     x_column1='Gini', y_column1='Smoothness',
#     x_column2='Asymmetry', y_column2='Smoothness',
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
# }, exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 5845'],hist=False)


# plot_llama_triptych(
#     x_column1='Gini', y_column1='Smoothness',
#     x_column2='Asymmetry', y_column2='Smoothness',
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
# }, exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 5845'],hist=False)


# plot_llama_triptych(
#     x_column1='Gini', y_column1='Smoothness',
#     x_column2='Asymmetry', y_column2='Smoothness',
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
# }, exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 5845'],hist=False)
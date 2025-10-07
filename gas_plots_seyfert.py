import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib import gridspec
#from IPython.display import display
import math
from scipy.stats import ks_2samp
from matplotlib.transforms import Bbox
import difflib
import re

def strip_units(colname: str) -> str:
    """Return column name without units in parentheses."""
    return re.sub(r"\s*\(.*?\)", "", colname).strip()

def get_errorbars(df, colname):
    """
    Return symmetric error bars if a matching column exists, else None.
    Expects a single '<colname>_err' column (or matching after stripping units).
    """
    # Try exact match
    err_col = f"{colname}_err"
    if err_col in df.columns:
        err = pd.to_numeric(df[err_col], errors="coerce")
        return err.values

    # Try stripping units
    base = strip_units(colname)
    possible_err = [c for c in df.columns if strip_units(c).lower() == f"{base}_err".lower()]

    if possible_err:
        err = pd.to_numeric(df[possible_err[0]], errors="coerce")
        return err.values

    return None

def is_categorical(series):
    """Return True if a Series is categorical or contains mostly non-numeric values."""
    # Ensure it's not empty
    if series.empty:
        return False

    # Check dtype directly
    if isinstance(series.dtype, pd.CategoricalDtype):
        return True

    # Try converting to numeric — if most fail, treat as categorical
    s = series.dropna().astype(str)
    numeric = pd.to_numeric(s, errors='coerce')
    non_numeric_fraction = numeric.isna().mean()

    if non_numeric_fraction > 0.5:
        print(f"Detected non-numeric string data, treating as categorical. Example values: {s.unique()[:5]}")
        return True
    return False

def plot_llama_property(x_column: str, y_column: str, AGN_data, inactive_data, agn_bol, inactive_bol, GB21, use_gb21=False, soloplot=None,exclude_names=None,logx=False,logy=False,background_image=None,manual_limits=None, legend_loc ='best' , truescale=False):
    """possible x_column: 'Distance (Mpc)', 'log LH (L⊙)', 'Hubble Stage', 'Axis Ratio', 'Bar'
       possible y_column: 'Smoothness', 'Asymmetry', 'Gini Coefficient', 'Sigma0', 'rs'"""

    # Load galaxy data
    df_AGN = pd.DataFrame(AGN_data)
    df_inactive = pd.DataFrame(inactive_data)
    common_columns = df_AGN.columns.intersection(df_inactive.columns)
    df_combined = pd.concat([df_AGN[common_columns], df_inactive[common_columns]], ignore_index=True)

    for df_name, df in [('AGN', df_AGN), ('inactive', df_inactive)]:
        if "Bar" in df.columns:
            df["Bar"] = (
                df["Bar"]
                .astype(str)
                .str.strip()
                .replace(["", "nan", "None", "NaN"], np.nan)
                .astype("category")
            )

    # Load LLAMA table
    llamatab = Table.read('/data/c3040163/llama/llama_main_properties.fits', format='fits')
    llama_df = llamatab.to_pandas()
    fit_data_AGN = pd.read_csv('/data/c3040163/llama/alma/gas_analysis_results/AGN/gas_analysis_summary.csv')
    fit_data_inactive = pd.read_csv('/data/c3040163/llama/alma/gas_analysis_results/inactive/gas_analysis_summary.csv')

    def normalize_name(col):
        # col can be a Pandas Series or Astropy Table Column
        s = pd.Series(col.astype(str))
        return s.str.replace('–', '-', regex=False) \
                .str.replace('−', '-', regex=False) \
                .str.strip() \
                .str.upper()

    # --- Example usage ---
    df_combined['Name_clean'] = normalize_name(df_combined['Name'])
    llamatab['name_clean'] = normalize_name(llamatab['name'])
    agn_bol['Name_clean'] = normalize_name(agn_bol['Name'])
    inactive_bol['Name_clean'] = normalize_name(inactive_bol['Name'])
    fit_data_AGN['Galaxy_clean'] = normalize_name(fit_data_AGN['Galaxy'])
    fit_data_inactive['Galaxy_clean'] = normalize_name(fit_data_inactive['Galaxy'])

    # --- Map name → id ---
    name_to_id = dict(zip(llamatab['name_clean'], llamatab['id']))
    df_combined['id'] = df_combined['Name_clean'].map(name_to_id)

    # --- Manual overrides for special cases ---
    manual_map = {
        "NGC 5128 (CEN A)": "NGC5128",
        "MCG-06-30-015": "MCG630",   # adjust ID if needed
        "NGC 1375": "NGC1375"
    }
    df_combined['id'] = df_combined['id'].fillna(df_combined['Name_clean'].map(manual_map))

    # Add derived LX column
    inactive_bol['log LAGN'] = pd.to_numeric(inactive_bol['log LAGN'], errors='coerce')
    inactive_bol['log LX'] = inactive_bol['log LAGN'].apply(
        lambda log_LAGN: math.log10((12.76 * (1 + (log_LAGN - math.log10(3.82e33)) / 12.15)**18.78) * 3.82e33)
    )

    # Merge with fit data
    merged_AGN = pd.merge(df_combined, fit_data_AGN, left_on='id', right_on='Galaxy_clean',how='right')
    merged_AGN = pd.merge(merged_AGN, agn_bol, left_on='Name_clean', right_on='Name_clean',how='left')
    merged_inactive = pd.merge(df_combined, fit_data_inactive, left_on='id', right_on='Galaxy_clean',how='right')
    merged_inactive = pd.merge(merged_inactive, inactive_bol, left_on='Name_clean', right_on='Name_clean',how='left')

    for df in [merged_AGN, merged_inactive]:
        if "Bar" in df.columns:
            df["Bar"] = df["Bar"].astype(str).str.strip().replace("", np.nan)
            df["Bar"] = df["Bar"].astype("category")

            # Determine if axes are categorical
    is_x_categorical = is_categorical(merged_AGN[x_column].dropna())
    is_y_categorical = is_categorical(merged_AGN[y_column].dropna())
    if is_x_categorical:
        print(f"Detected categorical x-axis: {x_column}")
    if is_y_categorical:
        print(f"Detected categorical y-axis: {y_column}")

    if not is_x_categorical and not is_y_categorical:

        # Clean AGN data
        merged_AGN[x_column] = pd.to_numeric(merged_AGN[x_column], errors='coerce')
        merged_AGN[y_column] = pd.to_numeric(merged_AGN[y_column], errors='coerce')
        merged_AGN_clean = merged_AGN.dropna(subset=[x_column, y_column])

        # Clean inactive data
        merged_inactive[x_column] = pd.to_numeric(merged_inactive[x_column], errors='coerce')
        merged_inactive[y_column] = pd.to_numeric(merged_inactive[y_column], errors='coerce')
        merged_inactive_clean = merged_inactive.dropna(subset=[x_column, y_column])

    else:
        merged_AGN_clean = merged_AGN.dropna(subset=[x_column, y_column])
        merged_inactive_clean = merged_inactive.dropna(subset=[x_column, y_column])

     # --- Exclude names here ---
    if exclude_names is not None:
        exclude_norm = [n.strip().upper() for n in exclude_names]

        merged_AGN_clean = merged_AGN_clean[
            ~merged_AGN_clean["Name"].str.strip().str.upper().isin(exclude_norm)
        ]
        merged_inactive_clean = merged_inactive_clean[
            ~merged_inactive_clean["Name"].str.strip().str.upper().isin(exclude_norm)
        ]
        agn_bol = agn_bol[~agn_bol["Name"].str.strip().str.upper().isin(exclude_norm)]
        inactive_bol = inactive_bol[~inactive_bol["Name"].str.strip().str.upper().isin(exclude_norm)]
        if use_gb21:
            GB21 = [row for row in GB21 if str(row["Name"]).strip().upper() not in exclude_norm]
    
    merged_AGN_clean = merged_AGN_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_column, y_column])
    merged_inactive_clean = merged_inactive_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_column, y_column])

    # Optional: Clean GB21 data
    if use_gb21:
        GB21_df = pd.DataFrame(GB21)
        GB21_df[x_column] = pd.to_numeric(GB21_df[x_column], errors='coerce')
        GB21_df[y_column] = pd.to_numeric(GB21_df[y_column], errors='coerce')
        GB21_clean = GB21_df.dropna(subset=[x_column, y_column])
        x_gb21 = GB21_clean[x_column]
        y_gb21 = GB21_clean[y_column]
        names_gb21 = GB21_clean["Name"].values

    # Extract values
    x_agn = merged_AGN_clean[x_column]
    y_agn = merged_AGN_clean[y_column]
    names_agn = merged_AGN_clean["Name_clean"].values
    xerr_agn = get_errorbars(merged_AGN_clean, x_column)
    yerr_agn = get_errorbars(merged_AGN_clean, y_column)

    x_inactive = merged_inactive_clean[x_column]
    y_inactive = merged_inactive_clean[y_column]
    names_inactive = merged_inactive_clean["Name_clean"].values
    xerr_inactive = get_errorbars(merged_inactive_clean, x_column)
    yerr_inactive = get_errorbars(merged_inactive_clean, y_column)


    if soloplot == 'AGN':
        if x_agn.empty or y_agn.empty:
            print("No valid AGN data to plot.")
            return
    elif soloplot == 'inactive':
        if x_inactive.empty or y_inactive.empty:
            print("No valid inactive data to plot.")
            return
    else:
        if x_agn.empty and x_inactive.empty and (not use_gb21 or x_gb21.empty):
            print("No valid X data to plot.")
            return
        if y_agn.empty and y_inactive.empty and (not use_gb21 or y_gb21.empty):
            print("No valid Y data to plot.")
            return

    # Set up figure

    if not is_x_categorical and not is_y_categorical:

        figsize = 8
        if truescale == True:
            if manual_limits is not None:
                xratio = (manual_limits[1] - manual_limits[0]) / (manual_limits[3] - manual_limits[2]) * 1.3
                yratio = (manual_limits[3] - manual_limits[2]) / (manual_limits[1] - manual_limits[0])
                fig = plt.figure(figsize=(figsize * xratio, figsize * yratio))
        else:
            fig = plt.figure(figsize=((figsize*1.1)*1.3, figsize*0.92))
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
        ax_scatter = fig.add_subplot(gs[0])
        ax_scatter.set_facecolor('none')

            # KS test only if both are present
        if soloplot is None:
            statistic, p_value = ks_2samp(y_inactive, y_agn)
            print(f'{y_column} stats')
            print(f"KS statistic: {statistic}")
            print(f"P-value: {p_value}")
            if p_value < 0.05:
                print("Distributions differ.")
            else:
                print("Distributions may be the same.")

            # Axis limits
        data_x = []
        data_y = []
        if soloplot in (None, 'AGN'):
            data_x.append(x_agn)
            data_y.append(y_agn)
        if soloplot in (None, 'inactive'):
            data_x.append(x_inactive)
            data_y.append(y_inactive)
        if soloplot is None and use_gb21:
            data_x.append(x_gb21)
            data_y.append(y_gb21)

        all_x = pd.concat(data_x)
        all_y = pd.concat(data_y)


        # set limits:

        if manual_limits is not None:
            xlower, xupper, ylower, yupper = manual_limits
            xlower, xupper, ylower, yupper = float(xlower), float(xupper), float(ylower), float(yupper)
        else:
            xspan = all_x.max() - all_x.min()
            yspan = all_y.max() - all_y.min()
            pad_x = (0.05 * xspan) if xspan > 0 else 0.1
            pad_y = (0.05 * yspan) if yspan > 0 else 0.05
            xlower = all_x.min() - pad_x
            xupper = all_x.max() + pad_x
            ylower = all_y.min() - pad_y
            yupper = all_y.max() + pad_y

        if logx:
            # prefer the manual lower if positive, otherwise pick 0.9 * min positive data
            if xlower <= 0:
                positives = all_x[all_x > 0]
                if positives.empty:
                    raise ValueError("Cannot use log x-axis: no positive x values available")
                xlower = float(positives.min()) * 0.9
                print("Adjusted x lower bound for log scale to", xlower)
        if logy:
            if ylower <= 0:
                positives = all_y[all_y > 0]
                if positives.empty:
                    raise ValueError("Cannot use log y-axis: no positive y values available")
                ylower = float(positives.min()) * 0.9
                print("Adjusted y lower bound for log scale to", ylower)

            # Background image:

        if background_image is not None:
            try:
                img = plt.imread(background_image)
                ax_img = fig.add_axes(ax_scatter.get_position(), zorder=-1)
                extent = [xlower, xupper, ylower, yupper]
                ax_img.imshow(img, extent=extent,
                origin='upper', alpha=1.0, aspect='auto', interpolation='none')
                ax_img.axis('off')

            except Exception as e:
                # avoid referencing ax_scatter before it's defined in other flows
                print(f"Could not load background image: {e}")

        # Plot scatter points
        if soloplot in (None, 'AGN'):
            ax_scatter.errorbar(
                x_agn, y_agn,
                xerr=xerr_agn, yerr=yerr_agn,
                fmt="s", color="red", label="LLAMA AGN", markersize=6,
                capsize=2, elinewidth=1, alpha=0.8
            )
            for x, y, name in zip(x_agn, y_agn, names_agn):
                ax_scatter.text(float(x + 0.005), float(y), name, fontsize=7, color='darkred', zorder=10)

        if soloplot in (None, 'inactive'):
            ax_scatter.errorbar(
                x_inactive, y_inactive,
                xerr=xerr_inactive, yerr=yerr_inactive,
                fmt="v", color="blue", label="LLAMA Inactive", markersize=6,
                capsize=2, elinewidth=1, alpha=0.8
            )
            for x, y, name in zip(x_inactive, y_inactive, names_inactive):
                ax_scatter.text(float(x), float(y), name, fontsize=7, color='navy', zorder=10)

        # apply scale and limits
        if logx:
            ax_scatter.set_xscale("log")
        if logy:
            ax_scatter.set_yscale("log")

        ax_scatter.set_xlim(xlower, xupper)
        ax_scatter.set_ylim(ylower, yupper)

        # Histogram bin edges

        y_for_bins = all_y[(all_y >= ylower) & (all_y <= yupper)]
        if y_for_bins.empty:
            y_for_bins = all_y
        bin_edges = np.histogram_bin_edges(y_for_bins, bins=7)

        # Scatter labels
        ax_scatter.set_xlabel(x_column)
        ax_scatter.set_ylabel(y_column)
        ax_scatter.grid(True)
        leg = ax_scatter.legend(loc=legend_loc)
        leg.set_zorder(30)
        ax_scatter.set_title(f'{y_column} vs {x_column}')

    # Histogram subplot
        ax_hist = fig.add_subplot(gs[1], sharey=ax_scatter)

        if soloplot in (None, 'AGN'):
            ax_hist.hist(y_agn, bins=bin_edges, orientation='horizontal', 
                        color='red', alpha=0.4, label='AGN')
            median_agn = np.median(y_agn)
            ax_hist.axhline(median_agn, color='red', linestyle='--')
            ax_hist.text(ax_hist.get_xlim()[1]*0.7, median_agn, 
                        f"{median_agn:.2f}", color='red', fontsize=8, va='center')

        if soloplot in (None, 'inactive'):
            ax_hist.hist(y_inactive, bins=bin_edges, orientation='horizontal', 
                        color='blue', alpha=0.4, label='Inactive')
            median_inactive = np.median(y_inactive)
            ax_hist.axhline(median_inactive, color='blue', linestyle='--')
            ax_hist.text(ax_hist.get_xlim()[1]*0.7, median_inactive, 
                        f"{median_inactive:.2f}", color='blue', fontsize=8, va='center')

        if soloplot is None and use_gb21:
            ax_hist.hist(y_gb21, bins=bin_edges, orientation='horizontal', 
                        color='green', alpha=0.4, label='GB21')
            median_gb21 = np.median(y_gb21)
            ax_hist.axhline(median_gb21, color='green', linestyle='--')
            ax_hist.text(ax_hist.get_xlim()[1]*0.7, median_gb21, 
                        f"{median_gb21:.2f}", color='green', fontsize=8, va='center')

        ax_hist.axis('off')

        # Save
        suffix = f"_{soloplot}" if soloplot else ""
        output_path = f'/data/c3040163/llama/alma/gas_analysis_results/Plots/{x_column}_vs_{y_column}{suffix}.png'
        plt.savefig(output_path)
        print(f"Saved plot to: {output_path}")

        diffs = []
        valid_pairs = 0

        for _, row in df_pairs.iterrows():
            agn_name = row["Active Galaxy"].strip()
            inactive_name = row["Inactive Galaxy"].strip()

            agn_val = merged_AGN.loc[merged_AGN["Name_clean"] == agn_name, y_column]
            inactive_val = merged_inactive.loc[merged_inactive["Name_clean"] == inactive_name, y_column]

            if not agn_val.empty and not inactive_val.empty:
                diff = float(agn_val.values[0]) - float(inactive_val.values[0])
                if diff == diff:  # Check for NaN
                    diffs.append(diff)
                    valid_pairs += 1

        diffs = np.array(diffs)
        diffs = diffs[np.isfinite(diffs)]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(diffs, bins=10, color="grey", alpha=0.4, edgecolor="black")
        ax.axvline(np.median(diffs), color="red", linestyle="--", label=f"Median = {np.median(diffs):.2f}")
        ax.axvline(np.percentile(diffs,25), color="blue", linestyle="--", label=f"Lower quartile = {np.percentile(diffs,25):.2f}")
        ax.axvline(np.percentile(diffs,75), color="blue", linestyle="--", label=f"Lower quartile = {np.percentile(diffs,75):.2f}")
        ax.axvline(0, color="black", linestyle="solid")  # reference line

        ax.set_xlabel(f"Δ {y_column} (AGN - Inactive)")
        ax.set_ylabel("Number of pairs")
        ax.set_title(f"Distribution of {y_column} differences across matched pairs\n(N={valid_pairs})")
        ax.legend()

        output_path = f'/data/c3040163/llama/alma/gas_analysis_results/Plots/pair_diffs/{y_column}_pair_differences.png'
        plt.savefig(output_path)
        print(f"Saved matched-pairs plot to: {output_path}")

    elif is_x_categorical or is_y_categorical:
        figsize = 8
        if truescale == True:
            if manual_limits is not None:
                xratio = (manual_limits[1] - manual_limits[0]) / (manual_limits[3] - manual_limits[2]) * 1.3
                yratio = (manual_limits[3] - manual_limits[2]) / (manual_limits[1] - manual_limits[0])
                fig = plt.figure(figsize=(figsize * xratio, figsize * yratio))
        else:
            fig = plt.figure(figsize=((figsize*1.1)*1.3, figsize*0.92))
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
        ax_bar = fig.add_subplot(gs[0])
        ax_bar.set_facecolor('none')
                    # Axis limits
        data_x = []
        data_y = []
        if soloplot in (None, 'AGN'):
            data_x.append(x_agn)
            data_y.append(y_agn)
        if soloplot in (None, 'inactive'):
            data_x.append(x_inactive)
            data_y.append(y_inactive)
        if soloplot is None and use_gb21:
            data_x.append(x_gb21)
            data_y.append(y_gb21)

        all_x = pd.concat(data_x)
        all_y = pd.concat(data_y)

        if is_x_categorical:
            cat_order = sorted(set(x_agn.dropna()) | set(x_inactive.dropna()))
            agn_median = []
            inactive_median = []
            agn_std = []
            inactive_std = []

            for cat in cat_order:
                agn_vals = merged_AGN_clean.loc[merged_AGN_clean[x_column] == cat, y_column].dropna()
                inact_vals = merged_inactive_clean.loc[merged_inactive_clean[x_column] == cat, y_column].dropna()

                if len(agn_vals) > 0:
                    agn_median.append(np.nanmedian(agn_vals))
                    agn_std.append(np.nanstd(agn_vals))
                else:
                    agn_median.append(np.nan)
                    agn_std.append(np.nan)

                if len(inact_vals) > 0:
                    inactive_median.append(np.nanmedian(inact_vals))
                    inactive_std.append(np.nanstd(inact_vals))
                else:
                    inactive_median.append(np.nan)
                    inactive_std.append(np.nan)

            x = np.arange(len(cat_order))
            width = 0.35

            ax_bar.bar(x - width/2, agn_median, width, yerr=agn_std, label="AGN", color="red", alpha=0.7, capsize=4)
            ax_bar.bar(x + width/2, inactive_median, width, yerr=inactive_std, label="Inactive", color="blue", alpha=0.7, capsize=4)
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(cat_order, rotation=45, ha="right")
            ax_bar.set_xlabel(x_column)
            ax_bar.set_ylabel(y_column)
            ax_bar.grid(True)
            leg = ax_bar.legend(loc=legend_loc)
            leg.set_zorder(30)
            ax_bar.set_title(f'{y_column} vs {x_column}')

            if manual_limits is not None:
                xlower, xupper, ylower, yupper = manual_limits
                xlower, xupper, ylower, yupper = float(xlower), float(xupper), float(ylower), float(yupper)
            else:
                yspan = all_y.max() - all_y.min()
                pad_y = (0.05 * yspan) if yspan > 0 else 0.05
                ylower = all_y.min() - pad_y
                yupper = all_y.max() + pad_y

                    # Histogram bin edges

            y_for_bins = all_y[(all_y >= ylower) & (all_y <= yupper)]
            if y_for_bins.empty:
                y_for_bins = all_y
            bin_edges = np.histogram_bin_edges(y_for_bins, bins=7)

            # Histogram subplot
            ax_hist = fig.add_subplot(gs[1], sharey=ax_bar)

            if soloplot in (None, 'AGN'):
                ax_hist.hist(y_agn, bins=bin_edges, orientation='horizontal', 
                            color='red', alpha=0.4, label='AGN')
                median_agn = np.median(y_agn)
                ax_hist.axhline(median_agn, color='red', linestyle='--')
                ax_hist.text(ax_hist.get_xlim()[1]*0.7, median_agn, 
                            f"{median_agn:.2f}", color='red', fontsize=8, va='center')

            if soloplot in (None, 'inactive'):
                ax_hist.hist(y_inactive, bins=bin_edges, orientation='horizontal', 
                            color='blue', alpha=0.4, label='Inactive')
                median_inactive = np.median(y_inactive)
                ax_hist.axhline(median_inactive, color='blue', linestyle='--')
                ax_hist.text(ax_hist.get_xlim()[1]*0.7, median_inactive, 
                            f"{median_inactive:.2f}", color='blue', fontsize=8, va='center')

            if soloplot is None and use_gb21:
                ax_hist.hist(y_gb21, bins=bin_edges, orientation='horizontal', 
                            color='green', alpha=0.4, label='GB21')
                median_gb21 = np.median(y_gb21)
                ax_hist.axhline(median_gb21, color='green', linestyle='--')
                ax_hist.text(ax_hist.get_xlim()[1]*0.7, median_gb21, 
                            f"{median_gb21:.2f}", color='green', fontsize=8, va='center')

            ax_hist.axis('off')

                    # Save
            suffix = f"_{soloplot}" if soloplot else ""
            output_path = f'/data/c3040163/llama/alma/gas_analysis_results/Plots/{x_column}_vs_{y_column}{suffix}.png'
            plt.savefig(output_path)
            print(f"Saved plot to: {output_path}")


        elif is_y_categorical:
            cat_order = sorted(set(y_agn.dropna()) | set(y_inactive.dropna()))
            agn_median = []
            inactive_median = []
            agn_std = []
            inactive_std = []

            for cat in cat_order:
                agn_vals = merged_AGN_clean.loc[merged_AGN_clean[y_column] == cat, x_column].dropna()
                inact_vals = merged_inactive_clean.loc[merged_inactive_clean[y_column] == cat, x_column].dropna()

                if len(agn_vals) > 0:
                    agn_median.append(np.nanmedian(agn_vals))
                    agn_std.append(np.nanstd(agn_vals))
                else:
                    agn_median.append(np.nan)
                    agn_std.append(np.nan)

                if len(inact_vals) > 0:
                    inactive_median.append(np.nanmedian(inact_vals))
                    inactive_std.append(np.nanstd(inact_vals))
                else:
                    inactive_median.append(np.nan)
                    inactive_std.append(np.nan)

            y = np.arange(len(cat_order))
            height = 0.35

            ax_bar.barh(y - height/2, agn_median, height, xerr=agn_std, label="AGN", color="red", alpha=0.7, capsize=4)
            ax_bar.barh(y + height/2, inactive_median, height, xerr=inactive_std, label="Inactive", color="blue", alpha=0.7, capsize=4)
            ax_bar.set_yticks(y)
            ax_bar.set_yticklabels(cat_order)
            ax_bar.set_xlabel(x_column)
            ax_bar.set_ylabel(y_column)
            ax_bar.grid(True)
            leg = ax_bar.legend(loc=legend_loc)
            leg.set_zorder(30)
            ax_bar.set_title(f'{y_column} vs {x_column}')
            suffix = f"_{soloplot}" if soloplot else ""
            output_path = f'/data/c3040163/llama/alma/gas_analysis_results/Plots/{x_column}_vs_{y_column}{suffix}.png'
            plt.savefig(output_path)
            print(f"Saved plot to: {output_path}")



AGN_data = [
    {"Name": "NGC 1365", "Distance (Mpc)": 18.0, "AGN Classification": "Sy 1.8a", "log LH (L⊙)": 10.58,
     "log L14–195 (erg s⁻¹)": 42.39, "log L2–10 (erg s⁻¹)": 41.83, "log L12 μm (erg s⁻¹)": 42.54,
     "log NH (cm⁻²)": "22.2d", "Hubble Stage": 3.0, "Axis Ratio": 0.55, "Bar": "B"},
    
    {"Name": "MCG-05-14-012", "Distance (Mpc)": 41.0, "AGN Classification": "Sy 1.0a", "log LH (L⊙)": 9.60,
     "log L14–195 (erg s⁻¹)": 42.60, "log L2–10 (erg s⁻¹)": 41.63, "log L12 μm (erg s⁻¹)": "L",
     "log NH (cm⁻²)": "≤21.9", "Hubble Stage": -1.0, "Axis Ratio": 0.86, "Bar": "L"},
    
    {"Name": "NGC 2110", "Distance (Mpc)": 34.0, "AGN Classification": "Sy 2 (1h)a", "log LH (L⊙)": 10.44,
     "log L14–195 (erg s⁻¹)": 43.64, "log L2–10 (erg s⁻¹)": 42.53, "log L12 μm (erg s⁻¹)": 43.04,
     "log NH (cm⁻²)": "23.0e", "Hubble Stage": -3.0, "Axis Ratio": 0.74, "Bar": "AB"},
    
    {"Name": "NGC 2992", "Distance (Mpc)": 36.0, "AGN Classification": "Sy 1.8a", "log LH (L⊙)": 10.31,
     "log L14–195 (erg s⁻¹)": 42.62, "log L2–10 (erg s⁻¹)": 42.05, "log L12 μm (erg s⁻¹)": 42.87,
     "log NH (cm⁻²)": 21.7, "Hubble Stage": 1.0, "Axis Ratio": 0.30, "Bar": "L"},
    
    {"Name": "MCG-05-23-016", "Distance (Mpc)": 35.0, "AGN Classification": "Sy 1.9a", "log LH (L⊙)": 9.94,
     "log L14–195 (erg s⁻¹)": 43.47, "log L2–10 (erg s⁻¹)": 43.11, "log L12 μm (erg s⁻¹)": 43.42,
     "log NH (cm⁻²)": 22.2, "Hubble Stage": -1.0, "Axis Ratio": 0.45, "Bar": "L"},
    
    {"Name": "NGC 3081", "Distance (Mpc)": 34.0, "AGN Classification": "Sy 2 (1h)a", "log LH (L⊙)": 10.15,
     "log L14–195 (erg s⁻¹)": 43.06, "log L2–10 (erg s⁻¹)": 41.54, "log L12 μm (erg s⁻¹)": 42.75,
     "log NH (cm⁻²)": 23.9, "Hubble Stage": 0.0, "Axis Ratio": 0.77, "Bar": "AB"},
    
    {"Name": "NGC 3783", "Distance (Mpc)": 38.0, "AGN Classification": "Sy 1.2a", "log LH (L⊙)": 10.29,
     "log L14–195 (erg s⁻¹)": 43.49, "log L2–10 (erg s⁻¹)": 43.12, "log L12 μm (erg s⁻¹)": 43.47,
     "log NH (cm⁻²)": 20.5, "Hubble Stage": 1.5, "Axis Ratio": 0.89, "Bar": "B"},
    
    {"Name": "NGC 4235", "Distance (Mpc)": 37.0, "AGN Classification": "Sy 1.2", "log LH (L⊙)": 10.43,
     "log L14–195 (erg s⁻¹)": 42.72, "log L2–10 (erg s⁻¹)": 41.66, "log L12 μm (erg s⁻¹)": 42.17,
     "log NH (cm⁻²)": 21.3, "Hubble Stage": 1.0, "Axis Ratio": 0.22, "Bar": "L"},
    
    {"Name": "NGC 4388", "Distance (Mpc)": 39.0, "AGN Classification": "Sy 2 (1h)", "log LH (L⊙)": 10.65,
     "log L14–195 (erg s⁻¹)": 43.70, "log L2–10 (erg s⁻¹)": 42.57, "log L12 μm (erg s⁻¹)": 42.93,
     "log NH (cm⁻²)": "23.5d", "Hubble Stage": 3.0, "Axis Ratio": 0.18, "Bar": "B"},
    
    {"Name": "NGC 4593", "Distance (Mpc)": 37.0, "AGN Classification": "Sy 1.0–1.2a", "log LH (L⊙)": 10.59,
     "log L14–195 (erg s⁻¹)": 43.16, "log L2–10 (erg s⁻¹)": 42.77, "log L12 μm (erg s⁻¹)": 42.97,
     "log NH (cm⁻²)": "≤19.2", "Hubble Stage": 3.0, "Axis Ratio": 0.74, "Bar": "B"},
    
    # ... continued for remaining 10 galaxies
    {"Name": "NGC 5128 (Cen A)", "Distance (Mpc)": 3.8, "AGN Classification": "Sy 2", "log LH (L⊙)": 10.22,
     "log L14–195 (erg s⁻¹)": 42.38, "log L2–10 (erg s⁻¹)": 41.50, "log L12 μm (erg s⁻¹)": 41.82,
     "log NH (cm⁻²)": "23.1d", "Hubble Stage": -2.0, "Axis Ratio": 0.78, "Bar": "L"},

    {"Name": "ESO 021-G004", "Distance (Mpc)": 39.0, "AGN Classification": "Syb", "log LH (L⊙)": 10.53,
     "log L14–195 (erg s⁻¹)": 42.49, "log L2–10 (erg s⁻¹)": 41.21, "log L12 μm (erg s⁻¹)": "L",
     "log NH (cm⁻²)": 23.8, "Hubble Stage": -0.4, "Axis Ratio": 0.45, "Bar": "L"},

    {"Name": "MCG-06–30-015", "Distance (Mpc)": 27.0, "AGN Classification": "Sy 1.2", "log LH (L⊙)": 9.59,
     "log L14–195 (erg s⁻¹)": 42.74, "log L2–10 (erg s⁻¹)": 42.51, "log L12 μm (erg s⁻¹)": 42.87,
     "log NH (cm⁻²)": 20.9, "Hubble Stage": -5.0, "Axis Ratio": 0.60, "Bar": "L"},

    {"Name": "NGC 5506", "Distance (Mpc)": 27.0, "AGN Classification": "Sy 2 (1i)", "log LH (L⊙)": 10.09,
     "log L14–195 (erg s⁻¹)": 43.32, "log L2–10 (erg s⁻¹)": 42.91, "log L12 μm (erg s⁻¹)": 43.28,
     "log NH (cm⁻²)": 22.4, "Hubble Stage": 1.0, "Axis Ratio": 0.23, "Bar": "L"},

    {"Name": "NGC 5728", "Distance (Mpc)": 39.0, "AGN Classification": "Sy 2", "log LH (L⊙)": 10.56,
     "log L14–195 (erg s⁻¹)": 43.21, "log L2–10 (erg s⁻¹)": 41.41, "log L12 μm (erg s⁻¹)": 42.35,
     "log NH (cm⁻²)": 24.2, "Hubble Stage": 1.0, "Axis Ratio": 0.57, "Bar": "B"},

    {"Name": "ESO 137-G034", "Distance (Mpc)": 35.0, "AGN Classification": "Sy 2", "log LH (L⊙)": 10.44,
     "log L14–195 (erg s⁻¹)": 42.62, "log L2–10 (erg s⁻¹)": 40.86, "log L12 μm (erg s⁻¹)": "L",
     "log NH (cm⁻²)": 24.3, "Hubble Stage": 0.0, "Axis Ratio": 0.79, "Bar": "AB"},

    {"Name": "NGC 6814", "Distance (Mpc)": 23.0, "AGN Classification": "Sy 1.5", "log LH (L⊙)": 10.31,
     "log L14–195 (erg s⁻¹)": 42.69, "log L2–10 (erg s⁻¹)": 42.17, "log L12 μm (erg s⁻¹)": 42.18,
     "log NH (cm⁻²)": 21.0, "Hubble Stage": 4.0, "Axis Ratio": 0.93, "Bar": "AB"},

    {"Name": "NGC 7172", "Distance (Mpc)": 37.0, "AGN Classification": "Sy 2 (1i)c", "log LH (L⊙)": 10.43,
     "log L14–195 (erg s⁻¹)": 43.45, "log L2–10 (erg s⁻¹)": 42.53, "log L12 μm (erg s⁻¹)": 42.88,
     "log NH (cm⁻²)": 22.9, "Hubble Stage": 1.4, "Axis Ratio": 0.56, "Bar": "L"},

    {"Name": "NGC 7213", "Distance (Mpc)": 25.0, "AGN Classification": "Sy 1", "log LH (L⊙)": 10.62,
     "log L14–195 (erg s⁻¹)": 42.50, "log L2–10 (erg s⁻¹)": 41.95, "log L12 μm (erg s⁻¹)": 42.58,
     "log NH (cm⁻²)": "≤20.4", "Hubble Stage": 1.0, "Axis Ratio": 0.90, "Bar": "L"},

    {"Name": "NGC 7582", "Distance (Mpc)": 22.0, "AGN Classification": "Sy 2 (1i)c", "log LH (L⊙)": 10.38,
     "log L14–195 (erg s⁻¹)": 42.67, "log L2–10 (erg s⁻¹)": 41.12, "log L12 μm (erg s⁻¹)": 42.81,
     "log NH (cm⁻²)": "24.2a", "Hubble Stage": 2.0, "Axis Ratio": 0.42, "Bar": "B"}
]

############# MATCHED PAIR DATA #############

# Inactive galaxies by match number (from the image; right-hand panel)
inactive_by_num = {
    1: "NGC 3351",
    2: "NGC 3175",
    3: "NGC 4254",
    4: "ESO 208-G021",
    5: "NGC 1079",
    6: "NGC 1947",
    7: "NGC 5921",
    8: "NGC 2775",
    9: "ESO 093-G003",
    10: "NGC 718",
    11: "NGC 3717",
    12: "NGC 5845",
    13: "NGC 7727",
    14: "IC 4653",
    15: "NGC 4260",
    16: "NGC 5037",
    17: "NGC 4224",
    18: "NGC 3749",
}

# Active galaxies with the exact numbers shown in their corners (from the image; left panel)
active_to_nums = {
    "NGC 1365": [7],
    "NGC 7582": [11, 17],
    "NGC 6814": [3],
    "NGC 4388": [11],
    "NGC 7213": [8],
    "MCG-06-30-015": [12],
    "NGC 5506": [2, 15, 16, 17, 18],
    "NGC 2110": [4, 6],
    "NGC 3081": [5, 9, 10],
    "MCG-05-28-016": [5, 14],
    "ESO 137-G034": [13],
    "NGC 2992": [2, 15, 16, 17, 18],
    "NGC 4235": [2, 16, 17, 18],
    "NGC 4593": [1, 8],
    "NGC 7172": [15, 16, 17],   # ← three pairs; includes NGC 4224 (17)
    "NGC 3783": [10],
    "ESO 021-G004": [17],
    "NGC 5728": [13, 17],
}

# Build all pairs (one row per link)
rows = []
for active, nums in active_to_nums.items():
    for n in nums:
        rows.append({
            "pair_id": n,
            "Active Galaxy": active,
            "Inactive Galaxy": inactive_by_num[n],
        })

df_pairs = pd.DataFrame(rows).sort_values(["pair_id", "Active Galaxy"]).reset_index(drop=True)



inactive_data = [
    {"Name": "NGC 718", "Distance (Mpc)": 23, "log LH (L⊙)": 9.89, "Hubble Stage": 1, "Axis Ratio": 0.87, "Bar": "AB"},
    {"Name": "NGC 1079", "Distance (Mpc)": 19, "log LH (L⊙)": 9.91, "Hubble Stage": 0, "Axis Ratio": 0.60, "Bar": "AB"},
    {"Name": "NGC 1315", "Distance (Mpc)": 21, "log LH (L⊙)": 9.47, "Hubble Stage": -1, "Axis Ratio": 0.89, "Bar": "B"},
    {"Name": "NGC 1947", "Distance (Mpc)": 19, "log LH (L⊙)": 10.07, "Hubble Stage": -3, "Axis Ratio": 0.87, "Bar": "L"},
    {"Name": "ESO 208-G021", "Distance (Mpc)": 17, "log LH (L⊙)": 10.88, "Hubble Stage": -3, "Axis Ratio": 0.70, "Bar": "AB"},
    {"Name": "NGC 2775", "Distance (Mpc)": 21, "log LH (L⊙)": 10.45, "Hubble Stage": 2, "Axis Ratio": 0.77, "Bar": "L"},
    {"Name": "NGC 3175", "Distance (Mpc)": 14, "log LH (L⊙)": 9.84, "Hubble Stage": 1, "Axis Ratio": 0.26, "Bar": "AB"},
    {"Name": "NGC 3351", "Distance (Mpc)": 11, "log LH (L⊙)": 10.07, "Hubble Stage": 3, "Axis Ratio": 0.93, "Bar": "B"},
    {"Name": "ESO 093-G003", "Distance (Mpc)": 22, "log LH (L⊙)": 9.86, "Hubble Stage": 0.3, "Axis Ratio": 0.60, "Bar": "AB"},
    {"Name": "NGC 3717", "Distance (Mpc)": 24, "log LH (L⊙)": 10.39, "Hubble Stage": 3, "Axis Ratio": 0.18, "Bar": "L"},
    {"Name": "NGC 3749", "Distance (Mpc)": 42, "log LH (L⊙)": 10.40, "Hubble Stage": 1, "Axis Ratio": 0.25, "Bar": "L"},
    {"Name": "NGC 4224", "Distance (Mpc)": 41, "log LH (L⊙)": 10.48, "Hubble Stage": 1, "Axis Ratio": 0.35, "Bar": "L"},
    {"Name": "NGC 4254", "Distance (Mpc)": 15, "log LH (L⊙)": 10.22, "Hubble Stage": 5, "Axis Ratio": 0.87, "Bar": "L"},
    {"Name": "NGC 4260", "Distance (Mpc)": 31, "log LH (L⊙)": 10.25, "Hubble Stage": 1, "Axis Ratio": 0.31, "Bar": "B"},
    {"Name": "NGC 5037", "Distance (Mpc)": 35, "log LH (L⊙)": 10.30, "Hubble Stage": 1, "Axis Ratio": 0.32, "Bar": "L"},
    {"Name": "NGC 5845", "Distance (Mpc)": 25, "log LH (L⊙)": 10.46, "Hubble Stage": -4.6, "Axis Ratio": 0.63, "Bar": "L"},
    {"Name": "NGC 5921", "Distance (Mpc)": 21, "log LH (L⊙)": 10.08, "Hubble Stage": 4, "Axis Ratio": 0.82, "Bar": "B"},
    {"Name": "IC 4653", "Distance (Mpc)": 26, "log LH (L⊙)": 9.48, "Hubble Stage": -0.5, "Axis Ratio": 0.63, "Bar": "B"},
    {"Name": "NGC 7727", "Distance (Mpc)": 26, "log LH (L⊙)": 10.41, "Hubble Stage": 1, "Axis Ratio": 0.74, "Bar": "AB"},
    {"Name": "NGC 1375", "Distance (Mpc)": 20, "log LH (L⊙)": 9.91, "Hubble Stage": -2.2, "Axis Ratio": 0.36, "Bar": "AB"}
]

agn_Rosario2018 = pd.DataFrame([
    {"Name": "ESO 021-G004", "log L′ CO": "8.083", "log LGAL": "43.45", "log LAGN": "42.30", "log LX": "42.19", "log NH": "23.8", "log LK,AGN": ""},
    {"Name": "ESO 137-G034", "log L′ CO": "7.820", "log LGAL": "43.68", "log LAGN": "43.23", "log LX": "42.34", "log NH": "24.3", "log LK,AGN": ""},
    {"Name": "MCG-05-23-016", "log L′ CO": "",      "log LGAL": "41.28", "log LAGN": "43.70", "log LX": "43.16", "log NH": "22.2", "log LK,AGN": "43.00"},
    {"Name": "MCG-06-30-015", "log L′ CO": "7.109", "log LGAL": "42.74", "log LAGN": "43.27", "log LX": "42.56", "log NH": "20.9", "log LK,AGN": "42.35"},
    {"Name": "NGC 1365",      "log L′ CO": "8.782", "log LGAL": "44.82", "log LAGN": "43.05", "log LX": "42.31", "log NH": "22.2", "log LK,AGN": "42.63"},
    {"Name": "NGC 2110",      "log L′ CO": "7.603", "log LGAL": "43.64", "log LAGN": "43.20", "log LX": "42.65", "log NH": "22.9", "log LK,AGN": "42.69"},
    {"Name": "NGC 2992",      "log L′ CO": "8.472", "log LGAL": "43.99", "log LAGN": "42.33", "log LX": "42.11", "log NH": "21.7", "log LK,AGN": "42.20"},
    {"Name": "NGC 3081",      "log L′ CO": "7.749", "log LGAL": "43.46", "log LAGN": "43.25", "log LX": "42.94", "log NH": "23.9", "log LK,AGN": "41.32"},
    {"Name": "NGC 3783",      "log L′ CO": "7.955", "log LGAL": "43.57", "log LAGN": "43.96", "log LX": "43.23", "log NH": "20.5", "log LK,AGN": "43.39"},
    {"Name": "NGC 4235",      "log L′ CO": "7.754", "log LGAL": "42.63", "log LAGN": "42.36", "log LX": "41.94", "log NH": "21.3", "log LK,AGN": ""},
    {"Name": "NGC 4388",      "log L′ CO": "8.151", "log LGAL": "44.25", "log LAGN": "43.51", "log LX": "43.20", "log NH": "23.5", "log LK,AGN": "41.75"},
    {"Name": "NGC 4593",      "log L′ CO": "8.146", "log LGAL": "43.71", "log LAGN": "43.20", "log LX": "42.91", "log NH": "",     "log LK,AGN": "42.42"},
    {"Name": "NGC 5506",      "log L′ CO": "7.874", "log LGAL": "43.62", "log LAGN": "43.61", "log LX": "43.10", "log NH": "22.4", "log LK,AGN": "43.09"},
    {"Name": "NGC 5728",      "log L′ CO": "8.531", "log LGAL": "44.23", "log LAGN": "42.22", "log LX": "43.14", "log NH": "24.1", "log LK,AGN": ""},
    {"Name": "NGC 6814",      "log L′ CO": "7.491", "log LGAL": "43.73", "log LAGN": "42.36", "log LX": "42.32", "log NH": "21.0", "log LK,AGN": "41.73"},
    {"Name": "NGC 7172",      "log L′ CO": "8.658", "log LGAL": "44.01", "log LAGN": "42.74", "log LX": "42.84", "log NH": "22.9", "log LK,AGN": "42.46"},
    {"Name": "NGC 7213",      "log L′ CO": "7.959", "log LGAL": "43.43", "log LAGN": "42.97", "log LX": "42.06", "log NH": "",     "log LK,AGN": "42.46"},
    {"Name": "NGC 7582",      "log L′ CO": "8.917", "log LGAL": "44.48", "log LAGN": "43.29", "log LX": "42.90", "log NH": "24.3", "log LK,AGN": "42.78"},
])

inactive_Rosario2018 = pd.DataFrame([
    {"Name": "ESO 093-G003", "log L′ CO": "8.233", "log LGAL": "43.91", "log LAGN": "40.8"},
    {"Name": "ESO 208-G021", "log L′ CO": "",      "log LGAL": "41.93", "log LAGN": "41.3"},
    {"Name": "IC 4653",      "log L′ CO": "7.642", "log LGAL": "43.15", "log LAGN": "41.7"},
    {"Name": "NGC 1079",     "log L′ CO": "6.610", "log LGAL": "42.51", "log LAGN": "39.6"},
    {"Name": "NGC 1947",     "log L′ CO": "7.888", "log LGAL": "42.72", "log LAGN": "39.5"},
    {"Name": "NGC 2775",     "log L′ CO": "",      "log LGAL": "43.32", "log LAGN": "40.6"},
    {"Name": "NGC 3175",     "log L′ CO": "8.040", "log LGAL": "43.68", "log LAGN": "40.0"},
    {"Name": "NGC 3351",     "log L′ CO": "7.911", "log LGAL": "43.55", "log LAGN": "39.4"},
    {"Name": "NGC 3717",     "log L′ CO": "8.489", "log LGAL": "43.96", "log LAGN": "40.7"},
    {"Name": "NGC 3749",     "log L′ CO": "8.691", "log LGAL": "43.86", "log LAGN": "40.7"},
    {"Name": "NGC 4224",     "log L′ CO": "8.086", "log LGAL": "42.93", "log LAGN": "42.0"},
    {"Name": "NGC 4254",     "log L′ CO": "8.134", "log LGAL": "44.84", "log LAGN": "40.5"},
    {"Name": "NGC 4260",     "log L′ CO": "",      "log LGAL": "42.35", "log LAGN": "40.8"},
    {"Name": "NGC 5037",     "log L′ CO": "8.328", "log LGAL": "43.06", "log LAGN": "40.1"},
    {"Name": "NGC 5845",     "log L′ CO": "",      "log LGAL": "41.69", "log LAGN": "40.9"},
    {"Name": "NGC 5921",     "log L′ CO": "7.960", "log LGAL": "43.40", "log LAGN": "40.7"},
    {"Name": "NGC 718",      "log L′ CO": "7.262", "log LGAL": "42.66", "log LAGN": "38.8"},
    {"Name": "NGC 7727",     "log L′ CO": "7.449", "log LGAL": "42.56", "log LAGN": "41.2"},
    {"Name": "NGC 1375",     "log L′ CO": "NaN", "log LGAL": "NaN", "log LAGN": "NaN"}
])


GB21_density = pd.DataFrame([
    {"Name": "NGC 613",   "log10Σtorus H2": 4.03, "log10Σ50pc H2": 3.36, "log10Σ200pc H2": 2.54},
    {"Name": "NGC 1068",  "log10Σtorus H2": 2.60, "log10Σ50pc H2": 2.72, "log10Σ200pc H2": 2.97},
    {"Name": "NGC 1326",  "log10Σtorus H2": 3.08, "log10Σ50pc H2": 2.54, "log10Σ200pc H2": 1.81},
    {"Name": "NGC 1365",  "log10Σtorus H2": 2.44, "log10Σ50pc H2": 2.25, "log10Σ200pc H2": 1.55},
    {"Name": "NGC 1566",  "log10Σtorus H2": 2.92, "log10Σ50pc H2": 2.62, "log10Σ200pc H2": 1.92},
    {"Name": "NGC 1672",  "log10Σtorus H2": 3.28, "log10Σ50pc H2": 3.19, "log10Σ200pc H2": 2.36},
    {"Name": "NGC 1808",  "log10Σtorus H2": 4.15, "log10Σ50pc H2": 2.71, "log10Σ200pc H2": 2.23},
    {"Name": "NGC 3227",  "log10Σtorus H2": 2.91, "log10Σ50pc H2": 2.91, "log10Σ200pc H2": 2.77},
    {"Name": "NGC 4388",  "log10Σtorus H2": 2.26, "log10Σ50pc H2": 2.00, "log10Σ200pc H2": 2.16},
    {"Name": "NGC 4941",  "log10Σtorus H2": 1.75, "log10Σ50pc H2": 2.24, "log10Σ200pc H2": 1.45},
    {"Name": "NGC 5506",  "log10Σtorus H2": 2.72, "log10Σ50pc H2": 2.45, "log10Σ200pc H2": 2.37},
    {"Name": "NGC 5643",  "log10Σtorus H2": 3.42, "log10Σ50pc H2": 3.20, "log10Σ200pc H2": 2.65},
    {"Name": "NGC 6300",  "log10Σtorus H2": 3.18, "log10Σ50pc H2": 3.13, "log10Σ200pc H2": 2.45},
    {"Name": "NGC 6814",  "log10Σtorus H2": 1.80, "log10Σ50pc H2": 1.70, "log10Σ200pc H2": 0.99},
    {"Name": "NGC 7213",  "log10Σtorus H2": 1.61, "log10Σ50pc H2": 1.21, "log10Σ200pc H2": 0.66},
    {"Name": "NGC 7314",  "log10Σtorus H2": 2.17, "log10Σ50pc H2": 2.25, "log10Σ200pc H2": 1.56},
    {"Name": "NGC 7465",  "log10Σtorus H2": 2.58, "log10Σ50pc H2": 2.68, "log10Σ200pc H2": 2.10},
    {"Name": "NGC 7582",  "log10Σtorus H2": 2.62, "log10Σ50pc H2": 2.80, "log10Σ200pc H2": 2.96},
])

GB21_density["log LX"] = [
    41.3, 42.8, 39.8, 42.1, 40.9, 39.2, 39.9, 42.4, 42.5,
    41.4, 43.0, 42.4, 41.7, 42.2, 41.9, 42.2, 41.9, 43.5
]

GB21_density['Concentration'] = [a-b for a,b in zip(GB21_density["log10Σ50pc H2"], GB21_density["log10Σ200pc H2"])]



 #   """posible x_column: 'Distance (Mpc)', 'log LH (L⊙)', 'Hubble Stage', 'Axis Ratio', 'Bar'
 #      posible y_column: 'Smoothness', 'Asymmetry', 'Gini', 'Sigma0', 'rs'"""


############# CAS with stellar mass #############

plot_llama_property('log LH (L⊙)', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,use_gb21=False)
plot_llama_property('log LH (L⊙)', 'Asymmetry', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)
plot_llama_property('log LH (L⊙)', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)
plot_llama_property('log LH (L⊙)','Concentration',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)
plot_llama_property('log LH (L⊙)','clumping_factor',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)#,exclude_names=['NGC 2775','NGC 4260','ESO 208-G021','NGC 5845','NGC 2992','NGC 1079','NGC 4388'])

############# CAS with Hubble Stage #############

plot_llama_property('Hubble Stage', 'Concentration', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)
plot_llama_property('Hubble Stage', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)
plot_llama_property('Hubble Stage', 'Asymmetry', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)
plot_llama_property('Hubble Stage', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)
plot_llama_property('Hubble Stage','clumping_factor',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)#,exclude_names=['NGC 2775','NGC 4260','ESO 208-G021','NGC 5845','NGC 2992','NGC 1079','NGC 4388'])

############# CAS with X-ray luminosity #############

plot_llama_property('log LX', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False,soloplot='AGN')
plot_llama_property('log LX', 'Asymmetry', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False,soloplot='AGN')
plot_llama_property('log LX', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False,soloplot='AGN')
plot_llama_property('log LX','Concentration',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,use_gb21=True,soloplot='AGN')
plot_llama_property('log LX','clumping_factor',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,use_gb21=False,soloplot='AGN')#,exclude_names=['NGC 2775','NGC 4260','ESO 208-G021','NGC 5845','NGC 2992','NGC 1079','NGC 4388'])


plot_llama_property('log LX', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False,soloplot='inactive')
plot_llama_property('log LX', 'Asymmetry', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False,soloplot='inactive')
plot_llama_property('log LX', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False,soloplot='inactive')
plot_llama_property('log LX','Concentration',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,use_gb21=True,soloplot='inactive')
plot_llama_property('log LX','clumping_factor',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,use_gb21=False,soloplot='inactive')#,exclude_names=['NGC 2775','NGC 4260','ESO 208-G021','NGC 5845','NGC 2992','NGC 1079','NGC 4388'])


plot_llama_property('log LX','Concentration',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,use_gb21=True)

############## CAS with eachother #############

plot_llama_property('Gini', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)
plot_llama_property('Asymmetry', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)
plot_llama_property('Asymmetry', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)
plot_llama_property('Gini', 'clumping_factor', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)#,exclude_names=['NGC 2775','NGC 4260','ESO 208-G021','NGC 5845','NGC 2992','NGC 1079','NGC 4388'])
plot_llama_property('clumping_factor', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)
plot_llama_property('Asymmetry', 'clumping_factor', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)

############## CAS with concentration #############

plot_llama_property('Gini', 'Concentration', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)
plot_llama_property('Asymmetry', 'Concentration', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)
plot_llama_property('Smoothness', 'Concentration', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)
plot_llama_property('clumping_factor', 'Concentration', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)#,exclude_names=['NGC 2775','NGC 4260','ESO 208-G021','NGC 5845','NGC 2992','NGC 1079','NGC 4388'])

############### CAS with resolution #############

plot_llama_property('Resolution (pc)', 'Concentration', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)
plot_llama_property('Resolution (pc)', 'Asymmetry', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)
plot_llama_property('Resolution (pc)', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)
plot_llama_property('Resolution (pc)', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)
plot_llama_property('Resolution (pc)', 'clumping_factor', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False,logx=False,logy=True,background_image='/data/c3040163/llama/alma/gas_analysis_results/Leroy2013_plots/Clumping.png',manual_limits=[0,500,1,200],legend_loc='center right')#,exclude_names=['NGC 2775','NGC 4260','ESO 208-G021','NGC 5845','NGC 2992','NGC 1079','NGC 4388'])

plot_llama_property('Hubble Stage', 'Hubble Stage', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)

plot_llama_property('total_mass (M_sun)', 'Concentration', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)
plot_llama_property('total_mass (M_sun)', 'Asymmetry', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)
plot_llama_property('total_mass (M_sun)', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)
plot_llama_property('total_mass (M_sun)', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)
plot_llama_property('total_mass (M_sun)', 'clumping_factor', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)#,exclude_names=['NGC 2775','NGC 4260','ESO 208-G021','NGC 5845','NGC 2992','NGC 1079','NGC 4388'])

plot_llama_property('area_weighted_sd','mass_weighted_sd',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False,logx=True,logy=True,background_image='/data/c3040163/llama/alma/gas_analysis_results/Leroy2013_plots/Sigma.png',manual_limits=[0.5,5000,0.5,5000], truescale=True)

plot_llama_property('Bar', 'Concentration', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)
plot_llama_property('Bar', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)
plot_llama_property('Bar', 'Asymmetry', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)
plot_llama_property('Bar', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)
plot_llama_property('Bar','clumping_factor',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,False)

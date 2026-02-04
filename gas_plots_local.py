import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib import gridspec
import math
from scipy.stats import ks_2samp
from matplotlib.transforms import Bbox
import difflib
import re
import os
from astroquery.vizier import Vizier
from astroquery.exceptions import RemoteServiceError
import requests
import time
from IPython.display import display
from astropy.table import join
from astropy.table import MaskedColumn
pd.set_option('future.no_silent_downcasting', True)

stats_rows = []

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

def normalize_name(col):
    s = pd.Series(col.astype(str))
    return (
        s.str.replace('–', '-', regex=False)
        .str.replace('−', '-', regex=False)
        .str.strip()
        .str.upper()
    )

def plot_llama_property(x_column: str, y_column: str, AGN_data, inactive_data, agn_bol, inactive_bol, use_gb21=False, soloplot=None, exclude_names=None, logx=False, logy=False,  #see archived_comp_samp_build for rebuilding PHANGS WIS SIM GB21
                        background_image=None, manual_limits=None, square=False, best_fit=False, legend_loc='best', truescale=False, use_wis=False, use_phangs=False, use_sim=False,comb_llama=False,plotshared=True,rebin=None,mask=None,R_kpc=1,compare=False, which_compare=None, use_aux=False, use_cont = False,nativex=False,nativey=False):
    """possible x_column: '"Distance (Mpc)"', 'log LH (L⊙)', 'Hubble Stage', 'Axis Ratio', 'Bar'
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
    llamatab = Table.read('/Users/administrator/Astro/LLAMA/llama_main_properties.fits', format='fits')

    base_AGN = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/AGN"
    base_inactive = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/inactive"
    base_aux = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/aux"
    base_cont_AGN = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/cont_analysis_results/AGN"
    base_cont_inactive = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/cont_analysis_results/inactive"

    default_AGN = f"{base_AGN}/gas_analysis_summary.csv"
    default_inactive = f"{base_inactive}/gas_analysis_summary.csv"
    default_aux = f"{base_aux}/gas_analysis_summary.csv"

    ################## comparing different masks/radii ########################

    if (compare and use_phangs) or (compare and use_sim) or (compare and use_wis) or (compare and use_gb21) or (compare and not comb_llama):
        print('compare not compatible with use_phangs/use_sim/use_wis')
        # Handle comparison with other datasets
        pass

    if compare and comb_llama and not use_phangs and not use_sim and not use_wis and not use_gb21:
        import itertools



        compare_masks = ['strict', 'broad','flux90_strict','flux90_broad','120pc_flux90_strict', '120pc_flux90_broad','120pc_strict', '120pc_broad']
        compare_radii = [0.3, 1, 1.5]
        colours = ['red', 'blue', 'green', 'orange', 'purple', 'brown','pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'teal', 'navy', 'maroon', 'lime', 'coral', 'gold', 'indigo', 'violet', 'turquoise', 'salmon', 'plum', 'orchid']
        markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>', 'H', '+', '1', '2', '3', '4', '|', '_', '.', ',', '8', 'p', 'h']
        
        if which_compare is not None:
            compare_masks = which_compare[0]
            compare_radii = which_compare[1]
            n = len(compare_masks) * len(compare_radii)
            colours = colours[0:n]
            markers = markers[0:n]

        figsize = 8
        if truescale == True:
            if manual_limits is not None:
                xratio = (manual_limits[1] - manual_limits[0]) / (manual_limits[3] - manual_limits[2]) * 1.3
                yratio = (manual_limits[3] - manual_limits[2]) / (manual_limits[1] - manual_limits[0])
                fig = plt.figure(figsize=(figsize * xratio, figsize * yratio))
        else:
            fig = plt.figure(figsize=((figsize*1.1)*1.3, figsize*0.92))
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
        ax = fig.add_subplot(gs[0])
        ax.set_facecolor('none')

        all_x = []
        all_y = []
        all_xerr = []
        all_yerr = []

        for i, (m, r) in enumerate(itertools.product(compare_masks, compare_radii)):
            # Construct CSV paths
            path_AGN = f"{base_AGN}/gas_analysis_summary_{m}_{r}kpc.csv"
            path_inactive = f"{base_inactive}/gas_analysis_summary_{m}_{r}kpc.csv"
            path_aux = f"{base_aux}/gas_analysis_summary_{m}_{r}kpc.csv"
            path_cont_AGN = f"{base_AGN}/cont_analysis_results/cont_analysis_summary_{r}kpc.csv"
            path_cont_inactive = f"{base_inactive}/cont_analysis_results/cont_analysis_summary_{r}kpc.csv"

            if not os.path.exists(path_AGN):
                print(f"WARNING: {path_AGN} not found")
                continue
            if not os.path.exists(path_inactive):
                print(f"WARNING: {path_inactive} not found")
                continue
            if not os.path.exists(path_aux):
                print(f"WARNING: {path_aux} not found")
                use_aux = False

            fit_data_AGN = pd.read_csv(path_AGN)
            fit_data_inactive = pd.read_csv(path_inactive)
            if use_aux:
                fit_data_aux = pd.read_csv(path_aux)
            if use_cont:
                if os.path.exists(path_cont_AGN):
                    cont_data_AGN = pd.read_csv(path_cont_AGN)
                    overlap = set(fit_data_AGN.columns) & set(cont_data_AGN.columns)
                    overlap -= {"Galaxy"}  # keep join key
                    cont_data_AGN_clean = cont_data_AGN.drop(columns=list(overlap))
                    fit_data_AGN = pd.merge(fit_data_AGN, cont_data_AGN_clean, left_on='Galaxy', right_on='Galaxy',how='left')
                if os.path.exists(path_cont_inactive):
                    cont_data_inactive = pd.read_csv(path_cont_inactive)
                    overlap = set(fit_data_inactive.columns) & set(cont_data_inactive.columns)
                    overlap -= {"Galaxy"}  # keep join key
                    cont_data_inactive_clean = cont_data_inactive.drop(columns=list(overlap))
                    fit_data_inactive = pd.merge(fit_data_inactive, cont_data_inactive_clean, left_on='Galaxy', right_on='Galaxy',how='left')

            df_combined['Name_clean'] = normalize_name(df_combined['Name'])
            llamatab['name_clean'] = normalize_name(llamatab['name'])
            agn_bol['Name_clean'] = normalize_name(agn_bol['Name'])
            inactive_bol['Name_clean'] = normalize_name(inactive_bol['Name'])
            fit_data_AGN['Galaxy_clean'] = normalize_name(fit_data_AGN['Galaxy'])
            fit_data_inactive['Galaxy_clean'] = normalize_name(fit_data_inactive['Galaxy'])
            if use_aux:
                fit_data_aux['Galaxy_clean'] = normalize_name(fit_data_aux['Galaxy'])

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
            inactive_bol = inactive_bol.copy()
            inactive_bol.loc[:, 'log LAGN'] = pd.to_numeric(inactive_bol['log LAGN'], errors='coerce')
            # inactive_bol['log LX'] = inactive_bol['log LAGN'].apply(
            #     lambda log_LAGN: math.log10((12.76 * (1 + (log_LAGN - math.log10(3.82e33)) / 12.15)**18.78) * 3.82e33)
            # )

            # Merge with fit data
            merged_AGN = pd.merge(df_combined, fit_data_AGN, left_on='id', right_on='Galaxy_clean',how='right')
            merged_AGN = pd.merge(merged_AGN, agn_bol, left_on='Name_clean', right_on='Name_clean',how='left')
            merged_inactive = pd.merge(df_combined, fit_data_inactive, left_on='id', right_on='Galaxy_clean',how='right')
            merged_inactive = pd.merge(merged_inactive, inactive_bol, left_on='Name_clean', right_on='Name_clean',how='left')

            # Add derived log LX column from flux and distance
            BAT_sens = 0.535e-10 # erg/cm2/s
            BAT_sens_flux = BAT_sens * 1e-7 * 1e4
            merged_inactive['log LX'] = np.log10(BAT_sens_flux * 4 * math.pi * (merged_inactive['Distance (Mpc)'] * 3.086e24)**2)

                # Clean AGN data
            merged_AGN[x_column] = pd.to_numeric(merged_AGN[x_column], errors='coerce')
            merged_AGN[y_column] = pd.to_numeric(merged_AGN[y_column], errors='coerce')
            merged_AGN_clean = merged_AGN.dropna(subset=[x_column, y_column])

            # Clean inactive data
            merged_inactive[x_column] = pd.to_numeric(merged_inactive[x_column], errors='coerce')
            merged_inactive[y_column] = pd.to_numeric(merged_inactive[y_column], errors='coerce')
            merged_inactive_clean = merged_inactive.dropna(subset=[x_column, y_column])

            x_agn = merged_AGN_clean[x_column]
            y_agn = merged_AGN_clean[y_column]
            x_inactive = merged_inactive_clean[x_column]
            y_inactive = merged_inactive_clean[y_column]


                    # --- Exclude names here ---
            if exclude_names is not None:

                excluded_x = []
                excluded_y = []

                exclude_norm = [n.strip().upper() for n in exclude_names]

                # AGN
                if nativex:
                    excluded_rows = merged_AGN_clean_minres[
                        merged_AGN_clean_minres["Name_clean"].str.strip().str.upper().isin(exclude_norm)
                    ]
                    excluded_x.extend(excluded_rows[x_column].dropna().tolist())
                else:
                    mask_excluded_agn = merged_AGN_clean_maxres["Name_clean"].str.strip().str.upper().isin(exclude_norm)
                    excluded_x.extend(x_agn[mask_excluded_agn])

                if nativey:
                    excluded_rows = merged_AGN_clean_minres[
                        merged_AGN_clean_minres["Name_clean"].str.strip().str.upper().isin(exclude_norm)
                    ]
                    excluded_y.extend(excluded_rows[y_column].dropna().tolist())
                else:
                    mask_excluded_agn = merged_AGN_clean_maxres["Name_clean"].str.strip().str.upper().isin(exclude_norm)
                    excluded_y.extend(y_agn[mask_excluded_agn])

                # Inactive
                if nativex:
                    excluded_rows = merged_inactive_clean_minres[
                        merged_inactive_clean_minres["Name_clean"].str.strip().str.upper().isin(exclude_norm)
                    ]
                    excluded_x.extend(excluded_rows[x_column].dropna().tolist())
                else:
                    mask_excluded_inactive = merged_inactive_clean_maxres["Name_clean"].str.strip().str.upper().isin(exclude_norm)
                    excluded_x.extend(x_inactive[mask_excluded_inactive])

                if nativey:
                    excluded_rows = merged_inactive_clean_minres[
                        merged_inactive_clean_minres["Name_clean"].str.strip().str.upper().isin(exclude_norm)
                    ]
                    excluded_y.extend(excluded_rows[y_column].dropna().tolist())
                else:
                    mask_excluded_inactive = merged_inactive_clean_maxres["Name_clean"].str.strip().str.upper().isin(exclude_norm)
                    excluded_y.extend(y_inactive[mask_excluded_inactive])


                merged_AGN_clean = merged_AGN_clean[
                    ~merged_AGN_clean["Name_clean"].str.strip().str.upper().isin(exclude_norm)
                ]


                merged_inactive_clean = merged_inactive_clean[
                    ~merged_inactive_clean["Name_clean"].str.strip().str.upper().isin(exclude_norm)
                ]
                agn_bol = agn_bol[
                    ~agn_bol["Name_clean"].str.strip().str.upper().isin(exclude_norm)
                ]
                inactive_bol = inactive_bol[
                    ~inactive_bol["Name_clean"].str.strip().str.upper().isin(exclude_norm)
                ]

            merged_AGN_clean = merged_AGN_clean.replace(
                [np.inf, -np.inf], np.nan
            )

            merged_AGN_clean = merged_AGN_clean.infer_objects(copy=False)

            merged_AGN_clean = merged_AGN_clean.dropna(
                subset=[x_column, y_column]
            )

            merged_inactive_clean = merged_inactive_clean.replace(
                [np.inf, -np.inf], np.nan
            )

            merged_inactive_clean = merged_inactive_clean.infer_objects(copy=False)

            merged_inactive_clean = merged_inactive_clean.dropna(
                subset=[x_column, y_column]
            )


            x_agn = merged_AGN_clean[x_column]
            y_agn = merged_AGN_clean[y_column]
            names_agn = merged_AGN_clean["Name_clean"].str.replace(" ", "", regex=False).values
            xerr_agn = get_errorbars(merged_AGN_clean, x_column)
            yerr_agn = get_errorbars(merged_AGN_clean, y_column)

            x_inactive = merged_inactive_clean[x_column]
            y_inactive = merged_inactive_clean[y_column]

            names_inactive = merged_inactive_clean["Name_clean"].str.replace(" ", "", regex=False).values
            xerr_inactive = get_errorbars(merged_inactive_clean, x_column)
            yerr_inactive = get_errorbars(merged_inactive_clean, y_column)

            x_comb = pd.concat([x_agn.dropna(), x_inactive.dropna()])
            y_comb = pd.concat([y_agn.dropna(), y_inactive.dropna()])
            xerr_comb = pd.concat([pd.Series(xerr_agn).dropna(), pd.Series(xerr_inactive).dropna()])
            yerr_comb = pd.concat([pd.Series(yerr_agn).dropna(), pd.Series(yerr_inactive).dropna()])

            all_x.append(x_comb)
            all_y.append(y_comb)
            all_xerr.append(xerr_comb)
            all_yerr.append(yerr_comb)

            # Plot on same axes
            ax.errorbar(
                x_comb, y_comb,
                xerr=xerr_comb, yerr=yerr_comb,
                color=colours[i],
                fmt=markers[i],
                markersize=6,
                capsize=2,
                elinewidth=1,
                alpha=0.5,
                label=f"{m}_{r}kpc"
            )
        # --- Mean point (100% opacity, double size) ---
            mean_x = np.nanmean(x_comb)
            mean_y = np.nanmean(y_comb)

            ax.scatter(
                mean_x, mean_y,
                color=colours[i],
                marker=markers[i],
                s=6**2 * 2,   # double size relative to markersize=6
                edgecolor='black',
                linewidth=0.8,
                alpha=1.0,
                zorder=5
            )

        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.grid(True)
        ax.legend()
        ax.set_title(f"Comparison of {y_column} vs {x_column} for all masks and radii")
        # Build a filesystem-safe suffix listing all masks and radii
        suffix_masks = "_".join(str(m).replace(" ", "").replace(".", "p") for m in compare_masks)
        suffix_radii = "_".join(str(r).replace(" ", "").replace(".", "p") for r in compare_radii)
        suffix = f"_{suffix_masks}_{suffix_radii}"

        output_path = f'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/Plots/compare{suffix}_{x_column}_vs_{y_column}.png'
        plt.savefig(output_path)
        print(f"Saved comparison plot to: {output_path}")
        plt.close(fig)
        return

###################### end of comparison block ##########################
    if not compare:

        # Handle all filename logic
        if rebin is not None and mask is not None:
            AGN_path = f"{base_AGN}/gas_analysis_summary_{rebin}pc_{mask}_{R_kpc}kpc.csv"
            inactive_path = f"{base_inactive}/gas_analysis_summary_{rebin}pc_{mask}_{R_kpc}kpc.csv"

        else:
            AGN_path = f"{base_AGN}/gas_analysis_summary_{mask}_{R_kpc}kpc.csv"
            inactive_path = f"{base_inactive}/gas_analysis_summary_{mask}_{R_kpc}kpc.csv"

        # ---- Final fallback checks ----
        if not os.path.exists(AGN_path):
            print(f"WARNING: {AGN_path} not found")
            return

        if not os.path.exists(inactive_path):
            print(f"WARNING: {inactive_path} not found")
            return

        # ---- Load files ----
        fit_data_AGN = pd.read_csv(AGN_path)
        fit_data_inactive = pd.read_csv(inactive_path)
        if use_aux:
            if rebin is not None and mask is not None:
                aux_path = f"{base_aux}/gas_analysis_summary_{rebin}pc_{mask}_{R_kpc}kpc.csv"
            else:
                aux_path = f"{base_aux}/gas_analysis_summary_{mask}_{R_kpc}kpc.csv"
            if not os.path.exists(aux_path):
                print(f"WARNING: {aux_path} not found")
                use_aux = False
            if use_aux:
                fit_data_aux = pd.read_csv(aux_path)
        if use_cont:

            cont_AGN_path = f"{base_cont_AGN}/cont_analysis_summary_{R_kpc}kpc.csv"
            cont_inactive_path = f"{base_cont_inactive}/cont_analysis_summary_{R_kpc}kpc.csv"

            if os.path.exists(cont_AGN_path):
                cont_data_AGN = pd.read_csv(cont_AGN_path)
                overlap = set(fit_data_AGN.columns) & set(cont_data_AGN.columns)
                overlap -= {"Galaxy"}  # keep join key
                cont_data_AGN_clean = cont_data_AGN.drop(columns=list(overlap))
                fit_data_AGN = pd.merge(fit_data_AGN, cont_data_AGN_clean, left_on='Galaxy', right_on='Galaxy',how='left')
            else:
                print(f"WARNING: {cont_AGN_path} not found")
            if os.path.exists(cont_inactive_path):
                cont_data_inactive = pd.read_csv(cont_inactive_path)
                overlap = set(fit_data_inactive.columns) & set(cont_data_inactive.columns)
                overlap -= {"Galaxy"}  # keep join key
                cont_data_inactive_clean = cont_data_inactive.drop(columns=list(overlap))
                fit_data_inactive = pd.merge(fit_data_inactive, cont_data_inactive_clean, left_on='Galaxy', right_on='Galaxy',how='left')
            else:
                print(f"WARNING: {cont_inactive_path} not found")

        df_combined['Name_clean'] = normalize_name(df_combined['Name'])
        llamatab['name_clean'] = normalize_name(llamatab['name'])
        agn_bol['Name_clean'] = normalize_name(agn_bol['Name'])
        inactive_bol['Name_clean'] = normalize_name(inactive_bol['Name'])
        fit_data_AGN['Galaxy_clean'] = normalize_name(fit_data_AGN['Galaxy'])
        fit_data_inactive['Galaxy_clean'] = normalize_name(fit_data_inactive['Galaxy'])
        if use_aux:
            fit_data_aux['Galaxy_clean'] = normalize_name(fit_data_aux['Galaxy'])


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
        



        # Add derived LX column (deprecated)
        inactive_bol = inactive_bol.copy()
        inactive_bol.loc[:, 'log LAGN'] = pd.to_numeric(inactive_bol['log LAGN'], errors='coerce')
        # inactive_bol['log LX'] = inactive_bol['log LAGN'].apply(
        #     lambda log_LAGN: math.log10((12.76 * (1 + (log_LAGN - math.log10(3.82e33)) / 12.15)**18.78) * 3.82e33)
        # )
        


        # Merge with fit data
        merged_AGN = pd.merge(df_combined, fit_data_AGN, left_on='id', right_on='Galaxy_clean',how='right')
        merged_AGN = pd.merge(merged_AGN, agn_bol, left_on='Name_clean', right_on='Name_clean',how='left')
        merged_inactive = pd.merge(df_combined, fit_data_inactive, left_on='id', right_on='Galaxy_clean',how='right')
        merged_inactive = pd.merge(merged_inactive, inactive_bol, left_on='Name_clean', right_on='Name_clean',how='left')

        # Add derived log LX column from flux and distance
        BAT_sens = 0.535e-10 # erg/cm2/s
        BAT_sens_flux = BAT_sens * 1e-7 * 1e4
        merged_inactive['log LX'] = np.log10(BAT_sens_flux * 4 * math.pi * (merged_inactive['Distance (Mpc)'] * 3.086e24)**2)

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

        merged_AGN_clean_res = merged_AGN_clean.sort_values("Resolution (pc)", ascending=True)
        merged_AGN_clean_maxres = merged_AGN_clean_res.drop_duplicates(subset="Name_clean", keep="last")
        merged_AGN_clean_minres = merged_AGN_clean_res.drop_duplicates(subset="Name_clean", keep="first")

        merged_inactive_clean_res = merged_inactive_clean.sort_values("Resolution (pc)", ascending=True)
        merged_inactive_clean_maxres = merged_inactive_clean_res.drop_duplicates(subset="Name_clean", keep="last")
        merged_inactive_clean_minres = merged_inactive_clean_res.drop_duplicates(subset="Name_clean", keep="first")

        if nativex:
            x_agn = merged_AGN_clean_minres[x_column]
            x_inactive = merged_inactive_clean_minres[x_column]
        else:
            x_agn = merged_AGN_clean_maxres[x_column]
            x_inactive = merged_inactive_clean_maxres[x_column]
        if nativey:
            y_inactive = merged_inactive_clean_minres[y_column]
            y_agn = merged_AGN_clean_minres[y_column]
        else:
            y_agn = merged_AGN_clean_maxres[y_column]
            y_inactive = merged_inactive_clean_maxres[y_column]

        # --- Exclude names here ---
        if exclude_names is not None:

            excluded_x = []
            excluded_y = []

            exclude_norm = [n.strip().upper() for n in exclude_names]

            # AGN
            if nativex:
                excluded_rows = merged_AGN_clean_minres[
                    merged_AGN_clean_minres["Name_clean"].str.strip().str.upper().isin(exclude_norm)
                ]
                excluded_x.extend(excluded_rows[x_column].dropna().tolist())
            else:
                mask_excluded_agn = merged_AGN_clean_maxres["Name_clean"].str.strip().str.upper().isin(exclude_norm)
                excluded_x.extend(x_agn[mask_excluded_agn])

            if nativey:
                excluded_rows = merged_AGN_clean_minres[
                    merged_AGN_clean_minres["Name_clean"].str.strip().str.upper().isin(exclude_norm)
                ]
                excluded_y.extend(excluded_rows[y_column].dropna().tolist())
            else:
                mask_excluded_agn = merged_AGN_clean_maxres["Name_clean"].str.strip().str.upper().isin(exclude_norm)
                excluded_y.extend(y_agn[mask_excluded_agn])

            # Inactive
            if nativex:
                excluded_rows = merged_inactive_clean_minres[
                    merged_inactive_clean_minres["Name_clean"].str.strip().str.upper().isin(exclude_norm)
                ]
                excluded_x.extend(excluded_rows[x_column].dropna().tolist())
            else:
                mask_excluded_inactive = merged_inactive_clean_maxres["Name_clean"].str.strip().str.upper().isin(exclude_norm)
                excluded_x.extend(x_inactive[mask_excluded_inactive])

            if nativey:
                excluded_rows = merged_inactive_clean_minres[
                    merged_inactive_clean_minres["Name_clean"].str.strip().str.upper().isin(exclude_norm)
                ]
                excluded_y.extend(excluded_rows[y_column].dropna().tolist())
            else:
                mask_excluded_inactive = merged_inactive_clean_maxres["Name_clean"].str.strip().str.upper().isin(exclude_norm)
                excluded_y.extend(y_inactive[mask_excluded_inactive])

            # These are already DataFrames → filter with boolean masks
            merged_AGN_clean = merged_AGN_clean_maxres[
                ~merged_AGN_clean_maxres["Name_clean"].str.strip().str.upper().isin(exclude_norm)
            ]


            merged_inactive_clean = merged_inactive_clean_maxres[
                ~merged_inactive_clean_maxres["Name_clean"].str.strip().str.upper().isin(exclude_norm)
            ]
            agn_bol = agn_bol[
                ~agn_bol["Name_clean"].str.strip().str.upper().isin(exclude_norm)
            ]
            inactive_bol = inactive_bol[
                ~inactive_bol["Name_clean"].str.strip().str.upper().isin(exclude_norm)
            ]

 #see archived_comp_samp_build 

        
        merged_AGN_clean_maxres = (
                    merged_AGN_clean_maxres
                    .replace([np.inf, -np.inf], np.nan)
                    .infer_objects(copy=False)
                    .dropna(subset=[x_column, y_column])
                )


        merged_inactive_clean_maxres = (
                merged_inactive_clean_maxres
                .replace([np.inf, -np.inf], np.nan)
                .infer_objects(copy=False)
                .dropna(subset=[x_column, y_column])
            )
        
        if use_aux:
            fit_data_aux = (
                fit_data_aux
                .replace([np.inf, -np.inf], np.nan)
                .infer_objects(copy=False)
                .dropna(subset=[x_column, y_column])
            )
            fit_data_AGN[x_column] = pd.to_numeric(fit_data_AGN[x_column], errors='coerce')
            fit_data_AGN[y_column] = pd.to_numeric(fit_data_AGN[y_column], errors='coerce')
            x_aux = fit_data_aux[x_column]
            y_aux = fit_data_aux[y_column]
            names_aux = fit_data_aux["Galaxy"].values

        if use_gb21:
 #see archived_comp_samp_build 
            GB21_df = pd.read_csv("/Users/administrator/Astro/LLAMA/ALMA/comp_samples"+"/GB21_df.csv")
            GB21_df[x_column] = pd.to_numeric(GB21_df[x_column], errors='coerce')
            GB21_df[y_column] = pd.to_numeric(GB21_df[y_column], errors='coerce')
            GB21_clean = GB21_df.dropna(subset=[x_column, y_column])
            x_gb21 = GB21_clean[x_column]
            y_gb21 = GB21_clean[y_column]
            names_gb21 = GB21_clean["Name"].values
        if use_wis:
 #see archived_comp_samp_build 
            wis_df = pd.read_csv("/Users/administrator/Astro/LLAMA/ALMA/comp_samples"+"/wis_df.csv")
            wis_df[x_column] = pd.to_numeric(wis_df[x_column], errors='coerce')
            wis_df[y_column] = pd.to_numeric(wis_df[y_column], errors='coerce')
            wis_clean = wis_df.dropna(subset=[x_column, y_column])
            
            x_wis = wis_clean[x_column]
            y_wis = wis_clean[y_column]
            names_wis = wis_clean["Name"].values
        if use_phangs:
 #see archived_comp_samp_build 
            phangs_df = pd.read_csv("/Users/administrator/Astro/LLAMA/ALMA/comp_samples"+"/phangs_df.csv")
            phangs_df[x_column] = pd.to_numeric(phangs_df[x_column], errors='coerce')
            phangs_df[y_column] = pd.to_numeric(phangs_df[y_column], errors='coerce')
            phangs_clean = phangs_df.dropna(subset=[x_column, y_column])
            x_phangs = phangs_clean[x_column]
            y_phangs = phangs_clean[y_column]
            names_phangs = phangs_clean["Name"].values

        if use_sim:
 #see archived_comp_samp_build 
            sim_df = pd.read_csv("/Users/administrator/Astro/LLAMA/ALMA/comp_samples"+"/sim_df.csv")
            sim_df[x_column] = pd.to_numeric(sim_df[x_column], errors='coerce')
            sim_df[y_column] = pd.to_numeric(sim_df[y_column], errors='coerce')
            sim_clean = sim_df.dropna(subset=[x_column, y_column])
            x_sim = sim_clean[x_column]
            y_sim = sim_clean[y_column]
            names_sim = sim_clean["Name"].values


        if x_column != 'log L′ CO':

            x_agn = merged_AGN_clean_maxres[x_column]
            y_agn = merged_AGN_clean_maxres[y_column]
            names_agn = merged_AGN_clean_maxres["Name_clean"].str.replace(" ", "", regex=False).values
            xerr_agn = get_errorbars(merged_AGN_clean_maxres, x_column)
            yerr_agn = get_errorbars(merged_AGN_clean_maxres, y_column)

            x_inactive = merged_inactive_clean_maxres[x_column]
            y_inactive = merged_inactive_clean_maxres[y_column]

            names_inactive = merged_inactive_clean_maxres["Name_clean"].str.replace(" ", "", regex=False).values
            xerr_inactive = get_errorbars(merged_inactive_clean_maxres, x_column)
            yerr_inactive = get_errorbars(merged_inactive_clean_maxres, y_column)

############################## Special handling for L' CO comparison with ROS+18 ##############################

        if x_column == 'log L′ CO':
            ### insert code here ####
            df_ros_obs = pd.DataFrame(Rosario2018_obs)
            df_ros_obs['Name_clean'] = normalize_name(df_ros_obs['Name'])#.str.replace(" ", "", regex=False)
            merged_AGN_clean_maxres = merged_AGN_clean_maxres.merge(
                df_ros_obs[['Name_clean', 'Telescope']],
                left_on='Name_clean',
                right_on='Name_clean',
                how='left'
            )
            merged_inactive_clean_maxres = merged_inactive_clean_maxres.merge(
                df_ros_obs[['Name_clean', 'Telescope']],
                left_on='Name_clean',
                right_on='Name_clean',
                how='left'
            )
            telescope_to_col = {
                    'JCMT': "L'CO_JCMT (K km s pc2)",
                    'APEX': "L'CO_APEX (K km s pc2)"
                }
            telescope_to_errcol = {
                    'JCMT': "L'CO_JCMT_err (K km s pc2)",
                    'APEX': "L'CO_APEX_err (K km s pc2)"
                }

            def select_LCO(row,key):
                tel = row['Telescope']
                if tel in key:
                    return row[key[tel]]
                return np.nan

            x_agn = merged_AGN_clean_maxres[x_column]
            y_agn = merged_AGN_clean_maxres.apply(
                lambda row: select_LCO(row, telescope_to_col),
                axis=1
            )
            names_agn = merged_AGN_clean_maxres["Name_clean"].str.replace(" ", "", regex=False).values
            merged_AGN_clean_maxres['LCO_err_selected'] = merged_AGN_clean_maxres.apply(
                lambda row: select_LCO(row, telescope_to_errcol),
                axis=1
            )
            xerr_agn = get_errorbars(merged_AGN_clean_maxres, x_column)
            yerr_agn = get_errorbars(merged_AGN_clean_maxres, "LCO_err_selected")

            x_inactive = merged_inactive_clean_maxres[x_column]
            y_inactive = merged_inactive_clean_maxres.apply(
                lambda row: select_LCO(row, telescope_to_col),
                axis=1
            )
            names_inactive = merged_inactive_clean_maxres["Name_clean"].str.replace(" ", "", regex=False).values
            merged_inactive_clean_maxres['LCO_err_selected'] = merged_inactive_clean_maxres.apply(
                lambda row: select_LCO(row, telescope_to_errcol),
                axis=1
            )

            merged_AGN_clean_maxres['LCO_err_selected'] = pd.to_numeric(
            merged_AGN_clean_maxres['LCO_err_selected'], errors='coerce'
        )

            merged_inactive_clean_maxres['LCO_err_selected'] = pd.to_numeric(
            merged_inactive_clean_maxres['LCO_err_selected'], errors='coerce'
        )

            xerr_inactive = get_errorbars(merged_inactive_clean_maxres, x_column)
            yerr_inactive = get_errorbars(merged_inactive_clean_maxres, "LCO_err_selected")

        ############################## Prep data for plotting ##############################

        names_llama = list(names_agn) + list(names_inactive)
        if use_phangs:
            shared_names_phangs = [name if name in names_llama else None for name in names_phangs]
            if not plotshared:
                mask_keep = ~np.isin(names_phangs, names_llama)
                x_phangs = x_phangs[mask_keep]
                y_phangs = y_phangs[mask_keep]


        if use_wis:
            shared_names_wis = [name if name in names_llama or name in ["NGC1387","NGC5064"] else None for name in names_wis]
            if not plotshared:
                mask_keep = ~np.isin(names_wis, names_llama)
                x_wis = x_wis[mask_keep]
                y_wis = y_wis[mask_keep]

        if soloplot == 'AGN':
            if x_agn.empty or y_agn.empty:
                print("No valid AGN data to plot.")
                return
        elif soloplot == 'inactive':
            if x_inactive.empty or y_inactive.empty:
                print("No valid inactive data to plot.")
                return
        else:
            if x_agn.empty and x_inactive.empty and (not use_gb21 or x_gb21.empty) and (not use_wis or x_wis.empty) and (not use_phangs or x_phangs.empty) and (not use_sim or x_sim.empty):
                print("No valid X data to plot.")
                return
            if y_agn.empty and y_inactive.empty and (not use_gb21 or y_gb21.empty) and (not use_wis or y_wis.empty) and (not use_phangs or y_phangs.empty) and (not use_sim or y_sim.empty):
                print("No valid Y data to plot.")
                return

        ############################### Set up figure ##############################

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

        ############################## Spearman coefficients ##############################

            from scipy.stats import spearmanr
            from scipy.stats import pearsonr
            global stats_rows

            if soloplot is None: 
                statistic, p_value = ks_2samp(y_inactive, y_agn) 

            # Helper: return (rho, p) or (None, None)
            def safe_spearman(x, y, enabled):
                if not enabled:
                    return None, None
                if x is None or y is None or len(x) < 3 or len(y) < 3:
                    return None, None
                rho, p = spearmanr(x, y, nan_policy='omit')
                return float(rho), float(p)

            # Compute spearman values for each group
            rho_agn, p_agn = safe_spearman(x_agn, y_agn, True)
            rho_inact, p_inact = safe_spearman(x_inactive, y_inactive, True)

            rho_comb, p_comb = safe_spearman(
                pd.concat([x_agn, x_inactive]), 
                pd.concat([y_agn, y_inactive]), 
                True
            )
            if use_gb21:
                rho_gb21, p_gb21 = safe_spearman(x_gb21, y_gb21, use_gb21)
            else:
                rho_gb21, p_gb21 = None, None
            if use_wis:
                rho_wis, p_wis = safe_spearman(x_wis, y_wis, use_wis)
            else:
                rho_wis, p_wis = None, None
            if use_phangs:
                rho_phangs, p_phangs = safe_spearman(x_phangs, y_phangs, use_phangs)
            else:
                rho_phangs, p_phangs = None, None
            if use_sim:
                rho_sim, p_sim = safe_spearman(x_sim, y_sim, use_sim)
            else:
                rho_sim, p_sim = None, None

            # KS value (already computed earlier)
            ks_stat = statistic if soloplot is None else None
            ks_p = p_value if soloplot is None else None

            new_row = {
                "x_column": x_column,
                "y_column": y_column,
                "ks_stat": ks_stat,
                "ks_p": ks_p,
                "spearman_agn": rho_agn,
                "spearman_agn_p": p_agn,
                "spearman_inactive": rho_inact,
                "spearman_inactive_p": p_inact,
                "spearman_llama_comb": rho_comb,
                "spearman_llama_comb_p": p_comb,
                "spearman_gb21": rho_gb21,
                "spearman_gb21_p": p_gb21,
                "spearman_wis": rho_wis,
                "spearman_wis_p": p_wis,
                "spearman_phangs": rho_phangs,
                "spearman_phangs_p": p_phangs,
                "spearman_sim": rho_sim,
                "spearman_sim_p": p_sim,
                "rebin": rebin,
                "mask": mask,
                "R_kpc": R_kpc
            }
            
            stats_rows.append(new_row)   # new_row is a dict


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
            if soloplot is None and use_wis:
                data_x.append(x_wis)
                data_y.append(y_wis)
            if soloplot is None and use_phangs:
                data_x.append(x_phangs)
                data_y.append(y_phangs)
            if soloplot is None and use_sim:
                data_x.append(x_sim)
                data_y.append(y_sim)

            all_x = pd.concat(data_x)
            all_y = pd.concat(data_y)


            # ===================== Axis limits =====================

            if manual_limits is not None:
                xlower, xupper, ylower, yupper = map(float, manual_limits)

            else:
                # ---- finite values only ----
                finite_x = all_x[np.isfinite(all_x)]
                finite_y = all_y[np.isfinite(all_y)]

                if finite_x.empty or finite_y.empty:
                    raise ValueError("No finite data for axis limits")

                # ---- X limits ----
                if logx:
                    valid_x = finite_x[finite_x > 0]
                    if valid_x.empty:
                        raise ValueError("Cannot use log x-axis: no positive x values")
                    xmin, xmax = valid_x.min(), valid_x.max()
                    xlower = xmin / 10**0.1
                    xupper = xmax * 10**0.1
                else:
                    xmin, xmax = finite_x.min(), finite_x.max()
                    span = xmax - xmin
                    pad = 0.05 * span if span > 0 else 0.1
                    xlower = xmin - pad
                    xupper = xmax + pad

                # ---- Y limits ----
                if logy:
                    valid_y = finite_y[finite_y > 0]
                    if valid_y.empty:
                        raise ValueError("Cannot use log y-axis: no positive y values")
                    ymin, ymax = valid_y.min(), valid_y.max()
                    ylower = ymin / 10**0.1
                    yupper = ymax * 10**0.1
                else:
                    ymin, ymax = finite_y.min(), finite_y.max()
                    span = ymax - ymin
                    pad = 0.05 * span if span > 0 else 0.05
                    ylower = ymin - pad
                    yupper = ymax + pad

                # ===================== Square enforcement =====================
                if square:

                    # Convert bounds to "comparison space"
                    if logx:
                        x0, x1 = np.log10(xlower), np.log10(xupper)
                    else:
                        x0, x1 = xlower, xupper

                    if logy:
                        y0, y1 = np.log10(ylower), np.log10(yupper)
                    else:
                        y0, y1 = ylower, yupper

                    # Determine square extent
                    lo = min(x0, y0)
                    hi = max(x1, y1)

                    # Convert back
                    if logx:
                        xlower, xupper = 10**lo, 10**hi
                    else:
                        xlower, xupper = lo, hi

                    if logy:
                        ylower, yupper = 10**lo, 10**hi
                    else:
                        ylower, yupper = lo, hi

            # ===================== Final safety checks =====================
            if logx and xlower <= 0:
                positives = finite_x[finite_x > 0]
                xlower = positives.min() * 0.9

            if logy and ylower <= 0:
                positives = finite_y[finite_y > 0]
                ylower = positives.min() * 0.9

            for val, name in zip(
                [xlower, xupper, ylower, yupper],
                ["xlower", "xupper", "ylower", "yupper"]
            ):
                if not np.isfinite(val):
                    raise ValueError(f"{name} is invalid: {val}")

            # ===================== End limits =====================



                ############################### Background image ##############################

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


            ############################### Reference lines ##############################

                        # ---------------- 1:1 reference line ----------------

            if square:

                # Convert limits to comparison space
                if logx:
                    x0, x1 = np.log10(xlower), np.log10(xupper)
                else:
                    x0, x1 = xlower, xupper

                if logy:
                    y0, y1 = np.log10(ylower), np.log10(yupper)
                else:
                    y0, y1 = ylower, yupper

                # Overlapping square region
                lo = max(x0, y0)
                hi = min(x1, y1)

                if lo < hi:

                    # Convert back to plotting space
                    if logx:
                        xline = 10**np.array([lo, hi])
                    else:
                        xline = np.array([lo, hi])

                    if logy:
                        yline = 10**np.array([lo, hi])
                    else:
                        yline = np.array([lo, hi])

                    ax_scatter.plot(
                        xline,
                        yline,
                        linestyle='-',
                        color='darkgreen',
                        linewidth=1,
                        zorder=0
                    )
            if best_fit:

                # --- Line of best fit (handles log axes) ---
                try:
                    # Build DataFrame of all plotted points used for axis limits (all_x/all_y exist)
                    x_all = pd.to_numeric(all_x, errors='coerce')
                    y_all = pd.to_numeric(all_y, errors='coerce')
                    df_fit = pd.DataFrame({'x': x_all, 'y': y_all}).dropna()

                    # Apply positivity requirement for any axis that is log-scaled
                    if logx:
                        df_fit = df_fit[df_fit['x'] > 0]
                    if logy:
                        df_fit = df_fit[df_fit['y'] > 0]

                    if len(df_fit) < 2:
                        print("Not enough finite points for best-fit line.")
                    else:
                        # Transform to fitting space (the plotting/display space)
                        x_fit = np.log10(df_fit['x']) if logx else df_fit['x'].values
                        y_fit = np.log10(df_fit['y']) if logy else df_fit['y'].values

                        # Require at least 2 distinct x values
                        if np.nanstd(x_fit) == 0:
                            print("Cannot fit line: x values have zero variance.")
                        else:
                            # Linear fit in transformed space
                            m, c = np.polyfit(x_fit, y_fit, 1)

                            # Create line in original plotting coordinates
                            if logx:
                                x_line = np.logspace(np.log10(xlower), np.log10(xupper), 200)
                                x_line_t = np.log10(x_line)
                            else:
                                x_line = np.linspace(xlower, xupper, 200)
                                x_line_t = x_line

                            y_line_t = m * x_line_t + c
                            # Transform back if y is on log scale
                            if logy:
                                y_line = 10 ** (y_line_t)
                            else:
                                y_line = y_line_t

                            # Mask any non-finite values (can occur if y_line<=0 with logy)
                            mask_valid = np.isfinite(y_line) & np.isfinite(x_line)
                            x_line = x_line[mask_valid]
                            y_line = y_line[mask_valid]

                            if x_line.size > 0:
                                # Compute a single-number "gradient" to show in legend (1 decimal)
                                try:
                                    if logx and logy:
                                        # power-law exponent in log-log space
                                        grad_val = m
                                        grad_label = f"p={grad_val:.1f}"
                                    elif (not logx) and (not logy):
                                        # simple linear slope
                                        grad_val = m
                                        grad_label = f"m={grad_val:.1f}"
                                    elif logx and (not logy):
                                        # originally computed dy/dx; user requested swap x and y -> show dx/dy
                                        # y = m*log10(x) + c  => dy/dx = m / (x ln 10)
                                        # dx/dy = 1 / (dy/dx) = x ln(10) / m
                                        x_ref = float(10 ** np.median(np.log10(df_fit['x'])))
                                        try:
                                            grad_val = (x_ref * np.log(10)) / m
                                            grad_label = f"dx/dy={grad_val:.1f}"
                                        except Exception:
                                            grad_label = "dx/dy=inf"
                                    else:  # not logx and logy
                                        # originally computed dy/dx; user requested swap x and y -> show dx/dy
                                        # log10(y) = m*x + c => y = 10^(m x + c)
                                        # dy/dx = ln(10) * m * y  -> dx/dy = 1 / (ln(10) * m * y)
                                        x_ref = float(np.median(df_fit['x']))
                                        y_ref = 10 ** (m * x_ref + c)
                                        try:
                                            grad_val = 1.0 / (np.log(10) * m * y_ref)
                                            grad_label = f"dx/dy={grad_val:.1f}"
                                        except Exception:
                                            grad_label = "dx/dy=inf"
                                except Exception:
                                    grad_label = f"m={m:.1f}"

                                ax_scatter.plot(x_line, y_line, linestyle='--', color='black', linewidth=1.2, zorder=1)
                                # Optionally annotate slope and fit stats in the axes text
                                try:
                                    r, _ = pearsonr(x_fit, y_fit)
                                    stat_str = f"m={m:.3f}, c={c:.3f}, r={r:.2f}"
                                except Exception:
                                    stat_str = f"m={m:.3f}, c={c:.3f}"
                                xpos = 0.02 * (xupper - xlower) + xlower
                                ypos = 0.95 * (yupper - ylower) + ylower
                                ax_scatter.text(xpos, ypos, stat_str, fontsize=8, color='black', verticalalignment='top', clip_on=False)

                except Exception as e:
                    print(f"Best-fit line failed: {e}")

            ############################### Plot scatter points ##############################

            colour_AGN = 'red'
            colour_inactive = 'blue'
            label_AGN = 'LLAMA AGN'
            label_inactive = 'LLAMA Inactive'
            marker_AGN = 's'
            marker_inactive = 'v'
            if x_column == 'log LX':
                marker_inactive = '<'
            if comb_llama:
                colour_AGN = 'black'
                colour_inactive = 'black'
                label_AGN = 'LLAMA Galaxies'
                label_inactive = None
                marker_AGN = 'o'
                marker_inactive = 'o'

            if yerr_inactive is None:
                print("yerr_inactive is None")
            else:
                try:
                    # safe coercion to float array then count finite entries
                    arr = np.asarray(yerr_inactive, dtype=float)
                    finite_count = np.isfinite(arr).sum()
                    total = arr.size
                    print(f"yerr_inactive finite: {finite_count} / {total}")
                except Exception:
                    # fallback: iterate and try converting each element to float
                    vals = np.atleast_1d(yerr_inactive)
                    finite_mask = []
                    for v in vals:
                        try:
                            fv = float(v)
                            finite_mask.append(np.isfinite(fv))
                        except Exception:
                            finite_mask.append(False)
                    finite_count = int(np.sum(finite_mask))
                    total = len(finite_mask)
                    print(f"yerr_inactive finite (fallback): {finite_count} / {total}")
            # Safely show example values (handle None and unexpected types)
            if yerr_inactive is None:
                print("example yerr values: None")
            else:
                try:
                    sample = yerr_inactive[:10]
                except Exception:
                    sample = np.atleast_1d(yerr_inactive)[:10]
                print("example yerr values:", sample)

            if soloplot in (None, 'AGN'):
                ax_scatter.errorbar(
                    x_agn, y_agn,
                    xerr=xerr_agn, yerr=yerr_agn,
                    fmt=marker_AGN, color=colour_AGN, label=label_AGN, markersize=6,
                    capsize=2, elinewidth=1, alpha=0.8
                )
                if not comb_llama:
                    for x, y, name in zip(x_agn, y_agn, names_agn):
                        ax_scatter.text(float(x + 0.005), float(y), name, fontsize=7, color='darkred', zorder=10)
                elif comb_llama and use_phangs and use_wis:
                    names_phangs_wis = list(names_phangs) + list(names_wis)
                    shared_names_agn = [x if x in names_phangs_wis else None for x in names_agn]
                    for x, y, name in zip(x_agn, y_agn, shared_names_agn):
                        ax_scatter.text(float(x + 0.005), float(y), name, fontsize=7, color='black', zorder=10)
                elif comb_llama and use_phangs and not use_wis:
                    shared_names_agn = [x if x in names_wis else None for x in names_agn]
                    for x, y, name in zip(x_agn, y_agn, shared_names_agn):
                        ax_scatter.text(float(x + 0.005), float(y), name, fontsize=7, color='black', zorder=10)
                elif comb_llama and use_wis and not use_phangs:
                    shared_names_agn = [x if x in names_phangs else None for x in names_agn]
                    for x, y, name in zip(x_agn, y_agn, shared_names_agn):
                        ax_scatter.text(float(x + 0.005), float(y), name, fontsize=7, color='black', zorder=10)
                    



            if soloplot in (None, 'inactive'):
                ax_scatter.errorbar(
                    x_inactive, y_inactive,
                    xerr=xerr_inactive, yerr=yerr_inactive,
                    fmt=marker_inactive, color=colour_inactive, label=label_inactive, markersize=6,
                    capsize=2, elinewidth=1, alpha=0.8
                )
                if not comb_llama:
                    for x, y, name in zip(x_inactive, y_inactive, names_inactive):
                        ax_scatter.text(float(x), float(y), name, fontsize=7, color='navy', zorder=10)
                elif comb_llama and use_phangs and use_wis:
                    names_phangs_wis = list(names_phangs) + list(names_wis)
                    shared_names_inactive = [x if x in names_phangs_wis else None for x in names_inactive]
                    for x, y, name in zip(x_inactive, y_inactive, shared_names_inactive):
                        ax_scatter.text(float(x + 0.005), float(y), name, fontsize=7, color='black', zorder=10)
                elif comb_llama and use_phangs and not use_wis:
                    shared_names_inactive = [x if x in names_wis else None for x in names_inactive]
                    for x, y, name in zip(x_inactive, y_inactive, shared_names_inactive):
                        ax_scatter.text(float(x + 0.005), float(y), name, fontsize=7, color='black', zorder=10)
                elif comb_llama and use_wis and not use_phangs:
                    shared_names_inactive = [x if x in names_phangs else None for x in names_inactive]
                    for x, y, name in zip(x_inactive, y_inactive, shared_names_inactive):
                        ax_scatter.text(float(x + 0.005), float(y), name, fontsize=7, color='black', zorder=10)
            
    ############################## Plot comparison samples ##############################

            if soloplot is None and use_gb21:
                ax_scatter.scatter(
                x_gb21, y_gb21,
                marker='o', color='green', label='GB21', s=36, alpha=0.8, edgecolors='none'
                )
                if not comb_llama:
                    for x, y, name in zip(x_gb21, y_gb21, names_gb21):
                        ax_scatter.text(float(x), float(y), name, fontsize=7, color='darkgreen', zorder=10)

            if soloplot is None and use_wis:
                ax_scatter.scatter(
                x_wis, y_wis,
                marker='^', color='purple', label='WIS', s=56, alpha=0.8, edgecolors='none'
                )
                if not comb_llama:
                    for x, y, name in zip(x_wis, y_wis, names_wis):
                        ax_scatter.text(float(x), float(y), name, fontsize=7, color='indigo', zorder=10)
                elif comb_llama and plotshared:
                    names_llama = list(names_agn) + list(names_inactive)
                    for x, y, name in zip(x_wis, y_wis, shared_names_wis):
                        ax_scatter.text(float(x), float(y), name, fontsize=7, color='indigo', zorder=10)

            if soloplot is None and use_phangs:        
                ax_scatter.scatter(
                x_phangs, y_phangs,
                marker='D', color='orange', label='PHANGS', s=36, alpha=0.8, edgecolors='none'
                )
                if not comb_llama:
                    for x, y, name in zip(x_phangs, y_phangs, names_phangs):
                        ax_scatter.text(float(x), float(y), name, fontsize=7, color='darkorange', zorder=10)
                elif comb_llama and plotshared:
                    for x, y, name in zip(x_phangs, y_phangs, shared_names_phangs):
                        ax_scatter.text(float(x), float(y), name, fontsize=7, color='darkorange', zorder=10)

            if soloplot is None and use_sim:
                ax_scatter.scatter(
                x_sim, y_sim,
                marker='X', color='brown', label='Simulations', s=36, alpha=0.8, edgecolors='none'
                )
                if not comb_llama:
                    for x, y, name in zip(x_sim, y_sim, names_sim):
                        ax_scatter.text(float(x), float(y), name, fontsize=7, color='saddlebrown', zorder=10)
                elif comb_llama:
                    
                    shared_names_sim = [x if x in names_llama else None for x in names_sim]
                    for x, y, name in zip(x_sim, y_sim, shared_names_sim):
                        ax_scatter.text(float(x), float(y), name, fontsize=7, color='saddlebrown', zorder=10)

            if use_aux:
                print(x_aux, y_aux, names_aux)
                ax_scatter.scatter(
                x_aux, y_aux,
                marker='*', color='cyan', label='wis/phangs pipeline', s=100, alpha=0.9, edgecolors='black', linewidths=0.5
                )
                for x, y, name in zip(x_aux, y_aux, names_aux):
                    ax_scatter.text(float(x), float(y), name, fontsize=7, color='teal', zorder=10)

            ############################## Add excluded ticks ##############################

            if exclude_names is not None:
                            # For X-axis (horizontal ticks)
                for x_val in excluded_x:
                    ax_scatter.plot([x_val, x_val], [ylower, ylower * 0 + 1e-10], color='gray', marker='|', markersize=10, linestyle='None', alpha=0.7, clip_on=False)

                # For Y-axis (vertical ticks)
                for y_val in excluded_y:
                    ax_scatter.plot([xlower, xlower * 0 + 1e-10], [y_val, y_val], color='gray', marker='_', markersize=10, linestyle='None', alpha=0.7, clip_on=False)
                
                for x_val, name in zip(excluded_x, exclude_names):
                    ax_scatter.text(x_val, ylower, name, fontsize=6, rotation=90, color='gray', verticalalignment='bottom')

            ############################### apply scale and limits ##############################
            if logx:
                ax_scatter.set_xscale("log")
            if logy:
                ax_scatter.set_yscale("log")

            ax_scatter.set_xlim(xlower, xupper)
            ax_scatter.set_ylim(ylower, yupper)

            ############################### Histogram bin edges ##############################

            y_for_bins = all_y[(all_y >= ylower) & (all_y <= yupper)]
            if y_for_bins.empty:
                y_for_bins = all_y
            bin_edges = np.histogram_bin_edges(y_for_bins, bins=7)

            ############################### Scatter labels ###############################
            if x_column == 'log L′ CO':
                ax_scatter.set_xlabel(f"Single-dish log L′ CO (K km s pc$^2$)")
            else:
                ax_scatter.set_xlabel(x_column)
            if y_column == "L'CO_JCMT (K km s pc2)" or y_column == "L'CO_APEX (K km s pc2)":
                ax_scatter.set_ylabel(f"ALMA L′ CO (K km s pc$^2$)")
            else:
                ax_scatter.set_ylabel(y_column)
            ax_scatter.grid(True)
            leg = ax_scatter.legend(loc=legend_loc)
            leg.set_zorder(30)
            ax_scatter.set_title(f'{y_column} vs {x_column}')
            if x_column == 'log L′ CO':
                ax_scatter.set_title(f"ALMA L′ CO vs Single-dish log L′ CO")

            ############################### Standalone histogram ###############################

            # Create output directory
            hist_dir = f"/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/Plots/histograms/{mask}_{R_kpc}kpc/"
            os.makedirs(hist_dir, exist_ok=True)

            fig_hist, ax_hist_only = plt.subplots(figsize=(6, 4))

            # --- AGN ---
            if soloplot in (None, 'AGN') and not comb_llama:
                ax_hist_only.hist(
                    y_agn, bins=bin_edges,
                    color='red', alpha=0.4, label='AGN'
                )
                median_agn = np.median(y_agn)
                ax_hist_only.axvline(median_agn, color='red', linestyle='--')
                ax_hist_only.text(
                    median_agn, ax_hist_only.get_ylim()[1]*0.9,
                    f"{median_agn:.2f}", color='red', fontsize=9, ha='center'
                )

            # --- Inactive ---
            if soloplot in (None, 'inactive') and not comb_llama:
                ax_hist_only.hist(
                    y_inactive, bins=bin_edges,
                    color='blue', alpha=0.4, label='Inactive'
                )
                median_inactive = np.median(y_inactive)
                ax_hist_only.axvline(median_inactive, color='blue', linestyle='--')
                ax_hist_only.text(
                    median_inactive, ax_hist_only.get_ylim()[1]*0.9,
                    f"{median_inactive:.2f}", color='blue', fontsize=9, ha='center'
                )

            # --- Combined LLAMA ---
            if comb_llama:
                combined_y = pd.concat([y_agn, y_inactive])
                ax_hist_only.hist(
                    combined_y, bins=bin_edges,
                    color='black', alpha=0.4, label='LLAMA Galaxies'
                )
                median_combined = np.median(combined_y)
                ax_hist_only.axvline(median_combined, color='black', linestyle='--')
                ax_hist_only.text(
                    median_combined, ax_hist_only.get_ylim()[1]*0.9,
                    f"{median_combined:.2f}", color='black', fontsize=9, ha='center'
                )

            # --- External samples ---
            if soloplot is None and use_gb21:
                ax_hist_only.hist(
                    y_gb21, bins=bin_edges,
                    color='green', alpha=0.4, label='GB21'
                )
                median_gb21 = np.median(y_gb21)
                ax_hist_only.axvline(median_gb21, color='green', linestyle='--')
                ax_hist_only.text(
                    median_gb21, ax_hist_only.get_ylim()[1]*0.9,
                    f"{median_gb21:.2f}", color='green', fontsize=9, ha='center'
                )

            if use_wis:
                ax_hist_only.hist(
                    y_wis, bins=bin_edges,
                    color='purple', alpha=0.4, label='WIS'
                )
                median_wis = np.median(y_wis)
                ax_hist_only.axvline(median_wis, color='purple', linestyle='--')
                ax_hist_only.text(
                    median_wis, ax_hist_only.get_ylim()[1]*0.9,
                    f"{median_wis:.2f}", color='purple', fontsize=9, ha='center'
                )

            if use_phangs:
                ax_hist_only.hist(
                    y_phangs, bins=bin_edges,
                    color='orange', alpha=0.4, label='PHANGS'
                )
                median_phangs = np.median(y_phangs)
                ax_hist_only.axvline(median_phangs, color='orange', linestyle='--')
                ax_hist_only.text(
                    median_phangs, ax_hist_only.get_ylim()[1]*0.9,
                    f"{median_phangs:.2f}", color='orange', fontsize=9, ha='center'
                )

            if use_sim:
                ax_hist_only.hist(
                    y_sim, bins=bin_edges,
                    color='brown', alpha=0.4, label='Simulations'
                )
                median_sim = np.median(y_sim)
                ax_hist_only.axvline(median_sim, color='brown', linestyle='--')
                ax_hist_only.text(
                    median_sim, ax_hist_only.get_ylim()[1]*0.9,
                    f"{median_sim:.2f}", color='brown', fontsize=9, ha='center'
                )

            # Axis labels
            ax_hist_only.set_xlabel(y_column)
            ax_hist_only.set_ylabel("Number of galaxies")

            # Match x-limits to scatter y-range
            ax_hist_only.set_xlim(ylower, yupper)

            ax_hist_only.legend(fontsize=8)
            ax_hist_only.grid(True, axis='y', alpha=0.3)

            # Save
            parts = []
            if soloplot:
                parts.append(f"_{soloplot}")
            if use_gb21:
                parts.append('_gb21')
            if use_wis:
                parts.append('_wis')
            if use_phangs:
                parts.append('_phangs')
            if use_sim:
                parts.append('_sim')
            if comb_llama:
                parts.append('_comb')
            if rebin is not None:
                parts.append(f'_{rebin}pc')
            suffix = ''.join(parts)
            hist_path = (
                f"/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/Plots/"
                f"histograms/{mask}_{R_kpc}kpc/{suffix}_hist_{y_column}.png"
            )

            plt.tight_layout()
            plt.savefig(hist_path, dpi=300)
            plt.close(fig_hist)

            print(f"Saved histogram to: {hist_path}")

        ############################### X histogram bin edges ##############################

            x_for_bins = all_x[(all_x >= xlower) & (all_x <= xupper)]
            if x_for_bins.empty:
                x_for_bins = all_x

            bin_edges_x = np.histogram_bin_edges(x_for_bins, bins=7)

        ############################### Standalone X histogram ###############################

            hist_dir = f"/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/Plots/histograms/{mask}_{R_kpc}kpc/"
            os.makedirs(hist_dir, exist_ok=True)

            fig_hist_x, ax_hist_x = plt.subplots(figsize=(6, 4))

            # --- AGN ---
            if soloplot in (None, 'AGN') and not comb_llama:
                ax_hist_x.hist(
                    x_agn, bins=bin_edges_x,
                    color='red', alpha=0.4, label='AGN'
                )
                median_agn = np.median(x_agn)
                ax_hist_x.axvline(median_agn, color='red', linestyle='--')
                ax_hist_x.text(
                    median_agn, ax_hist_x.get_ylim()[1]*0.9,
                    f"{median_agn:.2f}", color='red', fontsize=9, ha='center'
                )

            # --- Inactive ---
            if soloplot in (None, 'inactive') and not comb_llama:
                ax_hist_x.hist(
                    x_inactive, bins=bin_edges_x,
                    color='blue', alpha=0.4, label='Inactive'
                )
                median_inactive = np.median(x_inactive)
                ax_hist_x.axvline(median_inactive, color='blue', linestyle='--')
                ax_hist_x.text(
                    median_inactive, ax_hist_x.get_ylim()[1]*0.9,
                    f"{median_inactive:.2f}", color='blue', fontsize=9, ha='center'
                )

            # --- Combined LLAMA ---
            if comb_llama:
                combined_x = pd.concat([x_agn, x_inactive])
                ax_hist_x.hist(
                    combined_x, bins=bin_edges_x,
                    color='black', alpha=0.4, label='LLAMA Galaxies'
                )
                median_combined = np.median(combined_x)
                ax_hist_x.axvline(median_combined, color='black', linestyle='--')
                ax_hist_x.text(
                    median_combined, ax_hist_x.get_ylim()[1]*0.9,
                    f"{median_combined:.2f}", color='black', fontsize=9, ha='center'
                )

            # --- External samples ---
            if soloplot is None and use_gb21:
                ax_hist_x.hist(
                    x_gb21, bins=bin_edges_x,
                    color='green', alpha=0.4, label='GB21'
                )
                median_gb21 = np.median(x_gb21)
                ax_hist_x.axvline(median_gb21, color='green', linestyle='--')
                ax_hist_x.text(
                    median_gb21, ax_hist_x.get_ylim()[1]*0.9,
                    f"{median_gb21:.2f}", color='green', fontsize=9, ha='center'
                )

            if use_wis:
                ax_hist_x.hist(
                    x_wis, bins=bin_edges_x,
                    color='purple', alpha=0.4, label='WIS'
                )
                median_wis = np.median(x_wis)
                ax_hist_x.axvline(median_wis, color='purple', linestyle='--')
                ax_hist_x.text(
                    median_wis, ax_hist_x.get_ylim()[1]*0.9,
                    f"{median_wis:.2f}", color='purple', fontsize=9, ha='center'
                )

            if use_phangs:
                ax_hist_x.hist(
                    x_phangs, bins=bin_edges_x,
                    color='orange', alpha=0.4, label='PHANGS'
                )
                median_phangs = np.median(x_phangs)
                ax_hist_x.axvline(median_phangs, color='orange', linestyle='--')
                ax_hist_x.text(
                    median_phangs, ax_hist_x.get_ylim()[1]*0.9,
                    f"{median_phangs:.2f}", color='orange', fontsize=9, ha='center'
                )

            if use_sim:
                ax_hist_x.hist(
                    x_sim, bins=bin_edges,
                    color='brown', alpha=0.4, label='Simulations'
                )
                median_sim = np.median(x_sim)
                ax_hist_x.axvline(median_sim, color='brown', linestyle='--')
                ax_hist_x.text(
                    median_sim, ax_hist_x.get_ylim()[1]*0.9,
                    f"{median_sim:.2f}", color='brown', fontsize=9, ha='center'
                )

            # Axis labels
            ax_hist_x.set_xlabel(x_column)
            ax_hist_x.set_ylabel("Number of galaxies")

            # Match x-limits to scatter
            ax_hist_x.set_xlim(xlower, xupper)

            ax_hist_x.legend(fontsize=8)
            ax_hist_x.grid(True, axis='y', alpha=0.3)

            # Save
            parts = []
            if soloplot:
                parts.append(f"_{soloplot}")
            if use_gb21:
                parts.append('_gb21')
            if use_wis:
                parts.append('_wis')
            if use_phangs:
                parts.append('_phangs')
            if use_sim:
                parts.append('_sim')
            if comb_llama:
                parts.append('_comb')
            if rebin is not None:
                parts.append(f'_{rebin}pc')
            suffix = ''.join(parts)

            hist_x_path = (
                f"/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/Plots/"
                f"histograms/{mask}_{R_kpc}kpc/{suffix}_hist_{x_column}.png"
            )

            plt.tight_layout()
            plt.savefig(hist_x_path, dpi=300)
            plt.close(fig_hist_x)

            print(f"Saved X histogram to: {hist_x_path}")

        ############################### Histogram subplot ##############################
            ax_hist = fig.add_subplot(gs[1], sharey=ax_scatter)

            if soloplot in (None, 'AGN') and not comb_llama:
                ax_hist.hist(y_agn, bins=bin_edges, orientation='horizontal', 
                            color='red', alpha=0.4, label='AGN')
                median_agn = np.median(y_agn)
                ax_hist.axhline(median_agn, color='red', linestyle='--')
                ax_hist.text(ax_hist.get_xlim()[1]*0.7, median_agn, 
                            f"{median_agn:.2f}", color='red', fontsize=8, va='center')

            if soloplot in (None, 'inactive') and not comb_llama:
                ax_hist.hist(y_inactive, bins=bin_edges, orientation='horizontal', 
                            color='blue', alpha=0.4, label='Inactive')
                median_inactive = np.median(y_inactive)
                ax_hist.axhline(median_inactive, color='blue', linestyle='--')
                ax_hist.text(ax_hist.get_xlim()[1]*0.7, median_inactive, 
                            f"{median_inactive:.2f}", color='blue', fontsize=8, va='center')
                
            if comb_llama:
                combined_y = pd.concat([y_agn, y_inactive])
                ax_hist.hist(combined_y, bins=bin_edges, orientation='horizontal', 
                            color='black', alpha=0.4, label='LLAMA Galaxies')
                median_combined = np.median(combined_y)
                ax_hist.axhline(median_combined, color='black', linestyle='--')
                ax_hist.text(ax_hist.get_xlim()[1]*0.7, median_combined, 
                            f"{median_combined:.2f}", color='black', fontsize=8, va='center')

            if soloplot is None and use_gb21:
                ax_hist.hist(y_gb21, bins=bin_edges, orientation='horizontal', 
                            color='green', alpha=0.4, label='GB21')
                median_gb21 = np.median(y_gb21)
                ax_hist.axhline(median_gb21, color='green', linestyle='--')
                ax_hist.text(ax_hist.get_xlim()[1]*0.7, median_gb21, 
                            f"{median_gb21:.2f}", color='green', fontsize=8, va='center')

            if use_wis:
                ax_hist.hist(y_wis, bins=bin_edges, orientation='horizontal', 
                            color='purple', alpha=0.4, label='WIS')
                median_wis = np.median(y_wis)
                ax_hist.axhline(median_wis, color='purple', linestyle='--')
                ax_hist.text(ax_hist.get_xlim()[1]*0.7, median_wis, 
                            f"{median_wis:.2f}", color='purple', fontsize=8, va='center')
            if use_phangs:
                ax_hist.hist(y_phangs, bins=bin_edges, orientation='horizontal', 
                            color='orange', alpha=0.4, label='PHANGS')
                median_phangs = np.median(y_phangs)
                ax_hist.axhline(median_phangs, color='orange', linestyle='--')
                ax_hist.text(ax_hist.get_xlim()[1]*0.7, median_phangs, 
                            f"{median_phangs:.2f}", color='orange', fontsize=8, va='center')
            if use_sim:
                ax_hist.hist(y_sim, bins=bin_edges, orientation='horizontal', 
                            color='brown', alpha=0.4, label='Simulations')
                median_sim = np.median(y_sim)
                ax_hist.axhline(median_sim, color='brown', linestyle='--')
                ax_hist.text(ax_hist.get_xlim()[1]*0.7, median_sim, 
                            f"{median_sim:.2f}", color='brown', fontsize=8, va='center')

            ax_hist.axis('off')

            ############################### Save ###############################
            parts = []
            if soloplot:
                parts.append(f"_{soloplot}")
            if use_gb21:
                parts.append('_gb21')
            if use_wis:
                parts.append('_wis')
            if use_phangs:
                parts.append('_phangs')
            if use_sim:
                parts.append('_sim')
            if comb_llama:
                parts.append('_comb')
            if rebin is not None:
                parts.append(f'_{rebin}pc')
            suffix = ''.join(parts)
            outputdir = f'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/Plots/{mask}_{R_kpc}kpc/'
            os.makedirs(outputdir, exist_ok=True)
            output_path = f'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/Plots/{mask}_{R_kpc}kpc/{suffix}_{x_column}_vs_{y_column}.png'
            if x_column == 'log L′ CO':
                output_path = f'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/Plots/{mask}_{R_kpc}kpc/{suffix}_LCO_singledish_vs_LCO_ALMA.png'

            plt.savefig(output_path)
            print(f"Saved plot to: {output_path}")
            plt.close(fig)

            ############################## Matched-pairs difference histogram ##############################

            diffs = []
            valid_pairs = 0

            df_pairs['Active Galaxy'] = normalize_name(df_pairs['Active Galaxy'])
            df_pairs['Inactive Galaxy'] = normalize_name(df_pairs['Inactive Galaxy'])

            for _, row in df_pairs.iterrows():
                agn_name = row["Active Galaxy"].strip()
                inactive_name = row["Inactive Galaxy"].strip()

                # Extract rows for each galaxy
                agn_rows = merged_AGN[merged_AGN["Name_clean"] == agn_name]
                inactive_rows = merged_inactive[merged_inactive["Name_clean"] == inactive_name]

                if agn_rows.empty or inactive_rows.empty:
                    continue

                # =========================
                # NATIVE RESOLUTION MODE
                # =========================
                if nativey:

                    # Select best (smallest) resolution row for each galaxy
                    agn_row = (
                        agn_rows
                        .sort_values("Resolution (pc)")
                        .iloc[0]
                    )
                    inactive_row = (
                        inactive_rows
                        .sort_values("Resolution (pc)")
                        .iloc[0]
                    )

                    val_agn = agn_row[y_column]
                    val_inactive = inactive_row[y_column]

                    if not (np.isfinite(val_agn) and np.isfinite(val_inactive)):
                        continue

                    diff = float(val_agn) - float(val_inactive)
                    diffs.append(diff)
                    valid_pairs += 1

                    continue  # move to next pair

                # =========================
                # MATCHED RESOLUTION MODE
                # =========================
                for _, agn_row in agn_rows.iterrows():
                    res_agn = agn_row["Resolution (pc)"]
                    val_agn = agn_row[y_column]

                    if not np.isfinite(val_agn):
                        continue

                    # Find inactive rows at the SAME resolution
                    match_inactive = inactive_rows[
                        inactive_rows["Resolution (pc)"] == res_agn
                    ]

                    if match_inactive.empty:
                        continue

                    inactive_row = match_inactive.iloc[0]
                    val_inactive = inactive_row[y_column]

                    if not np.isfinite(val_inactive):
                        continue

                    diff = float(val_agn) - float(val_inactive)
                    diffs.append(diff)
                    valid_pairs += 1

                    # Stop after first valid match for this pair
                    break

            diffs = np.asarray(diffs)
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
            outputdir = f'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/Plots/pair_diffs/{mask}_{R_kpc}kpc/'
            os.makedirs(outputdir, exist_ok=True)
            output_path = f'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/Plots/pair_diffs/{mask}_{R_kpc}kpc/{y_column}_pair_differences.png'
            plt.savefig(output_path)
            print(f"Saved matched-pairs plot to: {output_path}")
            plt.close(fig)

########################################## Categorical X or Y axis handling ##########################################

        elif is_x_categorical or is_y_categorical:
            figsize = 8
            if truescale == True:
                if manual_limits is not None:
                    xratio = (manual_limits[1] - manual_limits[0]) / (manual_limits[3] - manual_limits[2]) * 1.3
                    yratio = (manual_limits[3] - manual_limits[2]) / (manual_limits[1] - manual_limits[0])
                    fig = plt.figure(figsize=(figsize * xratio, figsize * yratio),
                            constrained_layout=True)
            else:
                # fig = plt.figure(figsize=((figsize*1.1)*1.3, figsize*0.92))
                fig = plt.figure(
                            figsize=((figsize*1.1)*1.3, figsize*0.92)
                        )
                fig.subplots_adjust(
                    left=0,   # ← tighten this (0.10–0.14 usually ideal)
                    right=0.98,
                    bottom=0.12,
                    top=0.95,
                    wspace=0.05
                )
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
            if soloplot is None and use_wis:
                data_x.append(x_wis)
                data_y.append(y_wis)
            if soloplot is None and use_phangs:
                data_x.append(x_phangs)
                data_y.append(y_phangs)
            if soloplot is None and use_sim:
                data_x.append(x_sim)
                data_y.append(y_sim)

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
                
                if soloplot is None and use_wis:
                    ax_hist.hist(y_wis, bins=bin_edges, orientation='horizontal', 
                                color='purple', alpha=0.4, label='WIS')
                    median_wis = np.median(y_wis)
                    ax_hist.axhline(median_wis, color='purple', linestyle='--')
                    ax_hist.text(ax_hist.get_xlim()[1]*0.7, median_wis, 
                                f"{median_wis:.2f}", color='purple', fontsize=8, va='center')
                if soloplot is None and use_phangs:
                    ax_hist.hist(y_phangs, bins=bin_edges, orientation='horizontal', 
                                color='orange', alpha=0.4, label='PHANGS')
                    median_phangs = np.median(y_phangs)
                    ax_hist.axhline(median_phangs, color='orange', linestyle='--')
                    ax_hist.text(ax_hist.get_xlim()[1]*0.7, median_phangs, 
                                f"{median_phangs:.2f}", color='orange', fontsize=8, va='center')
                if soloplot is None and use_sim:
                    ax_hist.hist(y_sim, bins=bin_edges, orientation='horizontal', 
                                color='brown', alpha=0.4, label='Simulations')
                    median_sim = np.median(y_sim)
                    ax_hist.axhline(median_sim, color='brown', linestyle='--')
                    ax_hist.text(ax_hist.get_xlim()[1]*0.7, median_sim, 
                                f"{median_sim:.2f}", color='brown', fontsize=8, va='center')

                ax_hist.axis('off')

                        # Save
                parts = []
                if soloplot:
                    parts.append(f"_{soloplot}")
                if use_gb21:
                    parts.append('_gb21')
                if use_wis:
                    parts.append('_wis')
                if use_phangs:
                    parts.append('_phangs')
                if use_sim:
                    parts.append('_sim')
                if comb_llama:
                    parts.append('_comb')
                suffix = ''.join(parts)
                outputdir = f'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/Plots/{mask}_{R_kpc}kpc/'
                os.makedirs(outputdir, exist_ok=True)
                output_path = f'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/Plots/{mask}_{R_kpc}kpc/{suffix}{x_column}_vs_{y_column}.png'
                plt.savefig(output_path)
                print(f"Saved plot to: {output_path}")
                plt.close(fig)  


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
                outputdir = f'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/Plots/{mask}_{R_kpc}kpc/'
                os.makedirs(outputdir, exist_ok=True)
                output_path = f'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/Plots/{mask}_{R_kpc}kpc/{suffix}_{x_column}_vs_{y_column}.png'
                plt.savefig(output_path)
                print(f"Saved plot to: {output_path}")
                plt.close(fig)

    



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
    19: "NGC 1375",
    20: "NGC 1315",
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
    "MCG-05-23-016": [5, 19],
    "ESO 137-G034": [13],
    "NGC 2992": [2, 15, 16, 17, 18],
    "NGC 4235": [2, 16, 17, 18],
    "NGC 4593": [1, 8],
    "NGC 7172": [15, 16, 17],
    "NGC 3783": [10],
    "ESO 021-G004": [17],
    "NGC 5728": [13, 17],
    "MCG-05-14-012": [20],
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
print(len(df_pairs), "matched pairs constructed.")



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
    {"Name": "MCG-05-23-016", "log L′ CO": "7.445",      "log LGAL": "41.28", "log LAGN": "43.70", "log LX": "43.16", "log NH": "22.2", "log LK,AGN": "43.00"},
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
    {"Name": "ESO 208-G021", "log L′ CO": "6.34",      "log LGAL": "41.93", "log LAGN": "41.3"},
    {"Name": "IC 4653",      "log L′ CO": "7.642", "log LGAL": "43.15", "log LAGN": "41.7"},
    {"Name": "NGC 1079",     "log L′ CO": "6.610", "log LGAL": "42.51", "log LAGN": "39.6"},
    {"Name": "NGC 1947",     "log L′ CO": "7.888", "log LGAL": "42.72", "log LAGN": "39.5"},
    {"Name": "NGC 2775",     "log L′ CO": "6.520",      "log LGAL": "43.32", "log LAGN": "40.6"},
    {"Name": "NGC 3175",     "log L′ CO": "8.040", "log LGAL": "43.68", "log LAGN": "40.0"},
    {"Name": "NGC 3351",     "log L′ CO": "7.911", "log LGAL": "43.55", "log LAGN": "39.4"},
    {"Name": "NGC 3717",     "log L′ CO": "8.489", "log LGAL": "43.96", "log LAGN": "40.7"},
    {"Name": "NGC 3749",     "log L′ CO": "8.691", "log LGAL": "43.86", "log LAGN": "40.7"},
    {"Name": "NGC 4224",     "log L′ CO": "8.086", "log LGAL": "42.93", "log LAGN": "42.0"},
    {"Name": "NGC 4254",     "log L′ CO": "8.134", "log LGAL": "44.84", "log LAGN": "40.5"},
    {"Name": "NGC 4260",     "log L′ CO": "7.258",      "log LGAL": "42.35", "log LAGN": "40.8"},
    {"Name": "NGC 5037",     "log L′ CO": "8.328", "log LGAL": "43.06", "log LAGN": "40.1"},
    {"Name": "NGC 5845",     "log L′ CO": "6.999",      "log LGAL": "41.69", "log LAGN": "40.9"},
    {"Name": "NGC 5921",     "log L′ CO": "7.960", "log LGAL": "43.40", "log LAGN": "40.7"},
    {"Name": "NGC 718",      "log L′ CO": "7.262", "log LGAL": "42.66", "log LAGN": "38.8"},
    {"Name": "NGC 7727",     "log L′ CO": "7.449", "log LGAL": "42.56", "log LAGN": "41.2"},
    {"Name": "NGC 1375",     "log L′ CO": "NaN", "log LGAL": "NaN", "log LAGN": "NaN"}
])

Rosario2018_obs = [
    # ---------------- AGN ----------------
    dict(Name="ESO 021-G004", Distance_Mpc=39, AGN_type="2", Telescope="APEX",
         ICO_Kkms=4.5, ICO_err=1.0, ICO_ul=False,
         SCO_Jykms=132, SCO_err=15, SCO_ul=False),

    dict(Name="ESO 137-G034", Distance_Mpc=35, AGN_type="2", Telescope="APEX",
         ICO_Kkms=3.0, ICO_err=0.0, ICO_ul=False,
         SCO_Jykms=89, SCO_err=10, SCO_ul=False),

    dict(Name="MCG-05-23-016", Distance_Mpc=35, AGN_type="1i", Telescope="APEX",
         ICO_Kkms=1.3, ICO_err=np.nan, ICO_ul=True,
         SCO_Jykms=38, SCO_err=np.nan, SCO_ul=True),

    dict(Name="MCG-06-30-015", Distance_Mpc=27, AGN_type="1.2", Telescope="APEX",
         ICO_Kkms=1.0, ICO_err=0.0, ICO_ul=False,
         SCO_Jykms=29, SCO_err=5, SCO_ul=False),

    dict(Name="NGC 1365", Distance_Mpc=18, AGN_type="1.8", Telescope="SEST",
         ICO_Kkms=150.0, ICO_err=10.0, ICO_ul=False,
         SCO_Jykms=3075, SCO_err=205, SCO_ul=False),

    dict(Name="NGC 2110", Distance_Mpc=34, AGN_type="1i", Telescope="JCMT",
         ICO_Kkms=3.4, ICO_err=0.0, ICO_ul=False,
         SCO_Jykms=57, SCO_err=6, SCO_ul=False),

    dict(Name="NGC 2992", Distance_Mpc=36, AGN_type="1i", Telescope="JCMT",
         ICO_Kkms=22.4, ICO_err=2.0, ICO_ul=False,
         SCO_Jykms=377, SCO_err=28, SCO_ul=False),

    dict(Name="NGC 3081", Distance_Mpc=34, AGN_type="1h", Telescope="JCMT",
         ICO_Kkms=4.8, ICO_err=1.0, ICO_ul=False,
         SCO_Jykms=80, SCO_err=17, SCO_ul=False),

    dict(Name="NGC 3783", Distance_Mpc=38, AGN_type="1.5", Telescope="APEX",
         ICO_Kkms=3.5, ICO_err=0.0, ICO_ul=False,
         SCO_Jykms=103, SCO_err=12, SCO_ul=False),

    dict(Name="NGC 4235", Distance_Mpc=37, AGN_type="1.2", Telescope="APEX",
         ICO_Kkms=2.3, ICO_err=0.0, ICO_ul=False,
         SCO_Jykms=68, SCO_err=9, SCO_ul=False),

    dict(Name="NGC 4388", Distance_Mpc=25, AGN_type="1h", Telescope="JCMT",
         ICO_Kkms=22.3, ICO_err=2.0, ICO_ul=False,
         SCO_Jykms=374, SCO_err=29, SCO_ul=False),

    dict(Name="NGC 4593", Distance_Mpc=37, AGN_type="1.0", Telescope="JCMT",
         ICO_Kkms=10.0, ICO_err=2.0, ICO_ul=False,
         SCO_Jykms=168, SCO_err=28, SCO_ul=False),

    dict(Name="NGC 5506", Distance_Mpc=27, AGN_type="1i", Telescope="JCMT",
         ICO_Kkms=10.1, ICO_err=1.0, ICO_ul=False,
         SCO_Jykms=169, SCO_err=20, SCO_ul=False),

    dict(Name="NGC 5728", Distance_Mpc=39, AGN_type="2", Telescope="JCMT",
         ICO_Kkms=21.9, ICO_err=2.0, ICO_ul=False,
         SCO_Jykms=368, SCO_err=41, SCO_ul=False),

    dict(Name="NGC 6814", Distance_Mpc=23, AGN_type="1.5", Telescope="JCMT",
         ICO_Kkms=5.7, ICO_err=1.0, ICO_ul=False,
         SCO_Jykms=96, SCO_err=22, SCO_ul=False),

    dict(Name="NGC 7172", Distance_Mpc=37, AGN_type="1i", Telescope="APEX",
         ICO_Kkms=18.7, ICO_err=1.0, ICO_ul=False,
         SCO_Jykms=548, SCO_err=28, SCO_ul=False),

    dict(Name="NGC 7213", Distance_Mpc=25, AGN_type="1", Telescope="APEX",
         ICO_Kkms=8.2, ICO_err=1.0, ICO_ul=False,
         SCO_Jykms=240, SCO_err=16, SCO_ul=False),

    dict(Name="NGC 7582", Distance_Mpc=22, AGN_type="1i", Telescope="APEX",
         ICO_Kkms=95.8, ICO_err=3.0, ICO_ul=False,
         SCO_Jykms=2803, SCO_err=83, SCO_ul=False),

    # ---------------- Inactive ----------------
    dict(Name="ESO 093-G003", Distance_Mpc=22, AGN_type=None, Telescope="APEX",
         ICO_Kkms=19.9, ICO_err=1.0, ICO_ul=False,
         SCO_Jykms=581, SCO_err=38, SCO_ul=False),

    dict(Name="ESO 208-G021", Distance_Mpc=17, AGN_type=None, Telescope="APEX",
         ICO_Kkms=0.4, ICO_err=np.nan, ICO_ul=True,
         SCO_Jykms=12, SCO_err=np.nan, SCO_ul=True),

    dict(Name="IC 4653", Distance_Mpc=26, AGN_type=None, Telescope="APEX",
         ICO_Kkms=3.6, ICO_err=0.0, ICO_ul=False,
         SCO_Jykms=107, SCO_err=9, SCO_ul=False),

    dict(Name="NGC 1079", Distance_Mpc=19, AGN_type=None, Telescope="APEX",
         ICO_Kkms=0.6, ICO_err=0.0, ICO_ul=False,
         SCO_Jykms=19, SCO_err=5, SCO_ul=False),

    dict(Name="NGC 1947", Distance_Mpc=19, AGN_type=None, Telescope="APEX",
         ICO_Kkms=12.0, ICO_err=1.0, ICO_ul=False,
         SCO_Jykms=351, SCO_err=26, SCO_ul=False),

    dict(Name="NGC 2775", Distance_Mpc=21, AGN_type=None, Telescope="APEX",
         ICO_Kkms=0.4, ICO_err=np.nan, ICO_ul=True,
         SCO_Jykms=12, SCO_err=np.nan, SCO_ul=True),

    dict(Name="NGC 3175", Distance_Mpc=14, AGN_type=None, Telescope="APEX",
         ICO_Kkms=31.4, ICO_err=1.0, ICO_ul=False,
         SCO_Jykms=918, SCO_err=23, SCO_ul=False),

    dict(Name="NGC 3351", Distance_Mpc=11, AGN_type=None, Telescope="APEX",
         ICO_Kkms=37.7, ICO_err=1.0, ICO_ul=False,
         SCO_Jykms=1104, SCO_err=22, SCO_ul=False),

    dict(Name="NGC 3717", Distance_Mpc=24, AGN_type=None, Telescope="APEX",
         ICO_Kkms=30.1, ICO_err=1.0, ICO_ul=False,
         SCO_Jykms=881, SCO_err=29, SCO_ul=False),

    dict(Name="NGC 3749", Distance_Mpc=42, AGN_type=None, Telescope="APEX",
         ICO_Kkms=15.7, ICO_err=1.0, ICO_ul=False,
         SCO_Jykms=459, SCO_err=34, SCO_ul=False),

    dict(Name="NGC 4224", Distance_Mpc=41, AGN_type=None, Telescope="APEX",
         ICO_Kkms=4.1, ICO_err=0.0, ICO_ul=False,
         SCO_Jykms=120, SCO_err=11, SCO_ul=False),

    dict(Name="NGC 4254", Distance_Mpc=15, AGN_type=None, Telescope="APEX",
         ICO_Kkms=34.1, ICO_err=1.0, ICO_ul=False,
         SCO_Jykms=998, SCO_err=30, SCO_ul=False),

    dict(Name="NGC 4260", Distance_Mpc=31, AGN_type=None, Telescope="APEX",
         ICO_Kkms=1.1, ICO_err=np.nan, ICO_ul=True,
         SCO_Jykms=31, SCO_err=np.nan, SCO_ul=True),

    dict(Name="NGC 5037", Distance_Mpc=35, AGN_type=None, Telescope="APEX",
         ICO_Kkms=9.8, ICO_err=1.0, ICO_ul=False,
         SCO_Jykms=286, SCO_err=27, SCO_ul=False),

    dict(Name="NGC 5845", Distance_Mpc=25, AGN_type=None, Telescope="APEX",
         ICO_Kkms=0.9, ICO_err=np.nan, ICO_ul=True,
         SCO_Jykms=26, SCO_err=np.nan, SCO_ul=True),

    dict(Name="NGC 5921", Distance_Mpc=21, AGN_type=None, Telescope="APEX",
         ICO_Kkms=11.6, ICO_err=0.0, ICO_ul=False,
         SCO_Jykms=340, SCO_err=14, SCO_ul=False),

    dict(Name="NGC 718", Distance_Mpc=23, AGN_type=None, Telescope="APEX",
         ICO_Kkms=1.9, ICO_err=0.0, ICO_ul=False,
         SCO_Jykms=57, SCO_err=4, SCO_ul=False),

    dict(Name="NGC 7727", Distance_Mpc=26, AGN_type=None, Telescope="APEX",
         ICO_Kkms=2.3, ICO_err=0.0, ICO_ul=False,
         SCO_Jykms=68, SCO_err=8, SCO_ul=False),
]


 #see archived_comp_samp_build for rebuilding PHANGS WIS SIM GB21


 #   """posible x_column: '"Distance (Mpc)"', 'log LH (L⊙)', 'Hubble Stage', 'Axis Ratio', 'Bar'
 #      posible y_column: 'Smoothness', 'Asymmetry', 'Gini', 'Sigma0', 'rs'"""



#masks = ['broad', 'strict','flux90_strict']
masks = ['strict']
radii = [1, 1.5, 0.3]

for mask in masks:
    for R_kpc in radii:
        print(f"Running plots for mask={mask}, R_kpc={R_kpc}")

    # ############ CAS with stellar mass #############

    #     plot_llama_property('log LH (L⊙)', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('log LH (L⊙)', 'Asymmetry', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('log LH (L⊙)', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('log LH (L⊙)','Concentration',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('log LH (L⊙)','clumping_factor',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'])

    #     # ############# CAS with Hubble Stage #############

    #     plot_llama_property('Hubble Stage', 'Concentration', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('Hubble Stage', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('Hubble Stage', 'Asymmetry', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('Hubble Stage', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('Hubble Stage','clumping_factor',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'])#,exclude_names=['NGC 2775','NGC 4260','ESO 208-G021','NGC 5845','NGC 2992','NGC 1079','NGC 4388'])

    #     # ############# CAS with X-ray luminosity #############

    #     plot_llama_property('log LX', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('log LX', 'Asymmetry', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('log LX', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('log LX','Concentration',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=True,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('log LX','clumping_factor',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'])


    #     plot_llama_property('log LX', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=False,soloplot='inactive',mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('log LX', 'Asymmetry', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=False,soloplot='inactive',mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('log LX', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=False,soloplot='inactive',mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     # plot_llama_property('log LX','Concentration',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=True,soloplot='inactive',mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('log LX','clumping_factor',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=False,soloplot='inactive',mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'])

    #     # using GB24 for concentration

    #     plot_llama_property('log LX','Concentration',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=True,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])

    #     # ############## CAS with eachother #############

    #     plot_llama_property('Gini', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('Asymmetry', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('Asymmetry', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('Gini', 'clumping_factor', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'])
    #     plot_llama_property('clumping_factor', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'])
    #     plot_llama_property('Asymmetry', 'clumping_factor', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'])

    #     # ############## CAS with concentration #############

    #     plot_llama_property('Gini', 'Concentration', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018, False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('Asymmetry', 'Concentration', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('Smoothness', 'Concentration', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('clumping_factor', 'Concentration', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'])

    #     # ############### CAS with resolution #############

    #     plot_llama_property('Resolution (pc)', 'Concentration', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('Resolution (pc)', 'Asymmetry', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('Resolution (pc)', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('Resolution (pc)', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('Resolution (pc)', 'total_mass (M_sun)', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,mask=mask,R_kpc=R_kpc,exclude_names=None)
    #     plot_llama_property('Resolution (pc)', 'clumping_factor', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,mask=mask,R_kpc=R_kpc,logx=False,logy=True,background_image='/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/Leroy2013_plots/Clumping.png',manual_limits=[0,500,1,200],legend_loc='center right',exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'])
    #     plot_llama_property('Resolution (pc)', 'Resolution (pc)', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,mask=mask,R_kpc=R_kpc,exclude_names=None)

    #     # ############### CAS with Gas mass #############

        plot_llama_property('total_mass (M_sun)', 'Concentration', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,logx=True,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'],nativey=True)#,exclude_names=['NGC 1365'])
    #     plot_llama_property('total_mass (M_sun)', 'Asymmetry', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,logx=True,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('total_mass (M_sun)', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,logx=True,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('total_mass (M_sun)', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,logx=True,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1311','NGC 2775','NGC 4260'])
    #     plot_llama_property('total_mass (M_sun)', 'clumping_factor', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,logx=True,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'])

    #     # ############### Clumping factor plot #############

    #     plot_llama_property('area_weighted_sd','mass_weighted_sd',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,logx=True,logy=True,background_image='/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/Leroy2013_plots/Sigma.png',manual_limits=[0.5,5000,0.5,5000], truescale=True,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'])

    #     # ############### CAS with Bar #############

        plot_llama_property('Bar', 'Concentration', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'],nativey=True)
    #     plot_llama_property('Bar', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('Bar', 'Asymmetry', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('Bar', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
    #     plot_llama_property('Bar','clumping_factor',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])

        ############## L'CO comparison with Ros18 #####################

        plot_llama_property('log L′ CO',"L'CO_JCMT (K km s pc2)",AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,logy=True,square=True,best_fit=True,mask=mask,R_kpc=R_kpc,exclude_names=None)


    ############### CAS WISDOM, PHANGS coplot   #############

        if R_kpc == 1.5:
            plot_llama_property('Gini', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,use_wis=True,use_phangs=True,use_sim=False,comb_llama=True,rebin=120,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375'],use_aux=True)
            plot_llama_property('Asymmetry', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,use_wis=True,use_phangs=True,use_sim=False,comb_llama=True,rebin=120,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375'],use_aux=True)
            plot_llama_property('Asymmetry', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,use_wis=True,use_phangs=True,use_sim=False,comb_llama=True,rebin=120,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375'],use_aux=True) 

            plot_llama_property('Distance (Mpc)', 'log LH (L⊙)', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=False, use_wis=True, use_phangs=True, use_sim=False, comb_llama=True, plotshared=False, rebin=120, mask=mask, R_kpc=R_kpc, exclude_names=['NGC 1375'])
            plot_llama_property('Distance (Mpc)', 'Hubble Stage', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=False, use_wis=True, use_phangs=True, use_sim=False, comb_llama=True,plotshared=False, rebin=120, mask=mask, R_kpc=R_kpc, exclude_names=['NGC 1375'])
            plot_llama_property('Hubble Stage', 'log LH (L⊙)', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=False, use_wis=True, use_phangs=True, use_sim=False, comb_llama=True,plotshared=False, rebin=120, mask=mask, R_kpc=R_kpc, exclude_names=['NGC 1375'])

        # plot_llama_property('log LX', 'cont_power_jy', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=False,use_wis=False,use_phangs=False,use_sim=False,comb_llama=False,rebin=None,mask=mask,R_kpc=R_kpc,exclude_names=None,use_aux=False,use_cont=True,soloplot='AGN')
        # plot_llama_property('log LH (L⊙)', 'cont_power_jy', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=False,use_wis=False,use_phangs=False,use_sim=False,comb_llama=False,rebin=None,mask=mask,R_kpc=R_kpc,exclude_names=None,use_aux=False,use_cont=True)
        # plot_llama_property('Axis Ratio', 'cont_power_jy', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=False,use_wis=False,use_phangs=False,use_sim=False,comb_llama=False,rebin=None,mask=mask,R_kpc=R_kpc,exclude_names=None,use_aux=False,use_cont=True)
        # plot_llama_property('Concentration', 'cont_power_jy', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,use_gb21=False,use_wis=False,use_phangs=False,use_sim=False,comb_llama=False,rebin=None,mask=mask,R_kpc=R_kpc,exclude_names=None,use_aux=False,use_cont=True)
             
#         ###### compare on same axis ######

# plot_llama_property('Gini', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False, exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'],comb_llama=True,compare=True,which_compare=[['strict','120pc_strict','120pc_flux90_strict'],[1.5]])
# plot_llama_property('Asymmetry', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'],comb_llama=True,compare=True,which_compare=[['strict','120pc_strict','120pc_flux90_strict'],[1.5]])
# plot_llama_property('Asymmetry', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'],comb_llama=True,compare=True,which_compare=[['strict','120pc_strict','120pc_flux90_strict'],[1.5]])
# plot_llama_property('Gini', 'clumping_factor', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'],comb_llama=True,compare=True,which_compare=[['strict','120pc_strict','120pc_flux90_strict'],[1.5]])
# plot_llama_property('clumping_factor', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'],comb_llama=True,compare=True,which_compare=[['strict','120pc_strict','120pc_flux90_strict'],[1.5]])
# plot_llama_property('Asymmetry', 'clumping_factor', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'],comb_llama=True,compare=True,which_compare=[['strict','120pc_strict','120pc_flux90_strict'],[1.5]])

# plot_llama_property('Gini', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False, exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'],comb_llama=True,compare=True,which_compare=[['strict','broad'],[1.5]])
# plot_llama_property('Asymmetry', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'],comb_llama=True,compare=True,which_compare=[['strict','broad'],[1.5]])
# plot_llama_property('Asymmetry', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'],comb_llama=True,compare=True,which_compare=[['strict','broad'],[1.5]])
# plot_llama_property('Gini', 'clumping_factor', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'],comb_llama=True,compare=True,which_compare=[['strict','broad'],[1.5]])
# plot_llama_property('clumping_factor', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'],comb_llama=True,compare=True,which_compare=[['strict','broad'],[1.5]])
# plot_llama_property('Asymmetry', 'clumping_factor', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,False,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'],comb_llama=True,compare=True,which_compare=[['strict','broad'],[1.5]])


#         compare_masks = ['strict', 'broad','flux90_strict','flux90_broad','120pc_flux90_strict', '120pc_flux90_broad','120pc_strict', '120pc_broad']
  #      compare_radii = [0.3, 1, 1.5]

    ############## galaxy properties WISDOM, PHANGS coplot #############

# nothing considered useful yet

 #   """posible x_column: '"Distance (Mpc)"', 'log LH (L⊙)', 'Hubble Stage', 'Axis Ratio', 'Bar'
 #      posible y_column: 'Smoothness', 'Asymmetry', 'Gini', 'Sigma0', 'rs'"""




stats_table = pd.DataFrame(stats_rows)
stats_table.to_csv(
    "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/Plots/statistics_summary.csv",
    index=False
)
print("Saved all statistics → statistics_summary.csv")

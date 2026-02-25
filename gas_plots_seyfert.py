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

stats_table = pd.DataFrame()

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

def plot_llama_property(x_column: str, y_column: str, AGN_data, inactive_data, agn_bol, inactive_bol, GB21, wis, sim, phangs, use_gb21=False, soloplot=None, exclude_names=None, logx=False, logy=False,
                        background_image=None, manual_limits=None, legend_loc='best', truescale=False, use_wis=False, use_phangs=False, use_sim=False,comb_llama=False,plotshared=True,rebin=None,mask=None,R_kpc=1,compare=False):
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
    llamatab = Table.read('/data/c3040163/llama/llama_main_properties.fits', format='fits')

    base_AGN = "/data/c3040163/llama/alma/gas_analysis_results/AGN"
    base_inactive = "/data/c3040163/llama/alma/gas_analysis_results/inactive"

    base_AGN = "/data/c3040163/llama/alma/gas_analysis_results/AGN"
    base_inactive = "/data/c3040163/llama/alma/gas_analysis_results/inactive"

    default_AGN = f"{base_AGN}/gas_analysis_summary.csv"
    default_inactive = f"{base_inactive}/gas_analysis_summary.csv"

    ################## comparing different masks/radii ########################

    if (compare and use_phangs) or (compare and use_sim) or (compare and use_wis) or (compare and use_gb21) or (compare and not comb_llama):
        print('compare not compatible with use_phangs/use_sim/use_wis')
        # Handle comparison with other datasets
        pass

    if compare and comb_llama and not use_phangs and not use_sim and not use_wis and not use_gb21:
        import itertools

        compare_masks = ['strict', 'broad']
        compare_radii = [0.3, 1, 1.5]
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        markers = ['o', 's', '^', 'D', 'v', 'P']

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

            if not os.path.exists(path_AGN):
                path_AGN = default_AGN
            if not os.path.exists(path_inactive):
                path_inactive = default_inactive

            fit_data_AGN = pd.read_csv(path_AGN)
            fit_data_inactive = pd.read_csv(path_inactive)

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
            inactive_bol = inactive_bol.copy()
            inactive_bol.loc[:, 'log LAGN'] = pd.to_numeric(inactive_bol['log LAGN'], errors='coerce')
            inactive_bol['log LX'] = inactive_bol['log LAGN'].apply(
                lambda log_LAGN: math.log10((12.76 * (1 + (log_LAGN - math.log10(3.82e33)) / 12.15)**18.78) * 3.82e33)
            )

            # Merge with fit data
            merged_AGN = pd.merge(df_combined, fit_data_AGN, left_on='id', right_on='Galaxy_clean',how='right')
            merged_AGN = pd.merge(merged_AGN, agn_bol, left_on='Name_clean', right_on='Name_clean',how='left')
            merged_inactive = pd.merge(df_combined, fit_data_inactive, left_on='id', right_on='Galaxy_clean',how='right')
            merged_inactive = pd.merge(merged_inactive, inactive_bol, left_on='Name_clean', right_on='Name_clean',how='left')

                # Clean AGN data
            merged_AGN[x_column] = pd.to_numeric(merged_AGN[x_column], errors='coerce')
            merged_AGN[y_column] = pd.to_numeric(merged_AGN[y_column], errors='coerce')
            merged_AGN_clean = merged_AGN.dropna(subset=[x_column, y_column])

            # Clean inactive data
            merged_inactive[x_column] = pd.to_numeric(merged_inactive[x_column], errors='coerce')
            merged_inactive[y_column] = pd.to_numeric(merged_inactive[y_column], errors='coerce')
            merged_inactive_clean = merged_inactive.dropna(subset=[x_column, y_column])

                    # --- Exclude names here ---
            if exclude_names is not None:
                exclude_norm = [n.strip().upper() for n in exclude_names]

                # These are already DataFrames → filter with boolean masks
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

            merged_AGN_clean = (
                merged_AGN_clean
                    .replace([np.inf, -np.inf], np.nan)
                    .infer_objects(copy=False)
                    .dropna(subset=[x_column, y_column])
            )
            merged_inactive_clean = (
                merged_inactive_clean
                    .replace([np.inf, -np.inf], np.nan)
                    .infer_objects(copy=False)
                    .dropna(subset=[x_column, y_column])
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
                color=colors[i],
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
                color=colors[i],
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

        output_path = f'/data/c3040163/llama/alma/gas_analysis_results/Plots/compare_{x_column}_vs_{y_column}.png'
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
            print(f"WARNING: {AGN_path} not found → using default {default_AGN}")
            AGN_path = default_AGN

        if not os.path.exists(inactive_path):
            print(f"WARNING: {inactive_path} not found → using default {default_inactive}")
            inactive_path = default_inactive

        # ---- Load files ----
        fit_data_AGN = pd.read_csv(AGN_path)
        fit_data_inactive = pd.read_csv(inactive_path)

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
        inactive_bol = inactive_bol.copy()
        inactive_bol.loc[:, 'log LAGN'] = pd.to_numeric(inactive_bol['log LAGN'], errors='coerce')
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

            # These are already DataFrames → filter with boolean masks
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

            # GB21, WIS, PHANGS, SIM: detect if list or DataFrame
            if use_gb21:
                if isinstance(GB21, pd.DataFrame):
                    GB21 = GB21[~GB21["Name"].str.strip().str.upper().isin(exclude_norm)]
                else:  # list of dicts
                    GB21 = [row for row in GB21 if str(row["Name"]).strip().upper() not in exclude_norm]

            if use_wis:
                if isinstance(wis, pd.DataFrame):
                    wis = wis[~wis["Name"].str.strip().str.upper().isin(exclude_norm)]
                else:
                    wis = [row for row in wis if str(row["Name"]).strip().upper() not in exclude_norm]

            if use_phangs:
                if isinstance(phangs, pd.DataFrame):
                    phangs = phangs[~phangs["Name"].str.strip().str.upper().isin(exclude_norm)]
                else:
                    phangs = [row for row in phangs if str(row["Name"]).strip().upper() not in exclude_norm]

            if use_sim:
                if isinstance(sim, pd.DataFrame):
                    sim = sim[~sim["Name"].str.strip().str.upper().isin(exclude_norm)]
                else:
                    sim = [row for row in sim if str(row["Name"]).strip().str.upper() not in exclude_norm]

        
        merged_AGN_clean = merged_AGN_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_column, y_column])
        merged_inactive_clean = merged_inactive_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_column, y_column])

        if use_gb21:
            GB21_df = pd.DataFrame(GB21)
            GB21_df[x_column] = pd.to_numeric(GB21_df[x_column], errors='coerce')
            GB21_df[y_column] = pd.to_numeric(GB21_df[y_column], errors='coerce')
            GB21_clean = GB21_df.dropna(subset=[x_column, y_column])
            x_gb21 = GB21_clean[x_column]
            y_gb21 = GB21_clean[y_column]
            names_gb21 = GB21_clean["Name"].values
        if use_wis:
            wis_df = pd.DataFrame(wis)
            wis_df['Name'] = normalize_name(wis_df['Name'])
            wis_df['Name'] = wis_df['Name'].str.replace(" ", "", regex=False)   # remove all spaces
            df_wis = wis_properties
            df_wis['Name'] = df_wis.index
            df_wis['Name'] = normalize_name(df_wis['Name'])
            df_wis['Name'] = df_wis['Name'].str.replace(" ", "", regex=False)   # remove all spaces
            wis_H_phot_df = wis_H_phot.to_pandas()
            wis_H_phot_df['ID'] = normalize_name(wis_H_phot_df['ID'])
            df_wis = df_wis.merge(
        wis_H_phot_df,
        left_on="Name",
        right_on="ID",
        how="left"
    )
            

            wis_df = pd.merge(wis_df, df_wis, left_on='Name', right_on='Name',how='left')
            D_cm = pd.to_numeric(wis_df["Distance (Mpc)"], errors="coerce") * 3.0856776e24
            H_flux = pd.to_numeric(wis_df["H flux"], errors="coerce") if "H flux" in wis_df.columns else pd.Series(np.nan, index=wis_df.index)
            L = 4 * np.pi * D_cm**2 * (H_flux/0.21)*1.662 
            # only take log10 where L is positive, otherwise set NaN
            with np.errstate(invalid="ignore", divide="ignore"):
                wis_df["log LH (L⊙)"] = np.where(L > 0, np.log10(L / 3.828e33), np.nan)

            wis_df[x_column] = pd.to_numeric(wis_df[x_column], errors='coerce')
            wis_df[y_column] = pd.to_numeric(wis_df[y_column], errors='coerce')
            wis_clean = wis_df.dropna(subset=[x_column, y_column])
            x_wis = wis_clean[x_column]
            y_wis = wis_clean[y_column]
            names_wis = wis_clean["Name"].values
        if use_phangs:
            phangs_df = pd.DataFrame(phangs)
            phangs_df['Name'] = normalize_name(phangs_df['Name'])
            phangs_df['Name'] = phangs_df['Name'].str.replace(" ", "", regex=False)   # remove all spaces
            df_phangs = pd.DataFrame(phangs_properties)
            df_phangs2 = pd.DataFrame(phangs_properties2).T.reset_index()
            df_phangs2 = df_phangs2.rename(columns={'index': 'Name'})
            df_phangs = df_phangs.reset_index()  # moves index to column 'name'
            if 'name' in df_phangs.columns[df_phangs.columns.duplicated()]:
                df_phangs = df_phangs.loc[:, ~df_phangs.columns.duplicated()]
            df_phangs['name'] = df_phangs['name'].str.replace(" ", "", regex=False)   # remove all spaces
            df_phangs2['name'] = normalize_name(df_phangs['name'])
            df_phangs2['name'] = df_phangs['name'].str.replace(" ", "", regex=False)   # remove all spaces
            phangs_H_phot_df = phangs_H_phot.to_pandas()
            phangs_H_phot_df['ID'] = normalize_name(phangs_H_phot_df['ID'])
            phangs_H_phot_df['ID'] = phangs_H_phot_df['ID'].str.replace(" ", "", regex=False)
            # merge phangs H-photometry into df_phangs using normalized keys
            df_phangs = df_phangs.merge(
                phangs_H_phot_df,
                left_on="name",
                right_on="ID",
                how="left"
            )
            # ensure df_phangs2 was created/renamed correctly and normalize its Name column
            df_phangs2 = df_phangs2.rename(columns={'index': 'Name'})
            df_phangs2['Name'] = normalize_name(df_phangs2['Name'])
            df_phangs2['Name'] = df_phangs2['Name'].str.replace(" ", "", regex=False)

            # also normalize df_phangs 'name' (remove spaces already done above but keep for safety)
            df_phangs['name'] = normalize_name(df_phangs['name'])
            df_phangs['name'] = df_phangs['name'].str.replace(" ", "", regex=False)
            # merge additional properties from df_phangs2
            df_phangs = df_phangs.merge(
                df_phangs2,
                left_on="name",
                right_on="Name",
                how="left"
            )
            # Ensure phangs_df 'Name' is in the same normalized form as df_phangs['Name'] before final merge
            phangs_df['Name'] = normalize_name(phangs_df['Name'])
            phangs_df['Name'] = phangs_df['Name'].str.replace(" ", "", regex=False)

            phangs_df = pd.merge(phangs_df, df_phangs, left_on='Name', right_on='Name', how='left')

            D_cm = pd.to_numeric(phangs_df["Distance (Mpc)"], errors="coerce") * 3.0856776e24
            H_flux = pd.to_numeric(phangs_df["H flux"], errors="coerce") if "H flux" in phangs_df.columns else pd.Series(np.nan, index=phangs_df.index)

            L = 4 * np.pi * D_cm**2 * (H_flux/0.21)*1.662 # convert from L H (multiplied by bandwidth) to lambdafnu H
                    # only take log10 where L is positive, otherwise set NaN
            with np.errstate(invalid="ignore", divide="ignore"):
                phangs_df["log LH (L⊙)"] = np.where(L > 0, np.log10(L / 3.828e33), np.nan)

            phangs_df[x_column] = pd.to_numeric(phangs_df[x_column], errors='coerce')
            phangs_df[y_column] = pd.to_numeric(phangs_df[y_column], errors='coerce')
            phangs_clean = phangs_df.dropna(subset=[x_column, y_column])
            x_phangs = phangs_clean[x_column]
            y_phangs = phangs_clean[y_column]
            names_phangs = phangs_clean["Name"].values
        if use_sim:
            sim_df = pd.DataFrame(sim)
            sim_df[x_column] = pd.to_numeric(sim_df[x_column], errors='coerce')
            sim_df[y_column] = pd.to_numeric(sim_df[y_column], errors='coerce')
            sim_clean = sim_df.dropna(subset=[x_column, y_column])
            x_sim = sim_clean[x_column]
            y_sim = sim_clean[y_column]
            names_sim = sim_clean["Name"].values

        # Extract values
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

        names_llama = list(names_agn) + list(names_inactive)
        if use_phangs:
            shared_names_phangs = [name if name in names_llama else None for name in names_phangs]
            if not plotshared:
                mask_keep = ~np.isin(names_phangs, names_llama)
                x_phangs = x_phangs[mask_keep]
                y_phangs = y_phangs[mask_keep]


        if use_wis:
            shared_names_wis = [name if name in names_llama else None for name in names_wis]
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

            from scipy.stats import spearmanr

            # ---- Append statistics to global stats table ----
            global stats_table

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
            
            if stats_table.empty:
                stats_table = pd.DataFrame(columns=new_row.keys())
            stats_table.loc[len(stats_table)] = new_row


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

            colour_AGN = 'red'
            colour_inactive = 'blue'
            label_AGN = 'LLAMA AGN'
            label_inactive = 'LLAMA Inactive'
            marker_AGN = 's'
            marker_inactive = 'v'
            if comb_llama:
                colour_AGN = 'black'
                colour_inactive = 'black'
                label_AGN = 'LLAMA Galaxies'
                label_inactive = None
                marker_AGN = 'o'
                marker_inactive = 'o'

            # Plot scatter points
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
                marker='^', color='purple', label='WIS', s=36, alpha=0.8, edgecolors='none'
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
            output_path = f'/data/c3040163/llama/alma/gas_analysis_results/Plots/{mask}_{R_kpc}kpc/{suffix}_{x_column}_vs_{y_column}.png'
            plt.savefig(output_path)
            print(f"Saved plot to: {output_path}")
            plt.close(fig)

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

            output_path = f'/data/c3040163/llama/alma/gas_analysis_results/Plots/pair_diffs/{mask}_{R_kpc}kpc/{y_column}_pair_differences.png'
            plt.savefig(output_path)
            print(f"Saved matched-pairs plot to: {output_path}")
            plt.close(fig)


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
                output_path = f'/data/c3040163/llama/alma/gas_analysis_results/Plots/{mask}_{R_kpc}kpc/{suffix}{x_column}_vs_{y_column}.png'
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
                output_path = f'/data/c3040163/llama/alma/gas_analysis_results/Plots/{mask}_{R_kpc}kpc/{suffix}_{x_column}_vs_{y_column}.png'
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

wisdom = pd.DataFrame([
    ["FRL49", 0.30, 0.04, 0.17, 0.01, 0.41, 0.12],
    ["MRK567", 0.76, 0.05, 0.23, 0.01, 0.50, 0.05],
    ["NGC0383", 0.11, 0.01, 0.13, 0.02, 0.16, 0.03],
    ["NGC0449", 0.60, 0.02, 0.40, 0.01, 0.64, 0.01],
    ["NGC0524", 0.23, 0.01, 0.19, 0.01, 0.43, 0.02],
    ["NGC0612", 0.63, 0.03, 0.52, 0.01, 0.47, 0.02],
    ["NGC0708", 0.69, 0.01, 0.32, 0.03, 0.51, 0.06],
    ["NGC1387", 0.15, 0.02, 0.10, 0.01, 0.27, 0.02],
    ["NGC1574", 0.02, 0.01, 0.04, 0.01, 0.31, 0.05],
    ["NGC3169", 0.99, 0.03, 0.33, 0.01, 0.81, 0.03],
    ["NGC3368", 0.49, 0.04, 0.19, 0.01, 0.79, 0.04],
    ["NGC3607", 0.32, 0.01, 0.24, 0.02, 0.57, 0.01],
    ["NGC4061", 0.14, 0.01, 0.15, 0.02, 0.32, 0.06],
    ["NGC4429", 0.15, 0.01, 0.09, 0.02, 0.29, 0.06],
    ["NGC4435", 0.18, 0.01, 0.19, 0.03, 0.37, 0.08],
    ["NGC4438", 0.30, 0.04, 0.17, 0.01, 0.60, 0.04],
    ["NGC4501", 0.75, 0.02, 0.38, 0.01, 0.79, 0.03],
    ["NGC4697", 0.13, 0.01, 0.20, 0.02, 0.51, 0.04],
    ["NGC4826", 0.25, 0.02, 0.04, 0.01, 0.37, 0.03],
    ["NGC5064", 0.26, 0.03, 0.14, 0.02, 0.26, 0.07],
    ["NGC5765b", 0.67, 0.03, 0.43, 0.01, 0.64, 0.01],
    ["NGC5806", 0.36, 0.03, 0.32, 0.01, 0.53, 0.09],
    ["NGC6753", 0.46, 0.02, 0.26, 0.01, 0.44, 0.08],
    ["NGC6958", 0.11, 0.01, 0.10, 0.01, 0.42, 0.09],
    ["NGC7052", 0.22, 0.02, 0.14, 0.03, 0.34, 0.09],
    ["NGC7172", 0.21, 0.01, 0.18, 0.02, 0.64, 0.03]
], columns=["Name", "Asymmetry", "Asymmetry_err", "Smoothness", "Smoothness_err", "Gini", "Gini_err"])

simulations = pd.DataFrame([
    ["noB", 1.57, 0.08, 0.52, 0.027, 0.81, 0.03],
    ["B_M30_R1", 0.69, 0.05, 0.27, 0.020, 0.39, 0.02],
    ["B_M30_R2", 1.24, 0.08, 0.44, 0.037, 0.67, 0.03],
    ["B_M30_R3", 1.47, 0.11, 0.56, 0.026, 0.77, 0.02],
    ["B_M60_R1", 0.40, 0.04, 0.20, 0.013, 0.24, 0.02],
    ["B_M60_R3", 1.10, 0.06, 0.39, 0.015, 0.60, 0.02],
    ["B_M60_R2", 0.75, 0.07, 0.30, 0.027, 0.42, 0.04],
    ["B_M90_R1", 0.35, 0.04, 0.20, 0.012, 0.23, 0.02],
    ["B_M90_R2", 0.46, 0.05, 0.21, 0.016, 0.25, 0.02],
    ["B_M90_R3", 0.69, 0.05, 0.27, 0.015, 0.36, 0.02]
], columns=["Name", "Asymmetry", "Asymmetry_err", "Smoothness", "Smoothness_err", "Gini", "Gini_err"])

phangs = pd.DataFrame([
    ["IC1954", 0.97, 0.01, 0.30, 0.01, 0.57, 0.01],
    ["IC5273", 0.94, 0.02, 0.37, 0.01, 0.75, 0.01],
    ["NGC0628", 0.70, 0.02, 0.32, 0.01, 0.65, 0.01],
    ["NGC0685", 1.38, 0.01, 0.38, 0.01, 0.83, 0.01],
    ["NGC1087", 0.83, 0.03, 0.27, 0.01, 0.67, 0.01],
    ["NGC1097", 0.43, 0.01, 0.24, 0.01, 0.47, 0.05],
    ["NGC1300", 0.61, 0.02, 0.19, 0.01, 0.61, 0.10],
    ["NGC1317", 0.56, 0.01, 0.22, 0.01, 0.39, 0.01],
    ["NGC1365", 0.71, 0.02, 0.17, 0.01, 0.52, 0.04],
    ["NGC1385", 1.43, 0.01, 0.32, 0.01, 0.65, 0.01],
    ["NGC1433", 0.63, 0.01, 0.23, 0.01, 0.47, 0.06],
    ["NGC1511", 1.45, 0.02, 0.32, 0.01, 0.70, 0.01],
    ["NGC1512", 0.66, 0.02, 0.36, 0.02, 0.49, 0.05],
    ["NGC1546", 0.32, 0.01, 0.13, 0.01, 0.27, 0.03],
    ["NGC1559", 1.31, 0.01, 0.38, 0.01, 0.70, 0.01],
    ["NGC1566", 0.71, 0.05, 0.32, 0.01, 0.74, 0.04],
    ["NGC1637", 0.58, 0.03, 0.33, 0.01, 0.71, 0.01],
    ["NGC1672", 0.53, 0.02, 0.17, 0.01, 0.46, 0.06],
    ["NGC1792", 0.66, 0.04, 0.18, 0.01, 0.42, 0.02],
    ["NGC2090", 0.69, 0.01, 0.29, 0.01, 0.44, 0.01],
    ["NGC2566", 0.41, 0.01, 0.16, 0.01, 0.75, 0.06],
    ["NGC2903", 0.74, 0.01, 0.24, 0.01, 0.73, 0.04],
    ["NGC2997", 0.62, 0.02, 0.20, 0.01, 0.64, 0.04],
    ["NGC3059", 1.22, 0.01, 0.29, 0.01, 0.74, 0.01],
    ["NGC3137", 1.26, 0.01, 0.40, 0.01, 0.64, 0.02],
    ["NGC3351", 0.27, 0.03, 0.33, 0.01, 0.54, 0.18],
    ["NGC3511", 0.52, 0.03, 0.24, 0.01, 0.48, 0.02],
    ["NGC3507", 0.83, 0.05, 0.42, 0.02, 0.75, 0.03],
    ["NGC3521", 0.41, 0.03, 0.18, 0.01, 0.29, 0.01],
    ["NGC3596", 0.87, 0.03, 0.35, 0.01, 0.68, 0.01],
    ["NGC3621", 0.92, 0.03, 0.27, 0.01, 0.47, 0.01],
    ["NGC3626", 1.27, 0.01, 0.52, 0.01, 0.73, 0.01],
    ["NGC3627", 0.69, 0.04, 0.38, 0.01, 0.80, 0.01],
    ["NGC4207", 0.62, 0.01, 0.37, 0.01, 0.74, 0.02],
    ["NGC4254", 0.70, 0.01, 0.24, 0.01, 0.45, 0.01],
    ["NGC4293", 0.38, 0.03, 0.44, 0.01, 0.83, 0.03],
    ["NGC4298", 0.63, 0.01, 0.24, 0.01, 0.44, 0.01],
    ["NGC4303", 0.47, 0.01, 0.18, 0.01, 0.61, 0.07],
    ["NGC4321", 0.56, 0.01, 0.29, 0.01, 0.59, 0.08],
    ["NGC4424", 1.16, 0.03, 0.37, 0.02, 0.80, 0.03],
    ["NGC4457", 0.95, 0.01, 0.31, 0.01, 0.75, 0.01],
    ["NGC4496A", 1.61, 0.01, 0.51, 0.01, 0.85, 0.03],
    ["NGC4535", 0.41, 0.05, 0.21, 0.01, 0.82, 0.08],
    ["NGC4536", 0.34, 0.03, 0.18, 0.01, 0.68, 0.06],
    ["NGC4540", 1.31, 0.01, 0.47, 0.01, 0.74, 0.01],
    ["NGC4548", 0.46, 0.05, 0.35, 0.01, 0.92, 0.07],
    ["NGC4569", 0.71, 0.01, 0.22, 0.01, 0.73, 0.04],
    ["NGC4579", 0.93, 0.02, 0.35, 0.01, 0.71, 0.01],
    ["NGC4654", 0.37, 0.02, 0.13, 0.01, 0.45, 0.03],
    ["NGC4689", 0.85, 0.01, 0.27, 0.01, 0.45, 0.01],
    ["NGC4694", 1.50, 0.03, 0.35, 0.01, 0.91, 0.02],
    ["NGC4731", 1.56, 0.02, 0.52, 0.01, 0.91, 0.01],
    ["NGC4781", 0.97, 0.01, 0.27, 0.01, 0.57, 0.03],
    ["NGC4941", 0.83, 0.03, 0.35, 0.02, 0.92, 0.01],
    ["NGC5134", 1.62, 0.01, 0.51, 0.01, 0.85, 0.01],
    ["NGC5248", 0.39, 0.02, 0.17, 0.01, 0.52, 0.06],
    ["NGC5530", 1.00, 0.01, 0.37, 0.01, 0.60, 0.01],
    ["NGC5643", 0.89, 0.01, 0.28, 0.01, 0.82, 0.02],
    ["NGC6300", 0.78, 0.01, 0.36, 0.01, 0.87, 0.01],
    ["NGC7496", 0.53, 0.01, 0.28, 0.01, 0.77, 0.06]
], columns=["Name", "Asymmetry", "Asymmetry_err", "Smoothness", "Smoothness_err", "Gini", "Gini_err"])


wis_properties = {
    "FRL49": {
        "Type": "E★", "Distance (Mpc)": 85.7, "log_MH2": 8.68, "log_SigmaH2_1kpc": 2.91,
        "log_M": 10.30, "sigma": None, "ReKs": 0.78, "log_SFR": 9.31,
        "log_mu": 0.19, "Beam_arcsec": 77.2, "Beam_pc": None, 
        "Mass_Ref": "Lelli+ subm.", "Data_Ref": "Lelli+subm."
    },
    "MRK567": {
        "Type": "S", "Distance (Mpc)": 140.6, "log_MH2": 8.79, "log_SigmaH2_1kpc": 3.28,
        "log_M": 11.26, "sigma": None, "ReKs": 1.30, "log_SFR": 9.24,
        "log_mu": 0.14, "Beam_arcsec": 93.4, "Beam_pc": None, 
        "Mass_Ref": "C17", "Data_Ref": None
    },
    "NGC0383": {
        "Type": "E", "Distance (Mpc)": 66.6, "log_MH2": 9.18, "log_SigmaH2_1kpc": 2.66,
        "log_M": 11.82, "sigma": 239, "ReKs": 0.00, "log_SFR": 9.92,
        "log_mu": 0.13, "Beam_arcsec": 42.8, "Beam_pc": None, 
        "Mass_Ref": "MASSIVE", "Data_Ref": "North et al. (2019)"
    },
    "NGC0449": {
        "Type": "S", "Distance (Mpc)": 66.3, "log_MH2": 9.50, "log_SigmaH2_1kpc": 2.24,
        "log_M": 10.07, "sigma": 250, "ReKs": 1.19, "log_SFR": 8.60,
        "log_mu": 0.66, "Beam_arcsec": 211.2, "Beam_pc": None, 
        "Mass_Ref": "z0MGS", "Data_Ref": None
    },
    "NGC0524": {
        "Type": "E", "Distance (Mpc)": 23.3, "log_MH2": 7.95, "log_SigmaH2_1kpc": 1.41,
        "log_M": 11.40, "sigma": 220, "ReKs": -0.56, "log_SFR": 9.75,
        "log_mu": 0.32, "Beam_arcsec": 36.7, "Beam_pc": None, 
        "Mass_Ref": "z0MGS", "Data_Ref": "Smith et al. (2019)"
    },
    "NGC0612": {
        "Type": "E", "Distance (Mpc)": 130.4, "log_MH2": 10.30, "log_SigmaH2_1kpc": 1.73,
        "log_M": 11.76, "sigma": None, "ReKs": 0.85, "log_SFR": 9.13,
        "log_mu": 0.19, "Beam_arcsec": 122.2, "Beam_pc": None, 
        "Mass_Ref": "MKs", "Data_Ref": "Ruffa+ in prep"
    },
    "NGC0708": {
        "Type": "E", "Distance (Mpc)": 58.3, "log_MH2": 8.48, "log_SigmaH2_1kpc": 2.04,
        "log_M": 11.75, "sigma": 230, "ReKs": -0.29, "log_SFR": 9.30,
        "log_mu": 0.09, "Beam_arcsec": 24.1, "Beam_pc": None, 
        "Mass_Ref": "MASSIVE", "Data_Ref": "North et al. (2021)"
    },
    "NGC1387": {
        "Type": "E", "Distance (Mpc)": 19.9, "log_MH2": 8.33, "log_SigmaH2_1kpc": 2.04,
        "log_M": 10.67, "sigma": 87, "ReKs": -0.68, "log_SFR": 9.51,
        "log_mu": 0.42, "Beam_arcsec": 40.3, "Beam_pc": None, 
        "Mass_Ref": "z0MGS", "Data_Ref": "Boyce+ in prep"
    },
    "NGC1574": {
        "Type": "E", "Distance (Mpc)": 19.3, "log_MH2": 7.64, "log_SigmaH2_1kpc": 2.02,
        "log_M": 10.79, "sigma": 180, "ReKs": -0.91, "log_SFR": 9.41,
        "log_mu": 0.17, "Beam_arcsec": 15.4, "Beam_pc": None, 
        "Mass_Ref": "z0MGS", "Data_Ref": "Ruffa+ in prep"
    },
    "NGC3169": {
        "Type": "S", "Distance (Mpc)": 18.7, "log_MH2": 9.53, "log_SigmaH2_1kpc": 2.29,
        "log_M": 10.84, "sigma": 165, "ReKs": 0.29, "log_SFR": 8.26,
        "log_mu": 0.60, "Beam_arcsec": 54.0, "Beam_pc": None, 
        "Mass_Ref": "z0MGS", "Data_Ref": None
    },
    "NGC3368": {
        "Type": "S", "Distance (Mpc)": 18.0, "log_MH2": 9.03, "log_SigmaH2_1kpc": 2.46,
        "log_M": 10.67, "sigma": 102, "ReKs": -0.29, "log_SFR": 8.87,
        "log_mu": 0.20, "Beam_arcsec": 17.9, "Beam_pc": None, 
        "Mass_Ref": "z0MGS", "Data_Ref": None
    },
    "NGC3607": {
        "Type": "E", "Distance (Mpc)": 22.2, "log_MH2": 8.42, "log_SigmaH2_1kpc": 1.86,
        "log_M": 11.34, "sigma": 207, "ReKs": -0.54, "log_SFR": 9.80,
        "log_mu": 0.55, "Beam_arcsec": 59.0, "Beam_pc": None, 
        "Mass_Ref": "A3D", "Data_Ref": None
    }
}

phangs_properties = [
    {"name":"NGC 0247","logMstar":9.53,"r25":10.6,"Re":5.0,"la":3.3,"logSFR":-0.75,"logLCO":6.79,"Corr":1.42,"logMstarHI":9.24,"is_limit":False},
    {"name":"NGC 0253","logMstar":10.64,"r25":14.4,"Re":4.7,"la":2.8,"logSFR":0.70,"logLCO":8.96,"Corr":1.00,"logMstarHI":9.33,"is_limit":False},
    {"name":"NGC 0300","logMstar":9.27,"r25":5.9,"Re":2.0,"la":1.3,"logSFR":-0.82,"logLCO":6.61,"Corr":1.50,"logMstarHI":9.32,"is_limit":False},
    {"name":"NGC 0628","logMstar":10.34,"r25":14.1,"Re":3.9,"la":2.9,"logSFR":0.24,"logLCO":8.41,"Corr":1.73,"logMstarHI":9.70,"is_limit":False},
    {"name":"NGC 0685","logMstar":10.07,"r25":8.7,"Re":5.0,"la":3.1,"logSFR":-0.38,"logLCO":7.87,"Corr":1.25,"logMstarHI":9.57,"is_limit":False},
    {"name":"NGC 1068","logMstar":10.91,"r25":12.4,"Re":0.9,"la":7.3,"logSFR":1.64,"logLCO":9.23,"Corr":1.30,"logMstarHI":9.06,"is_limit":False},
    {"name":"NGC 1097","logMstar":10.76,"r25":20.9,"Re":2.6,"la":4.3,"logSFR":0.68,"logLCO":8.93,"Corr":1.31,"logMstarHI":9.61,"is_limit":False},
    {"name":"NGC 1087","logMstar":9.94,"r25":6.9,"Re":3.2,"la":2.1,"logSFR":0.11,"logLCO":8.32,"Corr":1.06,"logMstarHI":9.10,"is_limit":False},
    {"name":"NGC 1313","logMstar":9.26,"r25":7.0,"Re":2.5,"la":2.1,"logSFR":-0.14,"logLCO":None,"Corr":None,"logMstarHI":9.28,"is_limit":False},
    {"name":"NGC 1300","logMstar":10.62,"r25":16.4,"Re":6.5,"la":3.7,"logSFR":0.07,"logLCO":8.50,"Corr":1.28,"logMstarHI":9.38,"is_limit":False},
    {"name":"NGC 1317","logMstar":10.62,"r25":8.5,"Re":1.8,"la":2.4,"logSFR":-0.32,"logLCO":8.10,"Corr":1.28,"logMstarHI":None,"is_limit":False},
    {"name":"IC 1954","logMstar":9.67,"r25":5.6,"Re":2.4,"la":1.5,"logSFR":-0.44,"logLCO":7.78,"Corr":1.10,"logMstarHI":8.85,"is_limit":False},
    {"name":"NGC 1365","logMstar":11.00,"r25":34.2,"Re":2.8,"la":13.1,"logSFR":1.24,"logLCO":9.49,"Corr":1.36,"logMstarHI":9.94,"is_limit":False},
    {"name":"NGC 1385","logMstar":9.98,"r25":8.5,"Re":3.4,"la":2.6,"logSFR":0.32,"logLCO":8.37,"Corr":1.09,"logMstarHI":9.19,"is_limit":False},
    {"name":"NGC 1433","logMstar":10.87,"r25":16.8,"Re":4.3,"la":6.9,"logSFR":0.05,"logLCO":8.47,"Corr":1.38,"logMstarHI":9.40,"is_limit":False},
    {"name":"NGC 1511","logMstar":9.92,"r25":8.2,"Re":2.4,"la":1.7,"logSFR":0.35,"logLCO":8.22,"Corr":1.09,"logMstarHI":9.57,"is_limit":False},
    {"name":"NGC 1512","logMstar":10.72,"r25":23.1,"Re":4.8,"la":6.2,"logSFR":0.11,"logLCO":8.26,"Corr":1.45,"logMstarHI":9.88,"is_limit":False},
    {"name":"NGC 1546","logMstar":10.37,"r25":9.5,"Re":2.2,"la":2.1,"logSFR":-0.08,"logLCO":8.44,"Corr":1.13,"logMstarHI":8.68,"is_limit":False},
    {"name":"NGC 1559","logMstar":10.37,"r25":11.8,"Re":3.9,"la":2.4,"logSFR":0.60,"logLCO":8.66,"Corr":1.11,"logMstarHI":9.52,"is_limit":False},
    {"name":"NGC 1566","logMstar":10.79,"r25":18.6,"Re":3.2,"la":3.9,"logSFR":0.66,"logLCO":8.89,"Corr":1.22,"logMstarHI":9.80,"is_limit":False},
    {"name":"NGC 1637","logMstar":9.95,"r25":5.4,"Re":2.8,"la":1.8,"logSFR":-0.20,"logLCO":7.98,"Corr":1.10,"logMstarHI":9.20,"is_limit":False},
    {"name":"NGC 1672","logMstar":10.73,"r25":17.4,"Re":3.4,"la":5.8,"logSFR":0.88,"logLCO":9.05,"Corr":1.25,"logMstarHI":10.21,"is_limit":False},
    {"name":"NGC 1809","logMstar":9.77,"r25":10.9,"Re":4.5,"la":2.4,"logSFR":0.76,"logLCO":7.49,"Corr":4.24,"logMstarHI":9.60,"is_limit":False},
    {"name":"NGC 1792","logMstar":10.62,"r25":13.1,"Re":4.1,"la":2.4,"logSFR":0.57,"logLCO":8.95,"Corr":1.11,"logMstarHI":9.25,"is_limit":False},
    {"name":"NGC 2090","logMstar":10.04,"r25":7.7,"Re":1.9,"la":1.7,"logSFR":-0.39,"logLCO":7.67,"Corr":1.47,"logMstarHI":9.37,"is_limit":False},
    {"name":"NGC 2283","logMstar":9.89,"r25":5.5,"Re":3.2,"la":1.9,"logSFR":-0.28,"logLCO":7.69,"Corr":1.16,"logMstarHI":9.70,"is_limit":False},
    {"name":"NGC 2566","logMstar":10.71,"r25":14.5,"Re":5.1,"la":4.0,"logSFR":0.93,"logLCO":9.06,"Corr":1.13,"logMstarHI":9.37,"is_limit":False},
    {"name":"NGC 2775","logMstar":11.07,"r25":14.3,"Re":4.6,"la":4.1,"logSFR":-0.06,"logLCO":8.40,"Corr":1.29,"logMstarHI":8.65,"is_limit":False},
    {"name":"NGC 2835","logMstar":10.00,"r25":11.4,"Re":3.3,"la":2.2,"logSFR":0.10,"logLCO":7.71,"Corr":1.72,"logMstarHI":9.48,"is_limit":False},
    {"name":"NGC 2903","logMstar":10.64,"r25":17.4,"Re":3.7,"la":3.5,"logSFR":0.49,"logLCO":8.76,"Corr":1.18,"logMstarHI":9.54,"is_limit":False},
    {"name":"NGC 2997","logMstar":10.73,"r25":21.0,"Re":6.1,"la":4.0,"logSFR":0.64,"logLCO":8.97,"Corr":1.25,"logMstarHI":9.86,"is_limit":False},
    {"name":"NGC 3059","logMstar":10.38,"r25":11.2,"Re":5.0,"la":3.2,"logSFR":0.38,"logLCO":8.59,"Corr":1.07,"logMstarHI":9.75,"is_limit":False},
    {"name":"NGC 3137","logMstar":9.88,"r25":13.2,"Re":4.1,"la":3.0,"logSFR":-0.30,"logLCO":7.60,"Corr":1.35,"logMstarHI":9.68,"is_limit":False},
    {"name":"NGC 3239","logMstar":9.18,"r25":5.7,"Re":3.1,"la":2.0,"logSFR":-0.41,"logLCO":6.62,"Corr":1.54,"logMstarHI":9.16,"is_limit":True},
    {"name":"NGC 3351","logMstar":10.37,"r25":10.5,"Re":3.0,"la":2.1,"logSFR":0.12,"logLCO":8.13,"Corr":1.55,"logMstarHI":8.93,"is_limit":False},
    {"name":"NGC 3489","logMstar":10.29,"r25":5.9,"Re":1.3,"la":1.4,"logSFR":-1.59,"logLCO":6.89,"Corr":1.37,"logMstarHI":7.40,"is_limit":False},
    {"name":"NGC 3511","logMstar":10.03,"r25":12.2,"Re":4.4,"la":2.4,"logSFR":-0.09,"logLCO":8.15,"Corr":1.07,"logMstarHI":9.37,"is_limit":False},
    {"name":"NGC 3507","logMstar":10.40,"r25":10.0,"Re":3.7,"la":2.3,"logSFR":-0.00,"logLCO":8.34,"Corr":1.17,"logMstarHI":9.32,"is_limit":False},
    {"name":"NGC 3521","logMstar":11.03,"r25":16.0,"Re":3.9,"la":4.9,"logSFR":0.57,"logLCO":8.98,"Corr":1.18,"logMstarHI":9.83,"is_limit":False},
    {"name":"NGC 3596","logMstar":9.66,"r25":6.0,"Re":1.6,"la":2.0,"logSFR":-0.52,"logLCO":7.81,"Corr":1.13,"logMstarHI":8.85,"is_limit":False},
    {"name":"NGC 3599","logMstar":10.04,"r25":6.9,"Re":1.7,"la":2.0,"logSFR":-1.35,"logLCO":6.70,"Corr":1.35,"logMstarHI":None,"is_limit":True},
    {"name":"NGC 3621","logMstar":10.06,"r25":9.8,"Re":2.7,"la":2.0,"logSFR":-0.00,"logLCO":8.13,"Corr":1.27,"logMstarHI":9.66,"is_limit":False},
    {"name":"NGC 3626","logMstar":10.46,"r25":8.6,"Re":1.8,"la":2.1,"logSFR":-0.68,"logLCO":7.75,"Corr":1.14,"logMstarHI":8.89,"is_limit":False},
    {"name":"NGC 3627","logMstar":10.84,"r25":16.9,"Re":3.6,"la":3.7,"logSFR":0.59,"logLCO":8.98,"Corr":1.16,"logMstarHI":9.09,"is_limit":False},
    {"name":"NGC 4207","logMstar":9.72,"r25":3.4,"Re":1.4,"la":0.7,"logSFR":-0.72,"logLCO":7.71,"Corr":1.03,"logMstarHI":8.58,"is_limit":False},
    {"name":"NGC 4254","logMstar":10.42,"r25":9.6,"Re":2.4,"la":1.8,"logSFR":0.49,"logLCO":8.93,"Corr":1.15,"logMstarHI":9.48,"is_limit":False},
    {"name":"NGC 4293","logMstar":10.52,"r25":14.3,"Re":4.7,"la":2.8,"logSFR":-0.30,"logLCO":8.12,"Corr":1.57,"logMstarHI":7.67,"is_limit":False},
    {"name":"NGC 4298","logMstar":10.04,"r25":5.5,"Re":3.0,"la":1.6,"logSFR":-0.34,"logLCO":8.26,"Corr":1.09,"logMstarHI":8.87,"is_limit":False},
    {"name":"NGC 4303","logMstar":10.51,"r25":17.0,"Re":3.4,"la":3.1,"logSFR":0.73,"logLCO":9.00,"Corr":1.40,"logMstarHI":9.67,"is_limit":False},
    {"name":"NGC 4321","logMstar":10.75,"r25":13.5,"Re":5.5,"la":3.6,"logSFR":0.55,"logLCO":9.02,"Corr":1.25,"logMstarHI":9.43,"is_limit":False},
    {"name":"NGC 4424","logMstar":9.93,"r25":7.2,"Re":3.7,"la":2.2,"logSFR":-0.53,"logLCO":7.59,"Corr":1.16,"logMstarHI":8.30,"is_limit":False},
    {"name":"NGC 4457","logMstar":10.42,"r25":6.1,"Re":1.5,"la":2.2,"logSFR":-0.52,"logLCO":8.21,"Corr":1.15,"logMstarHI":8.36,"is_limit":False},
    {"name":"NGC 4459","logMstar":10.68,"r25":9.6,"Re":2.1,"la":3.3,"logSFR":-0.65,"logLCO":7.46,"Corr":2.41,"logMstarHI":None,"is_limit":False},
    {"name":"NGC 4476","logMstar":9.81,"r25":4.3,"Re":1.2,"la":1.2,"logSFR":-1.39,"logLCO":7.05,"Corr":1.09,"logMstarHI":None,"is_limit":False},
    {"name":"NGC 4477","logMstar":10.59,"r25":8.5,"Re":2.1,"la":2.1,"logSFR":-1.10,"logLCO":6.76,"Corr":1.58,"logMstarHI":None,"is_limit":False},
    {"name":"NGC 4496A","logMstar":9.55,"r25":7.3,"Re":3.0,"la":1.9,"logSFR":-0.21,"logLCO":7.55,"Corr":1.15,"logMstarHI":9.24,"is_limit":False},
    {"name":"NGC 4535","logMstar":10.54,"r25":18.7,"Re":6.3,"la":3.8,"logSFR":0.34,"logLCO":8.61,"Corr":1.78,"logMstarHI":9.56,"is_limit":False},
    {"name":"NGC 4536","logMstar":10.40,"r25":16.7,"Re":4.4,"la":2.7,"logSFR":0.53,"logLCO":8.62,"Corr":1.06,"logMstarHI":9.54,"is_limit":False},
    {"name":"NGC 4540","logMstar":9.79,"r25":5.0,"Re":2.0,"la":1.4,"logSFR":-0.78,"logLCO":7.69,"Corr":1.16,"logMstarHI":8.44,"is_limit":False},
    {"name":"NGC 4548","logMstar":10.70,"r25":13.1,"Re":5.4,"la":3.0,"logSFR":-0.28,"logLCO":8.16,"Corr":2.00,"logMstarHI":8.84,"is_limit":False},
    {"name":"NGC 4569","logMstar":10.81,"r25":20.9,"Re":5.9,"la":4.3,"logSFR":0.12,"logLCO":8.81,"Corr":1.40,"logMstarHI":8.84,"is_limit":False},
    {"name":"NGC 4571","logMstar":10.10,"r25":7.7,"Re":3.8,"la":2.0,"logSFR":-0.54,"logLCO":7.88,"Corr":1.55,"logMstarHI":8.70,"is_limit":False},
    {"name":"NGC 4579","logMstar":11.15,"r25":15.3,"Re":5.4,"la":4.4,"logSFR":0.33,"logLCO":8.79,"Corr":1.38,"logMstarHI":9.02,"is_limit":False},
    {"name":"NGC 4596","logMstar":10.59,"r25":9.0,"Re":2.7,"la":3.8,"logSFR":-0.96,"logLCO":6.72,"Corr":1.83,"logMstarHI":None,"is_limit":False},
    {"name":"NGC 4654","logMstar":10.57,"r25":15.1,"Re":5.6,"la":4.0,"logSFR":0.58,"logLCO":8.84,"Corr":1.18,"logMstarHI":9.75,"is_limit":False},
    {"name":"NGC 4689","logMstar":10.24,"r25":8.3,"Re":4.7,"la":3.0,"logSFR":-0.39,"logLCO":8.22,"Corr":1.19,"logMstarHI":8.54,"is_limit":False},
    {"name":"NGC 4694","logMstar":9.90,"r25":4.6,"Re":1.9,"la":1.6,"logSFR":-0.81,"logLCO":7.41,"Corr":1.30,"logMstarHI":8.51,"is_limit":False},
    {"name":"NGC 4731","logMstar":9.50,"r25":12.2,"Re":7.3,"la":3.0,"logSFR":-0.22,"logLCO":7.29,"Corr":2.52,"logMstarHI":9.44,"is_limit":False},
    {"name":"NGC 4781","logMstar":9.64,"r25":6.1,"Re":2.0,"la":1.1,"logSFR":-0.32,"logLCO":7.82,"Corr":1.05,"logMstarHI":8.94,"is_limit":False},
    {"name":"NGC 4826","logMstar":10.24,"r25":6.7,"Re":1.5,"la":1.1,"logSFR":-0.69,"logLCO":7.79,"Corr":1.28,"logMstarHI":8.26,"is_limit":False},
    {"name":"NGC 4941","logMstar":10.18,"r25":7.3,"Re":3.4,"la":2.2,"logSFR":-0.35,"logLCO":7.80,"Corr":1.27,"logMstarHI":8.49,"is_limit":False},
    {"name":"NGC 4951","logMstar":9.79,"r25":6.9,"Re":1.9,"la":1.9,"logSFR":-0.46,"logLCO":7.65,"Corr":1.22,"logMstarHI":9.21,"is_limit":False},
    {"name":"NGC 4945","logMstar":10.36,"r25":11.8,"Re":4.5,"la":1.6,"logSFR":0.19,"logLCO":8.77,"Corr":0.97,"logMstarHI":8.92,"is_limit":False},
    {"name":"NGC 5042","logMstar":9.90,"r25":10.2,"Re":3.3,"la":2.4,"logSFR":-0.22,"logLCO":7.69,"Corr":1.84,"logMstarHI":9.29,"is_limit":False},
    {"name":"NGC 5068","logMstar":9.41,"r25":5.7,"Re":2.0,"la":1.3,"logSFR":-0.56,"logLCO":7.26,"Corr":1.38,"logMstarHI":8.82,"is_limit":False},
    {"name":"NGC 5134","logMstar":10.41,"r25":7.9,"Re":2.9,"la":2.1,"logSFR":-0.34,"logLCO":7.98,"Corr":1.14,"logMstarHI":8.92,"is_limit":False},
    {"name":"NGC 5128","logMstar":10.97,"r25":13.7,"Re":4.7,"la":4.1,"logSFR":0.09,"logLCO":8.40,"Corr":0.98,"logMstarHI":8.43,"is_limit":False},
    {"name":"NGC 5236","logMstar":10.53,"r25":9.7,"Re":3.5,"la":2.4,"logSFR":0.62,"logLCO":8.84,"Corr":1.14,"logMstarHI":9.98,"is_limit":False},
    {"name":"NGC 5248","logMstar":10.41,"r25":8.8,"Re":3.2,"la":2.0,"logSFR":0.36,"logLCO":8.77,"Corr":1.14,"logMstarHI":9.50,"is_limit":False},
    {"name":"ESO097-013","logMstar":10.53,"r25":5.3,"Re":1.9,"la":1.8,"logSFR":0.61,"logLCO":8.42,"Corr":1.40,"logMstarHI":9.81,"is_limit":False},
    {"name":"NGC 5530","logMstar":10.08,"r25":8.6,"Re":3.4,"la":1.7,"logSFR":-0.48,"logLCO":7.89,"Corr":1.34,"logMstarHI":9.11,"is_limit":False},
    {"name":"NGC 5643","logMstar":10.34,"r25":9.7,"Re":3.5,"la":1.6,"logSFR":0.41,"logLCO":8.56,"Corr":1.06,"logMstarHI":9.12,"is_limit":False},
    {"name":"NGC 6300","logMstar":10.47,"r25":9.0,"Re":3.6,"la":2.1,"logSFR":0.29,"logLCO":8.46,"Corr":1.12,"logMstarHI":9.13,"is_limit":False},
    {"name":"NGC 6744","logMstar":10.72,"r25":21.4,"Re":7.0,"la":4.8,"logSFR":0.38,"logLCO":8.27,"Corr":2.75,"logMstarHI":10.31,"is_limit":False},
    {"name":"IC 5273","logMstar":9.73,"r25":6.3,"Re":2.5,"la":1.3,"logSFR":-0.27,"logLCO":7.63,"Corr":1.14,"logMstarHI":8.95,"is_limit":False},
    {"name":"NGC 7456","logMstar":9.65,"r25":9.4,"Re":4.4,"la":2.9,"logSFR":-0.43,"logLCO":7.13,"Corr":2.02,"logMstarHI":9.28,"is_limit":False},
    {"name":"NGC 7496","logMstar":10.00,"r25":9.1,"Re":3.8,"la":1.5,"logSFR":0.35,"logLCO":8.33,"Corr":1.15,"logMstarHI":9.07,"is_limit":False},
    {"name":"IC 5332","logMstar":9.68,"r25":8.0,"Re":3.6,"la":2.8,"logSFR":-0.39,"logLCO":7.09,"Corr":2.26,"logMstarHI":9.30,"is_limit":False},
    {"name":"NGC 7743","logMstar":10.36,"r25":7.7,"Re":2.9,"la":1.9,"logSFR":-0.67,"logLCO":7.50,"Corr":2.65,"logMstarHI":8.50,"is_limit":False},
    {"name":"NGC 7793","logMstar":9.36,"r25":5.5,"Re":1.9,"la":1.1,"logSFR":-0.57,"logLCO":7.23,"Corr":1.34,"logMstarHI":8.70,"is_limit":False}
]

phangs_properties2 = {
    "ESO097-013X": {"vLSR": 430.3, "PA": 36.7, "i": 64.3, "Distance (Mpc)": 4.20},
    "IC 1954": {"vLSR": 1039.1, "PA": 63.4, "i": 57.1, "Distance (Mpc)": 12.80},
    "IC 5273": {"vLSR": 1286.0, "PA": 234.1, "i": 52.0, "Distance (Mpc)": 14.18},
    "IC 5332": {"vLSR": 699.3, "PA": 74.4, "i": 26.9, "Distance (Mpc)": 9.01},
    "NGC 0247X": {"vLSR": 148.8, "PA": 167.4, "i": 76.4, "Distance (Mpc)": 3.71},
    "NGC 0253X": {"vLSR": 235.4, "PA": 52.5, "i": 75.0, "Distance (Mpc)": 3.70},
    "NGC 0300X": {"vLSR": 155.5, "PA": 114.3, "i": 39.8, "Distance (Mpc)": 2.09},
    "NGC 0628": {"vLSR": 650.8, "PA": 20.7, "i": 8.9, "Distance (Mpc)": 9.84},
    "NGC 0685": {"vLSR": 1346.6, "PA": 100.9, "i": 23.0, "Distance (Mpc)": 19.94},
    "NGC 1068X": {"vLSR": 1130.1, "PA": 72.7, "i": 34.7, "Distance (Mpc)": 13.97},
    "NGC 1087": {"vLSR": 1501.5, "PA": 359.1, "i": 42.9, "Distance (Mpc)": 15.85},
    "NGC 1097": {"vLSR": 1257.5, "PA": 122.4, "i": 48.6, "Distance (Mpc)": 13.58},
    "NGC 1313X": {"vLSR": 451.2, "PA": 23.4, "i": 34.8, "Distance (Mpc)": 4.32},
    "NGC 1300": {"vLSR": 1545.4, "PA": 278.0, "i": 31.8, "Distance (Mpc)": 18.99},
    "NGC 1317": {"vLSR": 1930.5, "PA": 221.5, "i": 23.2, "Distance (Mpc)": 19.11},
    "NGC 1365": {"vLSR": 1613.3, "PA": 201.1, "i": 55.4, "Distance (Mpc)": 19.57},
    "NGC 1385": {"vLSR": 1476.8, "PA": 181.3, "i": 44.0, "Distance (Mpc)": 17.22},
    "NGC 1433": {"vLSR": 1057.4, "PA": 199.7, "i": 28.6, "Distance (Mpc)": 18.63},
    "NGC 1511": {"vLSR": 1331.0, "PA": 297.0, "i": 72.7, "Distance (Mpc)": 15.28},
    "NGC 1512": {"vLSR": 871.4, "PA": 261.9, "i": 42.5, "Distance (Mpc)": 18.83},
    "NGC 1546": {"vLSR": 1243.8, "PA": 147.8, "i": 70.3, "Distance (Mpc)": 17.69},
    "NGC 1559": {"vLSR": 1275.2, "PA": 244.5, "i": 65.4, "Distance (Mpc)": 19.44},
    "NGC 1566": {"vLSR": 1483.3, "PA": 214.7, "i": 29.5, "Distance (Mpc)": 17.69},
    "NGC 1637": {"vLSR": 698.9, "PA": 20.6, "i": 31.1, "Distance (Mpc)": 11.70},
    "NGC 1672": {"vLSR": 1318.3, "PA": 134.3, "i": 42.6, "Distance (Mpc)": 19.40},
    "NGC 1809": {"vLSR": 1290.4, "PA": 138.2, "i": 57.6, "Distance (Mpc)": 19.95},
    "NGC 1792": {"vLSR": 1175.9, "PA": 318.9, "i": 65.1, "Distance (Mpc)": 16.20},
    "NGC 2090": {"vLSR": 898.2, "PA": 192.5, "i": 64.5, "Distance (Mpc)": 11.75},
    "NGC 2283": {"vLSR": 821.9, "PA": -4.1, "i": 43.7, "Distance (Mpc)": 13.68},
    "NGC 2566": {"vLSR": 1609.6, "PA": 312.0, "i": 48.5, "Distance (Mpc)": 23.44},
    "NGC 2775": {"vLSR": 1339.2, "PA": 156.5, "i": 41.2, "Distance (Mpc)": 23.15},
    "NGC 2835": {"vLSR": 867.3, "PA": 1.0, "i": 41.3, "Distance (Mpc)": 12.22},
    "NGC 2903": {"vLSR": 547.0, "PA": 203.7, "i": 66.8, "Distance (Mpc)": 10.00},
    "NGC 2997": {"vLSR": 1076.9, "PA": 108.1, "i": 33.0, "Distance (Mpc)": 14.06},
    "NGC 3059": {"vLSR": 1236.5, "PA": -14.8, "i": 29.4, "Distance (Mpc)": 20.23},
    "NGC 3137": {"vLSR": 1086.6, "PA": -0.3, "i": 70.3, "Distance (Mpc)": 16.37},
    "NGC 3239": {"vLSR": 748.3, "PA": 72.9, "i": 60.3, "Distance (Mpc)": 10.86},
    "NGC 3351": {"vLSR": 774.7, "PA": 193.2, "i": 45.1, "Distance (Mpc)": 9.96},
    "NGC 3489X": {"vLSR": 692.1, "PA": 70.0, "i": 63.7, "Distance (Mpc)": 11.86},
    "NGC 3511": {"vLSR": 1096.7, "PA": 256.8, "i": 75.1, "Distance (Mpc)": 13.94},
    "NGC 3507": {"vLSR": 969.4, "PA": 55.8, "i": 21.7, "Distance (Mpc)": 23.55},
    "NGC 3521": {"vLSR": 798.0, "PA": 343.0, "i": 68.8, "Distance (Mpc)": 13.24},
    "NGC 3596": {"vLSR": 1187.9, "PA": 78.4, "i": 25.1, "Distance (Mpc)": 11.30},
    "NGC 3599X": {"vLSR": 836.8, "PA": 41.9, "i": 23.0, "Distance (Mpc)": 19.86},
    "NGC 3621": {"vLSR": 724.3, "PA": 343.8, "i": 65.8, "Distance (Mpc)": 7.06},
    "NGC 3626": {"vLSR": 1470.7, "PA": 165.2, "i": 46.6, "Distance (Mpc)": 20.05},
    "NGC 3627": {"vLSR": 715.4, "PA": 173.1, "i": 57.3, "Distance (Mpc)": 11.32},
    "NGC 4207": {"vLSR": 606.6, "PA": 121.9, "i": 64.5, "Distance (Mpc)": 15.78},
    "NGC 4254": {"vLSR": 2388.2, "PA": 68.1, "i": 34.4, "Distance (Mpc)": 13.10},
    "NGC 4293": {"vLSR": 926.2, "PA": 48.3, "i": 65.0, "Distance (Mpc)": 15.76},
    "NGC 4298": {"vLSR": 1138.1, "PA": 313.9, "i": 59.2, "Distance (Mpc)": 14.92},
    "NGC 4303": {"vLSR": 1559.8, "PA": 312.4, "i": 23.5, "Distance (Mpc)": 16.99},
    "NGC 4321": {"vLSR": 1572.3, "PA": 156.2, "i": 38.5, "Distance (Mpc)": 15.21},
    "NGC 4424": {"vLSR": 447.4, "PA": 88.3, "i": 58.2, "Distance (Mpc)": 16.20},
    "NGC 4457": {"vLSR": 886.0, "PA": 78.7, "i": 17.4, "Distance (Mpc)": 15.10},
    "NGC 4459X": {"vLSR": 1190.1, "PA": 108.8, "i": 47.0, "Distance (Mpc)": 15.85},
    "NGC 4476X": {"vLSR": 1962.7, "PA": 27.4, "i": 60.1, "Distance (Mpc)": 17.54},
    "NGC 4477X": {"vLSR": 1362.2, "PA": 25.7, "i": 33.5, "Distance (Mpc)": 15.76},
    "NGC 4496A": {"vLSR": 1721.8, "PA": 51.1, "i": 53.8, "Distance (Mpc)": 14.86},
    "NGC 4535": {"vLSR": 1953.6, "PA": 179.7, "i": 44.7, "Distance (Mpc)": 15.77},
    "NGC 4536": {"vLSR": 1794.6, "PA": 305.6, "i": 66.0, "Distance (Mpc)": 16.25},
    "NGC 4540": {"vLSR": 1286.5, "PA": 12.8, "i": 28.7, "Distance (Mpc)": 15.76},
    "NGC 4548": {"vLSR": 482.7, "PA": 138.0, "i": 38.3, "Distance (Mpc)": 16.22}
}

# for t in result: 
#     if 'T' in t.colnames: 
#         print(f"Table: {t.meta.get('name', 'unknown')}, Columns: {t['T']}")

wis_H_phot = Table.read('/data/c3040163/llama/wisdom_2mass_Hphotometry.fits', format='fits')
phangs_H_phot = Table.read('/data/c3040163/llama/phangs_2mass_Hphotometry.fits', format='fits')


def get_hubble_T(name):
    result = Vizier.query_object(name, catalog="VII/155/rc3")
    if len(result) == 0:
        result = Vizier.query_object(name, catalog="J/A+A/659/A188/ulx-xmm9")
    T_col = result[0]["T"]
    if isinstance(T_col, MaskedColumn):
        T_data = T_col.filled(np.nan)
    else:
        T_data = np.array(T_col)
    T_val = np.nanmedian(T_data)
    return T_val


###### update wisdom table ######
print("Updating WISDOM table with Hubble T from Vizier...")
wis_properties = pd.DataFrame.from_dict(wis_properties, orient="index")

for name_str in wis_properties.index:

    max_retries = 3
    hubble_T = None

    for attempt in range(max_retries):
        try:
            wis_properties.loc[name_str, "Hubble Stage"] = get_hubble_T(name_str)

            break

        except (requests.exceptions.ConnectionError,
                RemoteServiceError,
                requests.exceptions.ReadTimeout) as e:

            print(f"⚠️ Vizier query failed for {name_str} (attempt {attempt+1}/{max_retries}): {e}")

            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                print("❌ All Vizier attempts failed.")


###### update phangs table ######
print("Updating PHANGS table with Hubble Stage from Vizier...")
if isinstance(phangs_properties, list):
    phangs_properties = pd.DataFrame(phangs_properties)

if "name" in phangs_properties.columns:
    phangs_properties = phangs_properties.set_index("name")

for name_str in phangs_properties.index:

    max_retries = 3
    hubble_T = None

    for attempt in range(max_retries):
        try:
            phangs_properties.loc[name_str, "Hubble Stage"] = get_hubble_T(name_str)

            break

        except (requests.exceptions.ConnectionError,
                RemoteServiceError,
                requests.exceptions.ReadTimeout) as e:

            print(f"⚠️ Vizier query failed for {name_str} (attempt {attempt+1}/{max_retries}): {e}")

            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                print("❌ All Vizier attempts failed.")




 #   """posible x_column: '"Distance (Mpc)"', 'log LH (L⊙)', 'Hubble Stage', 'Axis Ratio', 'Bar'
 #      posible y_column: 'Smoothness', 'Asymmetry', 'Gini', 'Sigma0', 'rs'"""



masks = ['broad', 'strict']
radii = [1, 1.5, 0.3]

for mask in masks:
    for R_kpc in radii:
        print(f"Running plots for mask={mask}, R_kpc={R_kpc}")

    ############ CAS with stellar mass #############

        plot_llama_property('log LH (L⊙)', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('log LH (L⊙)', 'Asymmetry', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('log LH (L⊙)', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('log LH (L⊙)','Concentration',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('log LH (L⊙)','clumping_factor',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'])

        ############# CAS with Hubble Stage #############

        plot_llama_property('Hubble Stage', 'Concentration', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('Hubble Stage', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('Hubble Stage', 'Asymmetry', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('Hubble Stage', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('Hubble Stage','clumping_factor',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'])#,exclude_names=['NGC 2775','NGC 4260','ESO 208-G021','NGC 5845','NGC 2992','NGC 1079','NGC 4388'])

        ############# CAS with X-ray luminosity #############

        plot_llama_property('log LX', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('log LX', 'Asymmetry', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('log LX', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('log LX','Concentration',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,use_gb21=True,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('log LX','clumping_factor',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,use_gb21=False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'])


        plot_llama_property('log LX', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,use_gb21=False,soloplot='inactive',mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('log LX', 'Asymmetry', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,use_gb21=False,soloplot='inactive',mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('log LX', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,use_gb21=False,soloplot='inactive',mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('log LX','Concentration',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,use_gb21=True,soloplot='inactive',mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('log LX','clumping_factor',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,use_gb21=False,soloplot='inactive',mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'])


        plot_llama_property('log LX','Concentration',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,use_gb21=True,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])

        ############## CAS with eachother #############

        plot_llama_property('Gini', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('Asymmetry', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('Asymmetry', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('Gini', 'clumping_factor', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'])#,exclude_names=['NGC 2775','NGC 4260','ESO 208-G021','NGC 5845','NGC 2992','NGC 1079','NGC 4388'])
        plot_llama_property('clumping_factor', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'])
        plot_llama_property('Asymmetry', 'clumping_factor', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'])

        ############## CAS with concentration #############

        plot_llama_property('Gini', 'Concentration', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs, False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('Asymmetry', 'Concentration', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('Smoothness', 'Concentration', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('clumping_factor', 'Concentration', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'])

        ############### CAS with resolution #############

        plot_llama_property('Resolution (pc)', 'Concentration', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('Resolution (pc)', 'Asymmetry', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('Resolution (pc)', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('Resolution (pc)', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('Resolution (pc)', 'total_mass (M_sun)', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,mask=mask,R_kpc=R_kpc,exclude_names=None)
        plot_llama_property('Resolution (pc)', 'clumping_factor', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,mask=mask,R_kpc=R_kpc,logx=False,logy=True,background_image='/data/c3040163/llama/alma/gas_analysis_results/Leroy2013_plots/Clumping.png',manual_limits=[0,500,1,200],legend_loc='center right',exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'])
        plot_llama_property('Resolution (pc)', 'Resolution (pc)', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,mask=mask,R_kpc=R_kpc,exclude_names=None)

        ############### CAS with Gas mass #############

        plot_llama_property('total_mass (M_sun)', 'Concentration', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])#,exclude_names=['NGC 1365'])
        plot_llama_property('total_mass (M_sun)', 'Asymmetry', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('total_mass (M_sun)', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('total_mass (M_sun)', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('total_mass (M_sun)', 'clumping_factor', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'])

        ############### Clumping factor plot #############

        plot_llama_property('area_weighted_sd','mass_weighted_sd',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,logx=True,logy=True,background_image='/data/c3040163/llama/alma/gas_analysis_results/Leroy2013_plots/Sigma.png',manual_limits=[0.5,5000,0.5,5000], truescale=True,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'])

        ############### CAS with Bar #############

        plot_llama_property('Bar', 'Concentration', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('Bar', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('Bar', 'Asymmetry', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('Bar', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])
        plot_llama_property('Bar','clumping_factor',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'])

        ############## L'CO comparison with Ros18 #####################

        plot_llama_property('log L′ CO','L\'CO (K km_s pc2)',AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,logy=True,mask=mask,R_kpc=R_kpc,exclude_names=None)


    ############### CAS WISDOM, PHANGS coplot   #############

        if R_kpc == 1.5:
            plot_llama_property('Gini', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,use_wis=True,use_phangs=True,use_sim=False,comb_llama=True,rebin=120,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375'])
            plot_llama_property('Asymmetry', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,use_wis=True,use_phangs=True,use_sim=False,comb_llama=True,rebin=120,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375'])
            plot_llama_property('Asymmetry', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,use_wis=True,use_phangs=True,use_sim=False,comb_llama=True,rebin=120,mask=mask,R_kpc=R_kpc,exclude_names=['NGC 1375'])

            plot_llama_property('Distance (Mpc)', 'log LH (L⊙)', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018, GB21=GB21_density, wis=wisdom, sim=simulations, phangs=phangs, use_gb21=False, use_wis=True, use_phangs=True, use_sim=False, comb_llama=True, plotshared=False, rebin=120, mask=mask, R_kpc=R_kpc, exclude_names=['NGC 1375'])
            plot_llama_property('Distance (Mpc)', 'Hubble Stage', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018, GB21=GB21_density, wis=wisdom, sim=simulations, phangs=phangs, use_gb21=False, use_wis=True, use_phangs=True, use_sim=False, comb_llama=True,plotshared=False, rebin=120, mask=mask, R_kpc=R_kpc, exclude_names=['NGC 1375'])
            plot_llama_property('Hubble Stage', 'log LH (L⊙)', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018, GB21=GB21_density, wis=wisdom, sim=simulations, phangs=phangs, use_gb21=False, use_wis=True, use_phangs=True, use_sim=False, comb_llama=True,plotshared=False, rebin=120, mask=mask, R_kpc=R_kpc, exclude_names=['NGC 1375'])

        ###### compare on same axis ######

plot_llama_property('Gini', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False, exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'],comb_llama=True,compare=True)
plot_llama_property('Asymmetry', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'],comb_llama=True,compare=True)
plot_llama_property('Asymmetry', 'Gini', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260'],comb_llama=True,compare=True)
plot_llama_property('Gini', 'clumping_factor', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'],comb_llama=True,compare=True)
plot_llama_property('clumping_factor', 'Smoothness', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'],comb_llama=True,compare=True)
plot_llama_property('Asymmetry', 'clumping_factor', AGN_data, inactive_data, agn_Rosario2018, inactive_Rosario2018,GB21_density,wisdom, simulations, phangs,False,exclude_names=['NGC 1375','NGC 1315','NGC 2775','NGC 4260', 'NGC 5845'],comb_llama=True,compare=True)






    ############## galaxy properties WISDOM, PHANGS coplot #############

# nothing considered useful yet

 #   """posible x_column: '"Distance (Mpc)"', 'log LH (L⊙)', 'Hubble Stage', 'Axis Ratio', 'Bar'
 #      posible y_column: 'Smoothness', 'Asymmetry', 'Gini', 'Sigma0', 'rs'"""

stats_table.to_csv(
    "/data/c3040163/llama/alma/gas_analysis_results/Plots/statistics_summary.csv",
    index=False
)
print("Saved all statistics → statistics_summary.csv")

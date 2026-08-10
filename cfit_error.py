import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def fit_concentration_50pc(df,
                           R_col='Resolution (pc)',
                           C_col='Concentration',
                           R_sat=400,
                           C_sat=0.0426,
                           R_target=50,
                           extrapolate_hires=False):

    R = df[R_col]
    C = df[C_col]

    C_fit = C + ((C_sat - C) / (R_sat - R)) * (R_target - R)

    if extrapolate_hires:
        return C_fit
    else:
        return C.where(R < R_target, C_fit)


mask = 'broad'
R_kpc = 1.5

base_AGN = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/AGN"
base_inactive = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/inactive"

AGN_path = f"{base_AGN}/gas_analysis_summary_{mask}_{R_kpc}kpc_rescomp.csv"
inactive_path = f"{base_inactive}/gas_analysis_summary_{mask}_{R_kpc}kpc_rescomp.csv"

# Read the data
fit_data_AGN = pd.read_csv(AGN_path)
fit_data_inactive = pd.read_csv(inactive_path)

# Combine the dataframes
fit_data = pd.concat([fit_data_AGN, fit_data_inactive], ignore_index=True)

# Calculate fitted concentration
fit_data['C_fit'] = fit_concentration_50pc(
    fit_data,
    extrapolate_hires=False
)

# Get the true concentration from the native-resolution measurement for each galaxy
native_C = (
    fit_data.loc[fit_data['resolution_source'] == 'native',
                 ['Galaxy', 'Concentration']]
    .set_index('Galaxy')['Concentration']
)

# Match native concentration to every row by galaxy
fit_data['C_true'] = fit_data['Galaxy'].map(native_C)

# Calculate error relative to native value
#fit_data['C_err'] = fit_data['C_fit'] - fit_data['C_true']
#fit_data['C_err_frac'] = fit_data['C_fit'] / fit_data['C_true']
fit_data['C_err'] = 1-(fit_data['C_fit'] / fit_data['C_true'])
# # Save the result
# output_path = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/gas_analysis_summary_with_Cfit.csv"
# fit_data.to_csv(output_path, index=False)

# print(f"Saved to {output_path}")


summary = (
    fit_data
    .groupby('resolution_source')
    .agg(
        Resolution_pc=('Resolution (pc)', 'first'),
        C_err_mean=('C_err', 'mean'),
        C_err_min=('C_err', 'min'),
        C_err_max=('C_err', 'max'),
        N=('C_err', 'count')
    )
    .sort_values('Resolution_pc')
)
summary = summary[summary['Resolution_pc'] <= 180].copy()

# Asymmetric error bars
yerr = [
    summary['C_err_mean'] - summary['C_err_min'],  # lower
    summary['C_err_max'] - summary['C_err_mean']   # upper
]

plt.figure(figsize=(6,4))

plt.errorbar(
    summary['Resolution_pc'],
    summary['C_err_mean'],
    yerr=yerr,
    fmt='o-',
    capsize=4
)

plt.xlabel('Resolution (pc)')
plt.ylabel('Mean $C_{\\rm err}/C$')
plt.grid(True)

output_path = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/Cfit_error.csv"
summary.to_csv(output_path, index=False)

plt.show()


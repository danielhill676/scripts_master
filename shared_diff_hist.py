import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
})


# -----------------------------
# Δlog10 values (y-axis only)
# -----------------------------
dlog_y = np.array(
[-0.7809999999999999, 0.126, 0.076, -0.15759294953081412, -0.1399999999999999, -0.367, 0.04972451336192674, 0.10972451336192673, 0.20972451336192674, -0.2560988236195731, 0.16799999999999998, -0.062000000000000055, 0.01100000000000001, -0.06899999999999999, -0.089, -0.10899999999999999, -0.0065855889490386565, -0.7911422681081783, -0.1])
# -----------------------------
# Convert to linear factors
# -----------------------------
diffs_frac_conc = 10**dlog_y

diffs_frac_asym = np.array([1.348,1.735,0.522,1.190])
diffs_frac_clump = np.array([2.278,1.041,0.522,1.150])
diffs_frac_gini = np.array([1.155,1.108,0.561,1.191])


for diffs_frac, metric in zip(
    [diffs_frac_conc, diffs_frac_asym, diffs_frac_clump, diffs_frac_gini],
    ['Concentration', 'Asymmetry', 'Clumpiness', 'Gini']
):
    print(metric, diffs_frac)

    # -----------------------------
    # Summary statistics
    # -----------------------------
    mean_frac = np.mean(diffs_frac)
    sigma_frac = np.std(diffs_frac, ddof=1)
    mean_err_frac = sigma_frac / np.sqrt(len(diffs_frac))

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    font = 20


    ax.hist(diffs_frac, bins=12, color="green", histtype='bar', linewidth=4,alpha=0.3)
    ax.axvline(mean_frac, color="red", linestyle="-", label=f"Mean = {mean_frac:.2g} ± {mean_err_frac:.2g}",linewidth=3)
    ax.axvline(mean_frac - sigma_frac, color="blue", linestyle="--")
    ax.axvline(mean_frac + sigma_frac, color="blue", linestyle="--",label=f"$1\sigma = {(sigma_frac):.2g}$" if sigma_frac > 0 else r"$1\sigma = \infty$")
    ax.axvline(1, color="black", linestyle="solid")  # reference line


    ax.set_xlabel(rf"{metric} fractional difference equiv",fontsize=font)
    ax.set_ylabel("Number of measurements",fontsize=font)
    t_stat, p_value_t = ttest_1samp(diffs_frac, 1)
    ax.set_title(f"One-sample t-test p-value: {p_value_t:.4f}", fontsize=font)



    ax.legend()
    plt.tight_layout()
    path = f'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/Plots/Shared_systematics/{metric}.pdf'
    plt.savefig(path)
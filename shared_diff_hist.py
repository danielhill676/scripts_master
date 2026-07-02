import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp


# -----------------------------
# Δlog10 values (y-axis only)
# -----------------------------
dlog_y = np.array([
    -0.20004104036480325,
    -0.7789999999999999,
     0.132,
     0.08199999999999999,
    -0.061,
     0.014000000000000012,
    -0.06599999999999999,
    -0.086,
    -0.10599999999999998,
    -0.029999999999999916,
    -0.9247509045060849,
    -0.141,
     0.136,
    -0.09400000000000003,
    -0.24202056807051914,
    -0.026429895712630352,
     0.03357010428736965,
     0.13357010428736965,
    -0.6352481322164365
])

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

    ax.hist(
        diffs_frac,
        bins=12,
        color="grey",
        alpha=0.6,
        edgecolor=None
    )

    ax.axvline(
        mean_frac,
        color="red",
        linestyle="--",
        label=f"Mean = {mean_frac:.2g} ± {mean_err_frac:.2g}"
    )

    ax.axvline(
        mean_frac - sigma_frac,
        color="blue",
        linestyle="--"
    )

    ax.axvline(
        mean_frac + sigma_frac,
        color="blue",
        linestyle="--",
        label=f"1σ = {sigma_frac:.2g}" if sigma_frac > 0 else r"1σ = ∞"
    )

    ax.axvline(
        1.0,
        color="black",
        linestyle="solid",
        label="No offset (1:1)"
    )

    ax.set_xlabel(rf"{metric} fractional difference equiv")
    ax.set_ylabel("Number of measurements")
    t_stat, p_value_t = ttest_1samp(diffs_frac, 1)
    ax.set_title(f"One-sample t-test p-value: {p_value_t:.4f}", fontsize=16)

    ax.legend()
    plt.tight_layout()
    path = f'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/Plots/Shared_systematics/{metric}'
    plt.savefig(path)
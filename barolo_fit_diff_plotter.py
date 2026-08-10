import os
import numpy as np
import pandas as pd
from IPython.display import display


path_fit = "/Users/administrator/Astro/LLAMA/ALMA/barolo/phangsmask/phangsmask_fit1.csv"
path_used = "/Users/administrator/Astro/LLAMA/ALMA/LLAMA_coords.csv"

fit = pd.read_csv(path_fit)
used = pd.read_csv(path_used)

fit = pd.DataFrame(fit)
used = pd.DataFrame(used)

diff_df = pd.merge(
    fit,
    used,
    left_on="name",
    right_on="Galaxy",
    how="left"
)

diff_df = diff_df.rename(columns={
    "mean_INC_deg": "i_fit",
    "mean_PA_deg": "PA_fit",
    "Inclination (deg)": "i_used",
    "PA (deg)": "PA_used"
})


# ---------------------------------------------------------
# Remap all position angles to 0-360 degrees
# ---------------------------------------------------------

diff_df["PA_fit"] = diff_df["PA_fit"] % 180
diff_df["PA_used"] = diff_df["PA_used"] % 180


# ---------------------------------------------------------
# Calculate differences
# ---------------------------------------------------------

diff_df["i_diff"] = diff_df["i_fit"] - diff_df["i_used"]
# diff_df["PA_diff"] = diff_df["PA_fit"] - diff_df["PA_used"]
diff_df["PA_diff"] = (
    (diff_df["PA_fit"] - diff_df["PA_used"] + 90) % 180
) - 90



display(diff_df)


import matplotlib.pyplot as plt

# Sort independently by each difference
i_sorted = diff_df.sort_values("i_diff", ascending=True)
pa_sorted = diff_df.sort_values("PA_diff", ascending=True)


# ---------------------------------------------------------
# Inclination differences
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 14))

ax.barh(i_sorted["name"], i_sorted["i_diff"])

ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel(r"$i_{\rm fit} - i_{\rm used}$ (deg)")
ax.set_ylabel("Galaxy")
ax.set_title("Inclination differences")

ax.tick_params(axis="y", labelsize=9)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# Position angle differences
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 14))

ax.barh(pa_sorted["name"], pa_sorted["PA_diff"])

ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel(r"$PA_{\rm fit} - PA_{\rm used}$ (deg)")
ax.set_ylabel("Galaxy")
ax.set_title("Position angle differences")

ax.tick_params(axis="y", labelsize=9)
plt.tight_layout()
plt.show()

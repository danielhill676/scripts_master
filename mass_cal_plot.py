import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV
df = pd.read_csv("/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/error_calibration/mass_cal.csv")

# Create the scatter plot
plt.figure(figsize=(6, 6))

plt.scatter(
    df["L'CO_table1 (K km_s pc2)"],
    df["lprimeco_table2"],
    label="L'CO",
    alpha=0.7
)

plt.scatter(
    df["total_mass_table1 (M_sun)"],
    df["masses_table2"],
    label="Mass",
    alpha=0.7
)

# 1:1 line
xmin = min(
    df["L'CO_table1 (K km_s pc2)"].min(),
    df["lprimeco_table2"].min(),
    df["total_mass_table1 (M_sun)"].min(),
    df["masses_table2"].min(),
)
xmax = max(
    df["L'CO_table1 (K km_s pc2)"].max(),
    df["lprimeco_table2"].max(),
    df["total_mass_table1 (M_sun)"].max(),
    df["masses_table2"].max(),
)

plt.plot([xmin, xmax], [xmin, xmax], "k--", label="1:1")


# Label each point by its ID
for _, row in df.iterrows():
    plt.text(
        row["L'CO_table1 (K km_s pc2)"],
        row["lprimeco_table2"],
        row["id"],
        fontsize=8,
        ha="left",
        va="bottom"
    )

    plt.text(
        row["total_mass_table1 (M_sun)"],
        row["masses_table2"],
        row["id"],
        fontsize=8,
        ha="left",
        va="bottom"
    )

# 4/pi reference line
factor = 2.5

plt.plot(
    [xmin, xmax],
    [factor * xmin, factor * xmax],
    "r--",
    label=rf"${factor}$"
)

# plt.grid(True)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("My data")
plt.ylabel("David's check")
plt.legend()

plt.tight_layout()
plt.show()
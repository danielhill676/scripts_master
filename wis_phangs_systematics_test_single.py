import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

wis = pd.read_csv('/Users/administrator/Astro/LLAMA/ALMA/comp_samples/wis_new.csv')
phangs = pd.read_csv('/Users/administrator/Astro/LLAMA/ALMA/comp_samples/phangs_new.csv')
wis_metrics = pd.read_csv('/Users/administrator/Astro/LLAMA/ALMA/comp_samples/m0_metrics_wis.csv')
phangs_metrics = pd.read_csv('/Users/administrator/Astro/LLAMA/ALMA/comp_samples/m0_metrics_phangs.csv')

wis = pd.merge(wis_metrics, wis, left_on='name', right_on='Name', how='left')
phangs = pd.merge(phangs_metrics, phangs, left_on='name', right_on='Name', how='left')

datasets = [("WISDOM", wis), ("PHANGS", phangs)]
columns = ['Gini', 'Asymmetry', 'Smoothness_davis']

fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=False, sharey=False)

for i, (label, df) in enumerate(datasets):
    for j, column in enumerate(columns):
        ax = axes[i, j]

        y = df[column + '_x']
        x = df[column + '_y']

        ax.scatter(x, y)
        xmin ,xmax= min(x),max(x)
        ymin,ymax = min(y), max(y)
        maxmax = max(ymax,xmax)
        ax.plot([0, maxmax], [0, maxmax], "k--")
        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)
        column_title = 'Clumpiness' if column == 'Smoothness_davis' else column
        ax.set_title(f"{column_title} ({label})")

        if i == 1:
            ax.set_xlabel("Davis+22 Value")
        if j == 0:
            ax.set_ylabel("Reproduced value (this work)")

plt.tight_layout()
plt.savefig('/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/error_calibration/davis_systematic.pdf')
# plt.show()






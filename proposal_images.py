import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from astropy.table import Table
import matplotlib.patches as patches


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
})

llamatab = Table.read('/Users/administrator/Astro/LLAMA/llama_main_properties.fits', format='fits')
llamatab.sort('D [Mpc]')

AGN_target = ['NGC4388','NGC1365','NGC6814','NGC7582','NGC5506']
inactive_target = ['NGC3717','NGC5921','NGC4254','NGC4224','NGC3749'] # NGC 4224 and NGC 4260 can also be observed

def figure_maker(fig_y, fig_x, cols, rows, path1, path2, fig_title):
    fig, axes = plt.subplots(rows, cols, figsize=(fig_x, fig_y), constrained_layout=True)

    for idx, agn_name in enumerate(AGN_target):
        col = idx
        row_agn = 0
        row_inactive = 1

        # AGN image (top row)
        agn_img = mpimg.imread(f"{path1}/1_no_rebin_broad_{agn_name}_native.png")
        ax1 = axes[row_agn, col]
        ax1.imshow(agn_img, aspect='auto', interpolation='none')
        ax1.axis('off')
        ax1.set_title(agn_name, fontsize=24, pad=15)
 

        # Inactive galaxy image (bottom row)
        inactive_name = inactive_target[idx]
        inactive_img = mpimg.imread(f"{path2}/1_no_rebin_broad_{inactive_name}_native.png")
        ax2 = axes[row_inactive, col]
        ax2.imshow(inactive_img, aspect='auto', interpolation='none')
        ax2.axis('off')
        ax2.set_title(inactive_name, fontsize=24, pad=15)


    fig.subplots_adjust(hspace=0, wspace=-0)  # slightly increase vertical space between rows
    plt.savefig(f'/Users/administrator/Astro/proposals/ALMA_2026/{fig_title}.png',
                bbox_inches='tight', pad_inches=0.1, format='png')
    plt.close()


fig_x = 18  # wider figure for more columns
fig_y = 8   # shorter figure, now just 2 rows
cols = 5
rows = 2

figure_maker(fig_y, fig_x, cols, rows,
             '/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/AGN/m0_plots',
             '/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/inactive/m0_plots',
             'torus_prop_pairs_rowwise_26')
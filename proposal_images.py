import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from astropy.table import Table
import matplotlib.patches as patches

llamatab = Table.read('/Users/administrator/Astro/LLAMA/llama_main_properties.fits', format='fits')
llamatab.sort('D [Mpc]')

AGN_target = ['NGC7213','NGC7582','NGC6814','NGC5506']
inactive_target = ['NGC2775','NGC3717','NGC4254','NGC3175']

def figure_maker(fig_y, fig_x, cols, rows, path1, path2, fig_title):
    fig, axes = plt.subplots(rows, cols, figsize=(fig_x, fig_y), constrained_layout=True)

    for idx, agn_name in enumerate(AGN_target):
        col = idx
        row_agn = 0
        row_inactive = 1

        # AGN image (top row)
        agn_img = mpimg.imread(f"{path1}/{agn_name}_square.png")
        ax1 = axes[row_agn, col]
        ax1.imshow(agn_img, aspect='auto', interpolation='none')
        ax1.axis('off')
 

        # Inactive galaxy image (bottom row)
        inactive_name = inactive_target[idx]
        inactive_img = mpimg.imread(f"{path2}/{inactive_name}_square.png")
        ax2 = axes[row_inactive, col]
        ax2.imshow(inactive_img, aspect='auto', interpolation='none')
        ax2.axis('off')


    fig.subplots_adjust(hspace=0, wspace=-0)  # slightly increase vertical space between rows
    plt.savefig(f'/Users/administrator/Astro/proposals/ALMA_2025/{fig_title}.png',
                bbox_inches='tight', pad_inches=0.1, format='png')
    plt.close()


fig_x = 15  # wider figure for more columns
fig_y = 8   # shorter figure, now just 2 rows
cols = 4
rows = 2

figure_maker(fig_y, fig_x, cols, rows,
             '/Users/administrator/Astro/LLAMA/ALMA/AGN_images/m8_plots',
             '/Users/administrator/Astro/LLAMA/ALMA/inactive_images/m8_plots',
             'torus_prop_pairs_rowwise')
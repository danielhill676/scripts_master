import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from astropy.table import Table
from matplotlib.gridspec import GridSpec

# ----------------------------
# Load LLAMA table
# ----------------------------
llamatab = Table.read('/Users/administrator/Astro/LLAMA/llama_main_properties.fits', format='fits')
llamatab.sort('D [Mpc]')
llamatab_inactive = llamatab[llamatab['type'] == 'i']
llamatab_AGN = llamatab[llamatab['type'] != 'i']

# ----------------------------
# Existing multi-galaxy figure makers
# ----------------------------
def type_figure_maker(fig_y, fig_x, cols, rows, path, fig_title, type):
    """Create a tiled figure for a given galaxy type and moment map."""
    fig = plt.figure(figsize=(fig_x, fig_y), constrained_layout=True)
    ax = []

    if type == 'AGN':
        table = llamatab_AGN
    else:
        table = llamatab_inactive

    for i in range(len(table)):
        if table['id'][i] == 'IC4653':
            continue
        subplot = mpimg.imread(path + f'/{table["id"][i]}.png')
        ax.append(fig.add_subplot(rows, cols, i + 1))
        plt.imshow(subplot)
        plt.axis('off')

    fig.subplots_adjust(left=0, bottom=0, right=1, top=1,
                        wspace=0, hspace=0.325)
    plt.savefig('/Users/administrator/Astro/LLAMA/ALMA/plots/' +
                f'{fig_title}.png', bbox_inches='tight',
                pad_inches=0.0, format='png')


def matched_figure_maker(fig_y, fig_x, cols, rows, path, fig_title):
    """(Placeholder) Create matched galaxy figures — needs matching table."""
    fig = plt.figure(figsize=(fig_x, fig_y), constrained_layout=True)
    ax = []

    # This is placeholder logic until matching info is provided
    table = llamatab_AGN

    for i in range(len(table)):
        if table['id'][i] == 'IC4653':
            continue
        subplot = mpimg.imread(path + f'/{table["id"][i]}.png')
        ax.append(fig.add_subplot(rows, cols, i + 1))
        plt.imshow(subplot)
        plt.axis('off')

    fig.subplots_adjust(left=0, bottom=0, right=1, top=1,
                        wspace=0, hspace=0.325)
    plt.savefig('/Users/administrator/Astro/LLAMA/ALMA/' +
                f'{fig_title}.png', bbox_inches='tight',
                pad_inches=0.0, format='png')


# ----------------------------
# New function: One figure per galaxy
# ----------------------------
def plot_galaxy_moments(llamatab, path, outpath, skip_ids=None):
    
    path0 = os.path.join(path, 'm0_plots')
    path1 = os.path.join(path, 'm1_plots')
    path2 = os.path.join(path, 'm2_plots')
    
    if skip_ids is None:
        skip_ids = []

    for gal_id in llamatab['id']:
        if gal_id in skip_ids:
            continue

        # File paths
        m0_file = os.path.join(path0, f"{gal_id}.png")
        m1_file = os.path.join(path1, f"{gal_id}.png")
        m2_file = os.path.join(path2, f"{gal_id}.png")

        # Skip if missing
        if not (os.path.exists(m0_file) and os.path.exists(m1_file) and os.path.exists(m2_file)):
            print(f"Skipping {gal_id}: missing moment map(s)")
            continue

        # Load images
        img0 = mpimg.imread(m0_file)
        img1 = mpimg.imread(m1_file)
        img2 = mpimg.imread(m2_file)

        # Get image dimensions (assuming same height for m1/m2)
        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]

        # Width ratio 2:1, height ratio 1:1 for top/bottom right
        total_width = w0 * 2 + w1  # scale so left is double width
        total_height = max(h0, h1 * 2)

        # Scale to a reasonable figure size (e.g., height 6 inches)
        scale = 6 / total_height
        fig_w = total_width * scale
        fig_h = total_height * scale

        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1],
                      wspace=0, hspace=0, figure=fig)

        # Moment 0 (left, spanning both rows)
        ax0 = fig.add_subplot(gs[:, 0])
        ax0.imshow(img0, aspect='equal')
        ax0.axis('off')

        # Moment 1 (top right)
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(img1, aspect='equal')
        ax1.axis('off')

        # Moment 2 (bottom right)
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.imshow(img2, aspect='equal')
        ax2.axis('off')

        plt.savefig(os.path.join(outpath, f"{gal_id}_moments.png"),
                    bbox_inches='tight', pad_inches=0.0, format='png')
        plt.close(fig)

        print(f"Saved plot for {gal_id}")






# ----------------------------
# Run multi-galaxy overview plots
# ----------------------------
fig_x = 10
fig_y = 8
cols = 5
rows = 4

type_figure_maker(fig_y, fig_x, cols, rows,
                  '/Users/administrator/Astro/LLAMA/ALMA/AGN_images/m0_plots',
                  'Moment 0 maps for LLAMA AGN', 'AGN')

type_figure_maker(fig_y, fig_x, cols, rows,
                  '/Users/administrator/Astro/LLAMA/ALMA/AGN_images/m8_plots',
                  'Moment 8 maps for LLAMA AGN', 'AGN')

type_figure_maker(fig_y, fig_x, cols, rows,
                  '/Users/administrator/Astro/LLAMA/ALMA/inactive_images/m0_plots',
                  'Moment 0 maps for LLAMA Inactive galaxies', 'inactive')

type_figure_maker(fig_y, fig_x, cols, rows,
                  '/Users/administrator/Astro/LLAMA/ALMA/inactive_images/m8_plots',
                  'Moment 8 maps for LLAMA Inactive galaxies', 'inactive')

# ----------------------------
# Run one-figure-per-galaxy plots
# ----------------------------
plot_galaxy_moments(
    llamatab,
    path='/Users/administrator/Astro/LLAMA/ALMA/AGN_images',
    outpath='/Users/administrator/Astro/LLAMA/ALMA/plots/Individual_plots/AGN',
    skip_ids=['IC4653']
)

plot_galaxy_moments(
    llamatab,
    path='/Users/administrator/Astro/LLAMA/ALMA/inactive_images',
    outpath='/Users/administrator/Astro/LLAMA/ALMA/plots/Individual_plots/inactive',
    skip_ids=['IC4653']
)



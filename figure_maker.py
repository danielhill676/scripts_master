import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from astropy.table import Table
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec


llamatab = Table.read('/Users/administrator/Astro/LLAMA/llama_main_properties.fits', format='fits')
llamatab.sort('D [Mpc]')
llamatab_inactive = llamatab[llamatab['type'] == 'i']
llamatab_AGN = llamatab[llamatab['type'] != 'i']

def figure_maker(
    fig_y, fig_x, cols, rows, path, fig_title, type,
    m0=True, R_kpc=1.5, rebin=None, mask='strict',
    norm=False, colourbar=False
):

    fig = plt.figure(figsize=(fig_x, fig_y), constrained_layout=True)

    if colourbar:
        gs = GridSpec(rows, cols + 1, figure=fig,
                      width_ratios=[1]*cols + [0.25])
    else:
        gs = GridSpec(rows, cols, figure=fig)

    if type == 'AGN':
        table = llamatab_AGN
    else:
        table = llamatab_inactive

    plot_index = 0

    for i in range(len(table)):

        if table['id'][i] in ['IC4653', 'NGC5128']:
            continue

        row = plot_index // cols
        col = plot_index % cols

        ax = fig.add_subplot(gs[row, col])

        # --- Build file path ---
        if m0 and rebin is None:
            subplot_path = path + f'/{R_kpc}_no_rebin_{mask}_{table["id"][i]}_native'
        elif not m0 and rebin is None:
            subplot_path = path + f'/0.3_no_rebin_{table["id"][i]}_native'
        elif m0 and rebin is not None:
            subplot_path = path + f'/{R_kpc}_{rebin}_{mask}_{table["id"][i]}'
        else:
            subplot_path = path + f'/{R_kpc}_{rebin}_{table["id"][i]}'

        if norm:
            subplot_path += '_norm'

        subplot_path += '.png'
        try:
            subplot = mpimg.imread(subplot_path)
        except:
            return

        ax.imshow(subplot)
        ax.axis('off')
        ax.set_title(table['name'][i], fontsize=11, pad=1)

        plot_index += 1

    # ---------------------------------------
    # Add colourbar spanning all rows
    # ---------------------------------------
    if colourbar:

        flux_flag = True if mask == 'flux90_strict' else False
        colourbar_path = '/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits' + f'/colourbar_{R_kpc}_{rebin}_{flux_flag}.png'
        try:
            cbar_img = mpimg.imread(colourbar_path)
        except:
            return

        cbar_ax = fig.add_subplot(gs[:, -1])  # span all rows, last column
        cbar_ax.imshow(cbar_img)
        cbar_ax.axis('off')

    #fig.suptitle(fig_title, y=0.99)

    plt.savefig(
        '/Users/administrator/Astro/LLAMA/ALMA/' + f'{fig_title}.pdf',
        bbox_inches='tight',
        pad_inches=0.2,
        format='png'
    )

    plt.close(fig)


def stack_with_colourbar(
    top_image,
    bottom_image,
    colourbar_image,
    output_file,
    figsize=(6, 8),
    width_ratios=(1, 0.1),
):
    """
    Stack two images vertically with a colourbar image alongside.

    Parameters
    ----------
    top_image : str
        Path to top image.
    bottom_image : str
        Path to bottom image.
    colourbar_image : str
        Path to tall colourbar image.
    output_file : str
        Output filename.
    figsize : tuple
        Figure size in inches.
    width_ratios : tuple
        Relative widths of image and colourbar columns.
    """

    fig = plt.figure(figsize=figsize, constrained_layout=True)

    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=[1, 1],
    )

    ax_top = fig.add_subplot(gs[0, 0])
    ax_bottom = fig.add_subplot(gs[1, 0])
    ax_cbar = fig.add_subplot(gs[:, 1])

    # Read images
    top = mpimg.imread(top_image)
    bottom = mpimg.imread(bottom_image)
    cbar = mpimg.imread(colourbar_image)

    # Display
    ax_top.imshow(top)
    ax_bottom.imshow(bottom)
    ax_cbar.imshow(cbar)

    # Remove axes
    for ax in (ax_top, ax_bottom, ax_cbar):
        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.close(fig)


    
# fig_x = 12
# fig_y = 10
fig_x = 12
fig_y = 10
cols = 5
rows = 4

for R_kpc in [1.5,0.3]:

    ########### normalised ###########
    figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/AGN/m0_plots',f'Strict mask Moment 0 maps for LLAMA AGN, normalised {2*R_kpc}x{2*R_kpc}kpc','AGN',norm=True,colourbar=False,R_kpc=R_kpc,mask='strict')
    figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/inactive/m0_plots',f'Strict mask Moment 0 maps for LLAMA Inactive galaxies, normalised {2*R_kpc}x{2*R_kpc}kpc','inactive',norm=True,colourbar=False,R_kpc=R_kpc,mask='strict')



    # figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/AGN/m0_plots',f'Moment 0 maps for LLAMA AGN {2*R_kpc}x{2*R_kpc}kpc','AGN',norm=False,colourbar=False,R_kpc=R_kpc)
    # figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/inactive/m0_plots',f'Moment 0 maps for LLAMA Inactive galaxies{2*R_kpc}x{2*R_kpc}kpc','inactive',norm=False,colourbar=False,R_kpc=R_kpc)


    # figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/AGN/m0_plots',f'Strict mask Moment 0 maps for LLAMA AGN {2*R_kpc}x{2*R_kpc}kpc','AGN',rebin=None,mask='strict',R_kpc=R_kpc)
    # figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/inactive/m0_plots',f'Strict mask Moment 0 maps for LLAMA Inactive galaxies {2*R_kpc}x{2*R_kpc}kpc','inactive',rebin=None,mask='strict',R_kpc=R_kpc)

    # figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/AGN/m0_plots',f'flux masked 120pc beam Moment 0 maps for LLAMA AGN {2*R_kpc}x{2*R_kpc}kpc','AGN',rebin=120,mask='flux90_strict',R_kpc=R_kpc)
    # figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/inactive/m0_plots',f'flux masked 120pc beam Moment 0 maps for LLAMA Inactive galaxies {2*R_kpc}x{2*R_kpc}kpc','inactive',rebin=120,mask='flux90_strict',R_kpc=R_kpc)

    # figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/cont_analysis_results/AGN/plots',f'Continuum maps for LLAMA AGN {2*R_kpc}x{2*R_kpc}kpc','AGN',R_kpc=R_kpc,m0=False)
    # figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/cont_analysis_results/inactive/plots',f'Continuum maps for LLAMA Inactive galaxies {2*R_kpc}x{2*R_kpc}kpc','inactive',R_kpc=R_kpc,m0=False)




# plt.show()

stack_with_colourbar(
    top_image="/Users/administrator/Astro/LLAMA/ALMA/Strict mask Moment 0 maps for LLAMA AGN, normalised 3.0x3.0kpc.pdf",
    bottom_image="/Users/administrator/Astro/LLAMA/ALMA/Strict mask Moment 0 maps for LLAMA Inactive galaxies, normalised 3.0x3.0kpc.pdf",
    colourbar_image="/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/colourbar_1.5_None_False.png",
    output_file="/Users/administrator/Astro/LLAMA/ALMA/Strict mask Moment 0 maps for combined LLAMA, normalised 3.0x3.0kpc.pdf",
)
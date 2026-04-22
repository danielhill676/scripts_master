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
        '/Users/administrator/Astro/LLAMA/ALMA/' + f'{fig_title}.png',
        bbox_inches='tight',
        pad_inches=0.2,
        format='png'
    )

    plt.close(fig)


    
# fig_x = 12
# fig_y = 10
fig_x = 12
fig_y = 10
cols = 5
rows = 4

for R_kpc in [1.5]:

    figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/AGN/m0_plots',f'Strict mask Moment 0 maps for LLAMA AGN, normalised {2*R_kpc}x{2*R_kpc}kpc','AGN',norm=True,colourbar=True,R_kpc=R_kpc)
    figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/inactive/m0_plots',f'Strict mask Moment 0 maps for LLAMA Inactive galaxies, normalised {2*R_kpc}x{2*R_kpc}kpc','inactive',norm=True,colourbar=True,R_kpc=R_kpc)

    # figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/AGN/m0_plots',f'Moment 0 maps for LLAMA AGN, normalised {2*R_kpc}x{2*R_kpc}kpc','AGN',norm=False,colourbar=False,R_kpc=R_kpc)
    # figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/inactive/m0_plots',f'Moment 0 maps for LLAMA Inactive galaxies, normalised {2*R_kpc}x{2*R_kpc}kpc','inactive',norm=False,colourbar=False,R_kpc=R_kpc)

    # figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/AGN/m0_plots',f'Moment 0 maps for LLAMA AGN, normalised {2*R_kpc}x{2*R_kpc}kpc','AGN',norm=False,colourbar=False,R_kpc=R_kpc)
    # figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/inactive/m0_plots',f'Moment 0 maps for LLAMA Inactive galaxies, normalised {2*R_kpc}x{2*R_kpc}kpc','inactive',norm=False,colourbar=False,R_kpc=R_kpc)

    figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/AGN/m0_plots',f'Strict mask Moment 0 maps for LLAMA AGN {2*R_kpc}x{2*R_kpc}kpc','AGN',rebin=None,mask='strict',R_kpc=R_kpc)
    figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/inactive/m0_plots',f'Strict mask Moment 0 maps for LLAMA Inactive galaxies {2*R_kpc}x{2*R_kpc}kpc','inactive',rebin=None,mask='strict',R_kpc=R_kpc)

    # figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/AGN/m0_plots',f'flux masked 120pc beam Moment 0 maps for LLAMA AGN {2*R_kpc}x{2*R_kpc}kpc','AGN',rebin=120,mask='flux90_strict',R_kpc=R_kpc)
    # figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/inactive/m0_plots',f'flux masked 120pc beam Moment 0 maps for LLAMA Inactive galaxies {2*R_kpc}x{2*R_kpc}kpc','inactive',rebin=120,mask='flux90_strict',R_kpc=R_kpc)

    # figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/cont_analysis_results/AGN/plots',f'Continuum maps for LLAMA AGN {2*R_kpc}x{2*R_kpc}kpc','AGN',R_kpc=R_kpc,m0=False)
    # figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/cont_analysis_results/inactive/plots',f'Continuum maps for LLAMA Inactive galaxies {2*R_kpc}x{2*R_kpc}kpc','inactive',R_kpc=R_kpc,m0=False)




# plt.show()
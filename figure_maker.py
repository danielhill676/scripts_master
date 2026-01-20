import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from astropy.table import Table
from matplotlib import rcParams

llamatab = Table.read('/Users/administrator/Astro/LLAMA/llama_main_properties.fits', format='fits')
llamatab.sort('D [Mpc]')
llamatab_inactive = llamatab[llamatab['type'] == 'i']
llamatab_AGN = llamatab[llamatab['type'] != 'i']

def figure_maker(fig_y,fig_x,cols,rows,path,fig_title,type,m0=True):
    
    fig = plt.figure(figsize=(fig_x, fig_y), constrained_layout=True)
    ax = []

    if type == 'AGN':
        table = llamatab_AGN
    else:
        table = llamatab_inactive

    for i in range(len(table)):
        if table['id'][i] == 'IC4653' or table['id'][i] == 'NGC5128':
            continue

        if m0 == True:
            subplot = mpimg.imread(
                path + '/1.5_no_rebin_strict_' + f"{table['id'][i]}" + '.png'
            )
        if m0 == False:
            subplot = mpimg.imread(
            path + '/0.3_no_rebin_' + f"{table['id'][i]}" + '.png'
        )

        axi = fig.add_subplot(rows, cols, i + 1)
        ax.append(axi)

        axi.imshow(subplot)
        axi.axis('off')
        axi.set_title(table['name'][i], fontsize=11, pad=1)

        
    fig.subplots_adjust(left=0, bottom=0, right=2, top=2, wspace=0, hspace=0.310)
    if m0==False:
        fig.subplots_adjust(left=0, bottom=0, right=12, top=2, wspace=0, hspace=0)
    rcParams.update({'figure.autolayout': True})
    plt.savefig('/Users/administrator/Astro/LLAMA/ALMA/'+f'{fig_title}.png',bbox_inches='tight',pad_inches=0.0,format='png')
    fig.suptitle(fig_title,y=0.99)
    
# fig_x = 12
# fig_y = 10
fig_x = 12
fig_y = 10
cols = 5
rows = 4

# figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/AGN/m0_plots','Moment 0 maps for LLAMA AGN','AGN')
# figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/inactive/m0_plots','Moment 0 maps for LLAMA Inactive galaxies','inactive')
figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/cont_analysis_results/AGN/plots','Continuum maps for LLAMA AGN','AGN',m0=False)
figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/cont_analysis_results/inactive/plots','Continuum maps for LLAMA Inactive galaxies','inactive',m0=False)


# plt.show()
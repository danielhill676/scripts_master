import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from astropy.table import Table

llamatab = Table.read('/Users/administrator/Astro/LLAMA/llama_main_properties.fits', format='fits')
llamatab.sort('D [Mpc]')
llamatab_inactive = llamatab[llamatab['type'] == 'i']
llamatab_AGN = llamatab[llamatab['type'] != 'i']

def figure_maker(fig_y,fig_x,cols,rows,path,fig_title,type):
    
    fig = plt.figure(figsize=(fig_x, fig_y),constrained_layout=True)
    ax = []

    if type == 'AGN':
        table = llamatab_AGN
    else:
        table = llamatab_inactive
    
    for i in range(len(table)):
        if table['id'][i]=='IC4653':
            continue
        subplot = mpimg.imread(path + f'/{table['id'][i]}'+'.png')
        ax.append(fig.add_subplot(rows, cols, i+1))
        im=plt.imshow(subplot)
        plt.axis('off')
        
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0.325)
    plt.savefig('/Users/administrator/Astro/LLAMA/ALMA/'+f'{fig_title}.png',bbox_inches='tight',pad_inches=0.0,format='png')
    # fig.suptitle(fig_title,y=0.99)
    
fig_x = 10
fig_y = 8
cols = 5
rows = 4

figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/AGN_images/m0_plots','Moment 0 maps for LLAMA AGN','AGN')
figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/AGN_images/m8_plots','Moment 8 maps for LLAMA AGN','AGN')
figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/inactive_images/m0_plots','Moment 0 maps for LLAMA Inactive galaxies','inactive')
figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/inactive_images/m8_plots','Moment 8 maps for LLAMA Inactive galaxies','inactive')


# plt.show()
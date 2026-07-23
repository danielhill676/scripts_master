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
from pdf2image import convert_from_path
import numpy as np
import fitz


llamatab = Table.read('/Users/administrator/Astro/LLAMA/llama_main_properties.fits', format='fits')
llamatab.sort('D [Mpc]')
llamatab_inactive = llamatab[llamatab['type'] == 'i']
llamatab_AGN = llamatab[llamatab['type'] != 'i']

def figure_maker(
    fig_y, fig_x, cols, rows, path, fig_title, type,
    m0=True, R_kpc=1.5, rebin=None, mask='strict',
    norm=False, colourbar=False
):

    if type == 'AGN':
        table = llamatab_AGN
    else:
        table = llamatab_inactive

    # create output PDF
    out = fitz.open()

    # A4-ish page size in points
    page_width = fig_x * 72
    page_height = fig_y * 72

    page = out.new_page(
        width=page_width,
        height=page_height
    )

    # grid spacing
    margin = 30
    cbar_width = 60 if colourbar else 0

    # gaps between panels
    h_gap = 7   # horizontal spacing between columns
    v_gap = 12.5   # vertical spacing between rows (increase this for more vertical separation)

    plot_width = (
        page_width
        - 2*margin
        - cbar_width
        - (cols-1)*h_gap
    ) / cols

    plot_height = (
        page_height
        - 2*margin
        - (rows-1)*v_gap
    ) / rows


    plot_index = 0

    for i in range(len(table)):

        if table['id'][i] in ['IC4653', 'NGC5128']:
            continue

        row = plot_index // cols
        col = plot_index % cols

        # --- Build file path ---
        if m0 and rebin is None:
            subplot_path = (
                path +
                f'/{R_kpc}_no_rebin_{mask}_{table["id"][i]}_native'
            )
        elif not m0 and rebin is None:
            subplot_path = (
                path +
                f'/0.3_no_rebin_{table["id"][i]}_native'
            )
        elif m0 and rebin is not None:
            subplot_path = (
                path +
                f'/{R_kpc}_{rebin}_{mask}_{table["id"][i]}'
            )
        else:
            subplot_path = (
                path +
                f'/{R_kpc}_{rebin}_{table["id"][i]}'
            )

        if norm:
            subplot_path += '_norm'

        subplot_path += '.pdf'


        try:
            pdf = fitz.open(subplot_path)
        except:
            continue


        x0 = margin + col*(plot_width + h_gap)
        y0 = margin + row*(plot_height + v_gap)

        rect = fitz.Rect(
            x0,
            y0,
            x0 + plot_width,
            y0 + plot_height
        )

        # insert the PDF page directly
        page.show_pdf_page(
            rect,
            pdf,
            0
        )

        text = table['name'][i]

        text_width = fitz.get_text_length(
            text,
            fontsize=11
        )

        x = x0 + (plot_width - text_width)/2

        # put text above subplot
        y = y0 - 2

        page.insert_text(
            (x, y),
            text,
            fontsize=11
        )
        plot_index += 1


    # Add colourbar
    if colourbar:

        flux_flag = mask == 'flux90_strict'

        colourbar_path = (
            '/Users/administrator/Astro/LLAMA/ALMA/'
            'gas_distribution_fits'
            f'/colourbar_{R_kpc}_{rebin}_{flux_flag}.pdf'
        )

        cbar_pdf = fitz.open(colourbar_path)

        cbar_rect = fitz.Rect(
            page_width-cbar_width,
            margin,
            page_width-margin,
            page_height-margin
        )

        page.show_pdf_page(
            cbar_rect,
            cbar_pdf,
            0
        )


    out.save(
        f'/Users/administrator/Astro/LLAMA/ALMA/{fig_title}.pdf'
    )

    out.close()


def stack_with_colourbar(
    top_image,
    bottom_image,
    colourbar_image,
    output_file,
):

    # Open PDFs
    top_pdf = fitz.open(top_image)
    bottom_pdf = fitz.open(bottom_image)
    cbar_pdf = fitz.open(colourbar_image)

    # Create new PDF
    out = fitz.open()

    # Page sizes
    top_page = top_pdf[0]
    width = top_page.rect.width
    height = top_page.rect.height

    # Create new page (roughly double height + colourbar space)
    new_page = out.new_page(
        width=width * 1.15,
        height=height * 2
    )

    # Define placements
    top_rect = fitz.Rect(
        0,
        0,
        width,
        height
    )

    bottom_rect = fitz.Rect(
        0,
        height,
        width,
        height * 2
    )

    cbar_rect = fitz.Rect(
        width,
        0,
        width * 1.15,
        height * 2
    )

    # Insert PDFs without rasterisation
    new_page.show_pdf_page(
        top_rect,
        top_pdf,
        0
    )

    new_page.show_pdf_page(
        bottom_rect,
        bottom_pdf,
        0
    )

    new_page.show_pdf_page(
        cbar_rect,
        cbar_pdf,
        0
    )

    out.save(output_file)
    out.close()

    
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



    figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/AGN/m0_plots',f'Strict mask Moment 0 maps for LLAMA AGN {2*R_kpc}x{2*R_kpc}kpc','AGN',rebin=None,mask='strict',R_kpc=R_kpc)
    figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/inactive/m0_plots',f'Strict mask Moment 0 maps for LLAMA Inactive galaxies {2*R_kpc}x{2*R_kpc}kpc','inactive',rebin=None,mask='strict',R_kpc=R_kpc)

    # figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/AGN/m0_plots',f'flux masked 120pc beam Moment 0 maps for LLAMA AGN {2*R_kpc}x{2*R_kpc}kpc','AGN',rebin=120,mask='flux90_strict',R_kpc=R_kpc)
    # figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/inactive/m0_plots',f'flux masked 120pc beam Moment 0 maps for LLAMA Inactive galaxies {2*R_kpc}x{2*R_kpc}kpc','inactive',rebin=120,mask='flux90_strict',R_kpc=R_kpc)

    # figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/cont_analysis_results/AGN/plots',f'Continuum maps for LLAMA AGN {2*R_kpc}x{2*R_kpc}kpc','AGN',R_kpc=R_kpc,m0=False)
    # figure_maker(fig_y,fig_x,cols,rows,'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/cont_analysis_results/inactive/plots',f'Continuum maps for LLAMA Inactive galaxies {2*R_kpc}x{2*R_kpc}kpc','inactive',R_kpc=R_kpc,m0=False)




# plt.show()

stack_with_colourbar(
    top_image="/Users/administrator/Astro/LLAMA/ALMA/Strict mask Moment 0 maps for LLAMA AGN, normalised 3.0x3.0kpc.pdf",
    bottom_image="/Users/administrator/Astro/LLAMA/ALMA/Strict mask Moment 0 maps for LLAMA Inactive galaxies, normalised 3.0x3.0kpc.pdf",
    colourbar_image="/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/colourbar_1.5_None_False.pdf",
    output_file="/Users/administrator/Astro/LLAMA/ALMA/Strict mask Moment 0 maps for combined LLAMA, normalised 3.0x3.0kpc.pdf",
)
import os
import fitz  # PyMuPDF
import pandas as pd
print(fitz.__version__)
import re


def label_mom0(name,type,r,rebin,mask,extra_text,rescomp):

    if rebin is not None:
        image_path = (
            f'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/'
            f'{type}/m0_plots/{r}_{rebin}_{mask}_{name}.pdf'
        )
        table_path = (
            f'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/'
            f'{type}/gas_analysis_summary_{rebin}pc_{mask}_{r}kpc.csv'
        ) 
    else:
        image_path = (
            f'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/'
            f'{type}/m0_plots/{r}_no_rebin_{mask}_{name}.pdf'
        )
        table_path = (
            f'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/'
            f'{type}/gas_analysis_summary_{mask}_{r}kpc.csv'
        ) if not rescomp else (
            f'/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/'
            f'{type}/gas_analysis_summary`_{mask}_{r}kpc_rescomp.csv'
        )

    image_outpath = image_path.replace(".pdf", "_labelled.pdf")


    # ---------------- Read table ----------------

    print('loading', table_path)

    table = pd.read_csv(table_path)
    
    search_name = name.split('_')[0] if type != 'aux' else name
    res_source = name.split('_')[1] if rebin == None else 'rebin'
    
    row = table.loc[
    (table["id"] == search_name) &
    (table["resolution_source"] == res_source)
].iloc[0]

    C = row["Concentration"]
    A = row["Asymmetry"]
    S = row["Smoothness_davis"]
    G = row["Gini"]

    text = (
        f"C = {C:.2f}\n"
        f"A = {A:.2f}\n"
        f"S = {S:.2f}\n"
        f"G = {G:.2f}"
    )


    # ---------------- Open PDF ----------------

    doc = fitz.open(image_path)
    page = doc[0]

    page_rect = page.rect

    # ---------------- Font size ----------------

    n_lines = 4

    # Approximate line spacing = 1.2 × fontsize
    fontsize = 180

    # ---------------- Position ----------------

    padding = 100  # points from bottom-left corner

    line_spacing = fontsize * 1.2
    text_height = (len(text.splitlines()) - 1) * line_spacing

    x = padding - 20
    y = page_rect.height - padding - text_height

    # ---------------- Colours ----------------

    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))

    text_colour = hex_to_rgb("#02FF00")
    outline_colour = hex_to_rgb("#053061")


    # ---------------- Draw text with outline ----------------

    shape = page.new_shape()

    offset = 2.5

    for dx, dy in [
        (-offset, 0), (offset, 0),
        (0, -offset), (0, offset),
        (-offset, -offset), (-offset, offset),
        (offset, -offset), (offset, offset),
    ]:
        shape.insert_text(
            fitz.Point(x + dx, y + dy),
            text,
            fontsize=fontsize,
            fontname="Times-Roman",
            fill=outline_colour, fill_opacity=0.2
        )

    shape.insert_text(
        fitz.Point(x, y),
        text,
        fontsize=fontsize,
        fontname="Times-Roman",
        fill=text_colour,
    )

    # ---------------- Add extra text top-right ----------------

    if extra_text.strip():

        extra_fontsize = 120
        extra_padding = 5

        # Get text width for right alignment
        extra_width = fitz.get_text_length(
            extra_text,
            fontname="Times-Roman",
            fontsize=extra_fontsize
        )

        extra_x = page_rect.width - extra_width - extra_padding
        extra_y = extra_padding + extra_fontsize * 0.2


        offset = 2.5

        for dx, dy in [
            (-offset, 0), (offset, 0),
            (0, -offset), (0, offset),
            (-offset, -offset), (-offset, offset),
            (offset, -offset), (offset, offset),
        ]:
            shape.insert_text(
                fitz.Point(extra_x + dx, extra_y + dy),
                extra_text,
                fontsize=extra_fontsize,
                fontname="Times-Roman",
                fill=outline_colour, fill_opacity=0.2
            )

        shape.insert_text(
            fitz.Point(extra_x, extra_y),
            extra_text,
            fontsize=extra_fontsize,
            fontname="Times-Roman",
            fill=text_colour,
        )


    shape.commit()

    doc.save(image_outpath)
    doc.close()

    print(f"Saved {image_outpath}") 



name = 'NGC4254_8'
type = 'inactive'
r = 1.5
rebin = None
mask = 'strict'
rescomp = True
extra_text = """

"""
label_mom0(name,type,r,rebin,mask,extra_text,rescomp)
# for name in ['NGC5506_native','NGC5506_NGC4260','NGC5506_NGC3749']:
#     label_mom0(name,type,r,rebin,mask,extra_text)


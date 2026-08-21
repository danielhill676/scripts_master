import os
import fitz


# ==========================================================================================
# RUN NAME
# ==========================================================================================
runname = 'phangsmask_cenfroz_axisfree'
# =========================================================================================
# RUN NUMBER
# ==========================================================================================
runn = 3
# =========================================================================================

BASE_DIR = f"/Users/administrator/Astro/LLAMA/ALMA/barolo/{runname}/plots"
COLOURBAR_DIR = f"/Users/administrator/Astro/LLAMA/ALMA/barolo/{runname}/colourbars"


def figure_maker(
    n,
    norm=False,
    fig_x=12,
    panel_height=2.5,
    margin=25,
    name_width=25,
    h_gap=8,
    v_gap=8,
    colourbar_width=35,
    colourbar_gap=8,
    output_file=None,
):

    if n not in [0, 1, 2]:
        raise ValueError("n must be 0, 1, or 2")

    suffix = "_norm.pdf" if norm else ".pdf"

    # ------------------------------------------------------------
    # Find galaxy directories
    # ------------------------------------------------------------

    galaxies = []

    for name in sorted(os.listdir(BASE_DIR)):

        galaxy_dir = os.path.join(BASE_DIR, name)

        if not os.path.isdir(galaxy_dir):
            continue

        files = [
            os.path.join(
                galaxy_dir,
                f"{name}_true_mom{n}{suffix}"
            ),
            os.path.join(
                galaxy_dir,
                f"{name}_fit_mom{n}{suffix}"
            ),
            os.path.join(
                galaxy_dir,
                f"{name}_res_mom{n}{suffix}"
            ),
        ]

        if all(os.path.isfile(f) for f in files):

            # IMPORTANT: exactly two items
            galaxies.append(
                (name, files)
            )

        else:

            missing = [
                os.path.basename(f)
                for f in files
                if not os.path.isfile(f)
            ]

            print(
                f"Skipping {name}: missing "
                + ", ".join(missing)
            )

    if not galaxies:
        raise RuntimeError(
            f"No complete sets of moment {n} PDFs found."
        )

    print(f"Using {len(galaxies)} galaxies")

    # ------------------------------------------------------------
    # Determine natural aspect ratio of the moment PDFs
    # ------------------------------------------------------------

    sample_pdf = fitz.open(galaxies[0][1][0])
    sample_page = sample_pdf[0]

    source_width = sample_page.rect.width
    source_height = sample_page.rect.height

    aspect_ratio = source_width / source_height

    sample_pdf.close()

    # ------------------------------------------------------------
    # Page dimensions
    # ------------------------------------------------------------

    rows = len(galaxies)

    panel_height_pt = panel_height * 72
    panel_width_pt = panel_height_pt * aspect_ratio

    # Colourbar dimensions
    cbar_width_pt = 25
    cbar_gap = 5

    page_width = (
        2 * margin
        + name_width
        + 3 * panel_width_pt
        + 2 * h_gap
    )

    if norm:
        page_width += cbar_gap + cbar_width_pt

    page_height = (
        2 * margin
        + rows * panel_height_pt
        + (rows - 1) * v_gap
    )
    
    # ------------------------------------------------------------
    # Colourbar dimensions
    #
    # Only add this column for norm=True.
    # ------------------------------------------------------------

    if norm:

        colourbar_width_pt = colourbar_width

        total_colourbar_space = (
            colourbar_gap
            + colourbar_width_pt
        )

    else:

        total_colourbar_space = 0

    # ------------------------------------------------------------
    # Page width
    # ------------------------------------------------------------
    cbar_gap = 5
    cbar_width_pt = 35  # narrow colourbar width

    page_width = (
        2 * margin
        + name_width
        + 3 * panel_width_pt
        + 2 * h_gap
        + cbar_gap
        + cbar_width_pt
    )

    page_height = (
        2 * margin
        + rows * panel_height_pt
        + (rows - 1) * v_gap
    )

    # ------------------------------------------------------------
    # Create PDF
    # ------------------------------------------------------------

    out = fitz.open()

    page = out.new_page(
        width=page_width,
        height=page_height
    )

    # ------------------------------------------------------------
    # Column headings
    # ------------------------------------------------------------

    headings = [
        "True",
        "Fit",
        "Residual"
    ]

    for col, heading in enumerate(headings):

        x0 = (
            margin
            + name_width
            + col * (panel_width_pt + h_gap)
        )

        text_width = fitz.get_text_length(
            heading,
            fontsize=11
        )

        page.insert_text(
            (
                x0 + (panel_width_pt - text_width) / 2,
                margin - 7
            ),
            heading,
            fontsize=11
        )

    # ------------------------------------------------------------
    # Colourbar heading
    # ------------------------------------------------------------

    if norm:

        colourbar_x = (
            margin
            + name_width
            + 3 * panel_width_pt
            + 2 * h_gap
            + colourbar_gap
        )

        text = "Colourbar"

        text_width = fitz.get_text_length(
            text,
            fontsize=9
        )

        page.insert_text(
            (
                colourbar_x
                + (colourbar_width_pt - text_width) / 2,
                margin - 7
            ),
            text,
            fontsize=9
        )

    # ------------------------------------------------------------
    # Galaxy rows
    # ------------------------------------------------------------

    for row, (name, files) in enumerate(galaxies):

        y0 = (
            margin
            + row * (panel_height_pt + v_gap)
        )

        # --------------------------------------------------------
        # Galaxy name
        # --------------------------------------------------------

        page.insert_text(
            (
                margin + name_width - 5,
                y0 + panel_height_pt / 2
            ),
            name,
            fontsize=9,
            rotate=90,
        )

        # --------------------------------------------------------
        # Three panels
        # --------------------------------------------------------

        for col, pdf_file in enumerate(files):

            x0 = (
                margin
                + name_width
                + col * (panel_width_pt + h_gap)
            )

            rect = fitz.Rect(
                x0,
                y0,
                x0 + panel_width_pt,
                y0 + panel_height_pt
            )

            pdf = fitz.open(pdf_file)

            page.show_pdf_page(
                rect,
                pdf,
                0,
                keep_proportion=True
            )

            pdf.close()

            # Thin black frame
            page.draw_rect(
                rect,
                color=(0, 0, 0),
                width=0.5,
                overlay=True
            )

        # --------------------------------------------------------
        # Colourbar
        # --------------------------------------------------------

        if norm:

            colourbar_file = os.path.join(
                BASE_DIR,
                "..",
                "colourbars",
                f"{name}_colourbar_mom{n}.pdf"
            )

            colourbar_file = os.path.abspath(
                colourbar_file
            )

            if not os.path.isfile(colourbar_file):

                print(
                    f"Missing colourbar for {name}: "
                    f"{colourbar_file}"
                )

            else:

                # x-position of residual panel
                residual_x0 = (
                    margin
                    + name_width
                    + 2 * (panel_width_pt + h_gap)
                )

                # Colourbar immediately to the right
                cbar_x0 = (
                    residual_x0
                    + panel_width_pt
                    + cbar_gap
                )

                cbar_rect = fitz.Rect(
                    cbar_x0,
                    y0,
                    cbar_x0 + cbar_width_pt,
                    y0 + panel_height_pt
                )

                cbar_pdf = fitz.open(
                    colourbar_file
                )

                page.show_pdf_page(
                    cbar_rect,
                    cbar_pdf,
                    0,
                    keep_proportion=True
                )

                cbar_pdf.close()

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------

    if output_file is None:

        norm_string = "_norm" if norm else ""

        output_file = os.path.join(
            BASE_DIR,
            f"all_mom{n}{norm_string}.pdf"
        )

    out.save(output_file)
    out.close()

    print("\nSaved:")
    print(output_file)


# ================================================================
# Run
# ================================================================

figure_maker(
    n=0,
    norm=False
)

figure_maker(
    n=0,
    norm=True
)

figure_maker(
    n=1,
    norm=False
)

figure_maker(
    n=1,
    norm=True
)

figure_maker(
    n=2,
    norm=False
)

figure_maker(
    n=2,
    norm=True
)
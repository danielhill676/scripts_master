import os
import pandas as pd
from astropy.io import fits

base_dir = "/Users/administrator/Astro/LLAMA/ALMA/pipeline_cubes"

AGN_table_dir = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/AGN/gas_analysis_summary_broad_1.5kpc.csv"
inactive_table_dir = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/inactive/gas_analysis_summary_broad_1.5kpc.csv"

fit_data_AGN = pd.read_csv(AGN_table_dir)
fit_data_inactive = pd.read_csv(inactive_table_dir)

outbase = "/Users/administrator/Astro/LLAMA/ALMA/barolo/mapsonly"
os.makedirs(outbase, exist_ok=True)

# ----------------------------------------------------------
# Execution script
# ----------------------------------------------------------

execfile = os.path.join(outbase, "barolo_execute.sh")

with open(execfile, "w") as f:
    f.write("""#!/bin/bash
shopt -s expand_aliases
source ~/.zshrc

""")

os.chmod(execfile, 0o755)


# ----------------------------------------------------------
# Loop over galaxies
# ----------------------------------------------------------

for name in sorted(os.listdir(base_dir)):

    subdir = os.path.join(base_dir, name)

    if not os.path.isdir(subdir):
        continue

    print(f"Processing {name}")

    file_pbcorr = os.path.join(
        subdir,
        f"{name}_12m_co21_pbcorr_trimmed.fits"
    )

    file_pb = os.path.join(
        subdir,
        f"{name}_12m_co21_trimmed_pb.fits"
    )

    if not (os.path.exists(file_pbcorr) and os.path.exists(file_pb)):
        print("    Missing cube or PB map.")
        continue
    # ------------------------------------------------------
    # Select correct catalogue
    # ------------------------------------------------------

    if name in fit_data_AGN["Galaxy"].values:
        table = fit_data_AGN

    elif name in fit_data_inactive["Galaxy"].values:
        table = fit_data_inactive

    else:
        print("    Galaxy not found in tables.")
        continue

    row = table[
        (table["Galaxy"] == name)
        & (table["resolution_source"] == "native")
    ]

    if len(row) == 0:
        print("    Native-resolution row not found.")
        continue


    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import numpy as np

    cube, header = fits.getdata(file_pbcorr, header=True)
    pb = fits.getdata(file_pb)

    # ------------------------------------------------------
    # Galaxy information
    # ------------------------------------------------------

    RA = row["RA (deg)"].iloc[0]
    DEC = row["DEC (deg)"].iloc[0]
    D_Mpc = row["D_Mpc"].iloc[0]

    R_kpc = 1.5          # radius you want
    pixel_scale_arcsec = abs(header["CDELT2"]) * 3600.

    R_pixel = int(R_kpc * (206.265 / D_Mpc) / pixel_scale_arcsec)

    # ------------------------------------------------------
    # Find galaxy centre
    # ------------------------------------------------------

    wcs = WCS(header).celestial
    centre = SkyCoord(RA*u.deg, DEC*u.deg)

    cx, cy = centre.to_pixel(wcs)

    cx = np.asarray(cx).item()
    cy = np.asarray(cy).item()

    cx = int(round(cx))
    cy = int(round(cy))

    # ------------------------------------------------------
    # Spatial limits
    # ------------------------------------------------------

    ny, nx = cube.shape[-2:]

    x1 = max(0, cx - R_pixel)
    x2 = min(nx, cx + R_pixel)

    y1 = max(0, cy - R_pixel)
    y2 = min(ny, cy + R_pixel)

    # ------------------------------------------------------
    # Crop cube
    # ------------------------------------------------------

    cube = cube[:, y1:y2, x1:x2]
    pb   = pb[:, y1:y2, x1:x2]

    # ------------------------------------------------------
    # Update WCS
    # ------------------------------------------------------

    header["CRPIX1"] -= x1
    header["CRPIX2"] -= y1
    header["NAXIS1"] = cube.shape[2]
    header["NAXIS2"] = cube.shape[1]



    # ------------------------------------------------------
    # Output folder
    # ------------------------------------------------------

    outsubdir = os.path.join(outbase, name)
    os.makedirs(outsubdir, exist_ok=True)

    trimmed_cube = os.path.join(outsubdir, f"{name}_trimmed.fits")
    fits.writeto(trimmed_cube, cube, header, overwrite=True)

    # ------------------------------------------------------
    # Parameter file
    # ------------------------------------------------------

    parfile = os.path.join(outsubdir, f"{name}.par")

    with open(parfile, "w") as f:

        f.write(f"""# ===================================================
# BBarolo moment-map generation only
# No fitting performed
# ===================================================

FITSFILE      {trimmed_cube}
OUTFOLDER     {outsubdir}

THREADS       8

# ---------------------------------------------------
# Products
# ---------------------------------------------------

TOTALMAP      true
VELOCITYMAP   true
DISPERSIONMAP true

MAPTYPE       MOMENT

RMSMAP        true
SNMAP         true

# ---------------------------------------------------
# Masking
# ---------------------------------------------------

NORM          LOCAL

MASK          SMOOTH&SEARCH
FACTOR        3
BLANKCUT      5

# ---------------------------------------------------
# End
# ---------------------------------------------------
""")

    # ------------------------------------------------------
    # Append command
    # ------------------------------------------------------

    with open(execfile, "a") as f:
        f.write(f'cd "{outsubdir}"\n')
        f.write(f'bbarolo -p "{parfile}"\n\n')

    print(f"    Wrote {parfile}")

print("\nFinished writing parameter files.")
print(f"Run:\n\n{execfile}")

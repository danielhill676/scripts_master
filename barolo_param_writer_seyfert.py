from astropy.table import Table
from astropy.io import fits
import math
import os
from astroquery.ipac.ned import Ned
import time
import requests
from astroquery.exceptions import RemoteServiceError

import os
import pandas as pd
from astropy.io import fits
import math

AGN_table_dir = "/data/c3040163/llama/alma/gas_analysis_results/AGN/gas_analysis_summary_broad_1.5kpc.csv"
inactive_table_dir = "/data/c3040163/llama/alma/gas_analysis_results/inactive/gas_analysis_summary_broad_1.5kpc.csv"

fit_data_AGN = pd.read_csv(AGN_table_dir)
fit_data_inactive = pd.read_csv(inactive_table_dir)

outbase = "/data/c3040163/llama/alma/barolo/phangsmask"
os.makedirs(outbase, exist_ok=True)

def format_coord(value):
    val_str = str(value)
    if not val_str.startswith('-'):
        val_str = '+' + val_str
    return val_str + 'd'

# ----------------------------------------------------------
# Execution script
# ----------------------------------------------------------

execfile = os.path.join(outbase, "barolo_execute.sh")

with open(execfile, "w") as f:
    f.write("""#!/bin/bash
shopt -s expand_aliases
source ~/.bashrc

""")

os.chmod(execfile, 0o755)
BBAROLO_EXE = "/data/c3040163/apps/BBarolo"   # absolute path

# ----------------------------------------------------------
# Loop over galaxies
# ----------------------------------------------------------


def barolo_param_writer(co32=False):

    if not co32:
        base_dir = "/data/c3040163/llama/alma/phangs_imaging_scripts-master/full_run_newkeys_all_arrays/reduction/derived"
    else:
        base_dir = "/data/c3040163/llama/alma/phangs_imaging_scripts-master/CO32_all_arrays/reduction/derived"
    for name in sorted(os.listdir(base_dir)):

        subdir = os.path.join(base_dir, name)

        if not os.path.isdir(subdir):
            continue

        if name in ['NGC5845','NGC3351','NGC4254']:
            continue
        print(f"Processing {name}")

        if not co32:
            file = os.path.join(
                subdir,
                f"{name}_12m_co21.fits"
            )
            mask_file = os.path.join(
            subdir,
            f"{name}_12m_co21_strictmask.fits"
        )
        else:
            file = os.path.join(
                subdir,
                f"{name}_12m_co32.fits"
            )
            mask_file = os.path.join(
            subdir,
            f"{name}_12m_co32_strictmask.fits")



        if not (os.path.exists(file) and os.path.exists(mask_file)):
            print("    Missing cube or mask.")
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

        cube, header = fits.getdata(file, header=True)
        mask = fits.getdata(mask_file)

        print('cube',cube.shape)
        print('mask',mask.shape)



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
        mask = mask[:, y1:y2, x1:x2]

        # ------------------------------------------------------
        # Update WCS
        # ------------------------------------------------------

        header["CRPIX1"] -= x1
        header["CRPIX2"] -= y1
        header["NAXIS1"] = cube.shape[2]
        header["NAXIS2"] = cube.shape[1]
        BMAJ = header.get("BMAJ", 0) * 3600.0  # deg -> arcsec


        nkpc = 1.5
        R_kpc = nkpc * (206.265 / D_Mpc)
        NRADII = math.floor(R_kpc / (2.5 * BMAJ)) if BMAJ > 0 else 1 # RADSEP changed from 1 to 1.5
        LINEAR = 0.425  # ALMA typical

        RA_hex  = format_coord(float(RA))
        DEC_hex = format_coord(float(DEC))

        # ------------------------------------------------------
        # Output folder
        # ------------------------------------------------------

        outsubdir = os.path.join(outbase, name)
        os.makedirs(outsubdir, exist_ok=True)

        trimmed_cube = os.path.join(outsubdir, f"{name}_trimmed.fits")
        trimmed_mask = os.path.join(outsubdir, f"{name}_trimmed_mask.fits")
        fits.writeto(trimmed_mask,cube,header, overwrite=True)

        # mask = mask.astype(bool)
        # masked_cube = cube.copy()
        # masked_cube[~mask] = 0.0

        fits.writeto(trimmed_cube,cube,header, overwrite=True)

        # ------------------------------------------------------
        # Parameter file
        # ------------------------------------------------------

        parfile = os.path.join(outsubdir, f"{name}.par")

        with open(parfile, "w") as f:

            f.write(f"""# ===================================================
    # ===================================================

    FITSFILE      {trimmed_cube}
    OUTFOLDER     {outsubdir}

    THREADS       24
    3DFIT       true
    NRADII      {NRADII}
    RADSEP      {(2.5*BMAJ):.3f}

    XPOS        {RA_hex}
    YPOS        {DEC_hex}


    #VROT        200
    #VDISP       10
    VRAD        0

    NORM        LOCAL
    MASK        {trimmed_mask}       

    TOTALMAP      true
    VELOCITYMAP   true
    DISPERSIONMAP true

    MAPTYPE       MOMENT

    RMSMAP        true
    SNMAP         true


    FREE        VROT VDISP PA INC

    TWOSTAGE    true
    REGTYPE     bezier
    #FTYPE       2
    #WFUNC       2
    LINEAR      {LINEAR}
    #SIDE        B
    FLAGERRORS  false
    BADOUT      true
    NORMALCUBE  true
    DISTANCE    {D_Mpc}
    """)

        # ------------------------------------------------------
        # Append command
        # ------------------------------------------------------

        with open(execfile, "a") as f:
            f.write(f'cd "{outsubdir}"\n')
            f.write(f'/data/c3040163/apps/BBarolo -p "{parfile}"\n\n')

        print(f"    Wrote {parfile}")

barolo_param_writer(co32=False)
barolo_param_writer(co32=True)

print("\nFinished writing parameter files.")
print(f"Run:\n\n{execfile}")


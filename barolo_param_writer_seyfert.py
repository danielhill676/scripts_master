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

run1_data_cenfroz_dir = '/data/c3040163/llama/alma/barolo/phangsmask/phangsmask_fit1.csv'
run1_data_cenfree_dir = '/data/c3040163/llama/alma/barolo/phangsmask_cenfree_axisfree/phangsmask_cenfree_axisfree_fit1.csv'


fit_data_AGN = pd.read_csv(AGN_table_dir)
fit_data_inactive = pd.read_csv(inactive_table_dir)
run1_data_cenfroz = pd.read_csv(run1_data_cenfroz_dir)
run1_data_cenfree = pd.read_csv(run1_data_cenfree_dir)




# ==========================================================================================
# RUN NAME
# ==========================================================================================
runname = 'phangsmask_cenfroz_axisfree_zfree'
# ==========================================================================================


outbase = f"/data/c3040163/llama/alma/barolo/{runname}"
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


def barolo_param_writer(co32=False, froz_centre = True, froz_axis_def = False, z0_pc = 10, zfree = True, vradfree = False, exclude_failed = False): # z0_def in pc scale height

    if not co32:
        base_dir = "/data/c3040163/llama/alma/phangs_imaging_scripts-master/full_run_newkeys_all_arrays/reduction/derived"
    else:
        base_dir = "/data/c3040163/llama/alma/phangs_imaging_scripts-master/CO32_all_arrays/reduction/derived"
    for name in sorted(os.listdir(base_dir)):

        subdir = os.path.join(base_dir, name)

        if not os.path.isdir(subdir):
            continue

# --------------------- Exclusions ------------------------------------------


        # if name not in ['NGC4254','NGC3351']:
        #     continue

        # if name not in ['NGC5728']:
        #     continue

        # Current targets which are not working for cenfree_axisfree or cenfroz_axisfree
        if exclude_failed:
            if name in ['NGC1079', 'NGC1947', 'NGC4235', 'NGC4260', 'NGC718', 'NGC3351', 'NGC4254']:
                continue

        # too low snr, or irrelevant
        if name in ['NGC2775','NGC1315','NGC1375','NGC5845','ngc1365_phangs','ngc2775_phangs','ngc3351_phangs','ngc4254_phangs','NGC5064_wis','NGC5128','NGC7172_wis','NGC1387_wis']:
            continue

        print(f"Processing {name}")
        froz_axis = froz_axis_def

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
            print("Galaxy not found in tables.")
            continue

        row = table[
            (table["Galaxy"] == name)
            & (table["resolution_source"] == "native")
        ]

        if len(row) == 0:
            print("    Native-resolution row not found.")
            continue

        if name in ['NGC1365','NGC3783','NGC4224','NGC4388','NGC7172']:
            axis_table = run1_data_cenfree
        else:
            axis_table = run1_data_cenfroz

        row_axis = axis_table[
            (axis_table["name"] == name)
        ]
        if froz_axis and len(row_axis) == 0:
            print(" No pre-fitted kinematic axis found")
            froz_axis = False

     
        from astropy.wcs import WCS
        from astropy.coordinates import SkyCoord
        from astropy import units as u
        import numpy as np

        cube, header = fits.getdata(file, header=True)
        mask = fits.getdata(mask_file)

        # print('cube',cube.shape)
        # print('mask',mask.shape)



        # ------------------------------------------------------
        # Galaxy information
        # ------------------------------------------------------

        RA = row["RA (deg)"].iloc[0]
        DEC = row["DEC (deg)"].iloc[0]
        D_Mpc = row["D_Mpc"].iloc[0]

        i = 'None'
        PA = 'None'

        try:
            i = row_axis['mean_INC_deg'].iloc[0]
            PA = row_axis['mean_PA_deg'].iloc[0]
        except:
            try:
                i = row['Inclination (deg)']
                PA = row['PA (deg)']
            except:
                i = 'None'
                PA = 'None'


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


        nkpc = 8
        R_arcsec = nkpc * (206.265 / D_Mpc)
        NRADII = math.floor(R_arcsec / (2.5 * BMAJ)) if BMAJ > 0 else 1
        LINEAR = 0.425  # ALMA typical

        Z0 = (z0_pc / 1000) * (206.265 / D_Mpc)
        zfree_write = 'Z0'  if zfree else ''
        pafree_write = 'PA' if not froz_axis else ''
        ifree_write = 'INC' if not froz_axis else ''
        vradfree_write = 'VRAD' if vradfree else ''

        RA_hex  = format_coord(float(RA)) if froz_centre else 'None'
        DEC_hex = format_coord(float(DEC)) if froz_centre else 'None'

        # ------------------------------------------------------
        # Output folder
        # ------------------------------------------------------

        outsubdir = os.path.join(outbase, name)
        os.makedirs(outsubdir, exist_ok=True)
        # trimmed_cube_subdir = "/data/c3040163/llama/alma/barolo/phangsmask/"+name
        # trimmed_cube = os.path.join(trimmed_cube_subdir, f"{name}_trimmed.fits")
        # trimmed_mask = os.path.join(trimmed_cube_subdir, f"{name}_trimmed_mask.fits")
        trimmed_cube = os.path.join(outsubdir, f"{name}_trimmed.fits")
        trimmed_mask = os.path.join(outsubdir, f"{name}_trimmed_mask.fits") 
        fits.writeto(trimmed_mask,mask,header, overwrite=True)

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
    PA          {PA}
    INC         {i}
    # PA          180
    # INC         25
    Z0          {Z0}

    NORM        LOCAL
    MASK        file({trimmed_mask})       

    TOTALMAP      true
    VELOCITYMAP   true
    DISPERSIONMAP true

    MAPTYPE       MOMENT

    RMSMAP        true
    SNMAP         true


    FREE        VROT VDISP {pafree_write} {ifree_write} {zfree_write} {vradfree_write}

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


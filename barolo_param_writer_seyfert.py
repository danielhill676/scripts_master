from astropy.table import Table
from astropy.io import fits
import math
import os
from astroquery.ipac.ned import Ned
import time
import requests
from astroquery.exceptions import RemoteServiceError

co32 = False

# ---------------------------------------------------------------------
# Load LLAMA table
# ---------------------------------------------------------------------
llamatab = Table.read(
    '/data/c3040163/llama/llama_main_properties.fits',
    format='fits'
)

# ---------------------------------------------------------------------
# Helper: add sign to RA/DEC
# ---------------------------------------------------------------------
def format_coord(value):
    val_str = str(value)
    if not val_str.startswith('-'):
        val_str = '+' + val_str
    return val_str + 'd'


# ---------------------------------------------------------------------
# Main parameter file writer
# ---------------------------------------------------------------------
def write_barolo_params(rootfolder, outfolder):

    os.makedirs(outfolder, exist_ok=True)

    # -----------------------------------------------------------------
    # ALWAYS RECREATE barolo_execute.sh
    # -----------------------------------------------------------------
    execfile = os.path.join(outfolder, 'barolo_execute.sh')

    with open(execfile, 'w') as f:
        f.write(
"""#!/bin/bash
# Auto-generated BBarolo runner
# Rewritten every time this script is executed

"""
        )
    os.chmod(execfile, 0o755)

    BBAROLO_EXE = "/data/c3040163/apps/BBarolo"   # absolute path


    # -----------------------------------------------------------------
    # Walk through subdirectories looking for {name}/{name}_12m_co21.fits
    # -----------------------------------------------------------------
    for name in os.listdir(rootfolder):
        subdir = os.path.join(rootfolder, name)
        if not os.path.isdir(subdir):
            continue

        cube_file = os.path.join(subdir, f"{name}_12m_co21.fits")
        if not os.path.exists(cube_file):
            print(f"❌ No cube found for {name} in {subdir}")
            continue

        print(f"🔍 Found cube: {cube_file}")

        # Lookup in LLAMA table
        row = llamatab[llamatab['id'] == name]
        if len(row) == 0:
            print(f"⚠️  Skipping {name}: ID not found in llama_main_properties.fits")
            continue

        # Extract FITS header beam
        header = fits.getheader(cube_file)
        BMAJ = header.get("BMAJ", 0) * 3600.0  # deg -> arcsec
        Xcen = header.get("CRPIX1", 0.0)
        print(f"Xcen: {Xcen}")
        Ycen = header.get("CRPIX2", 0.0)
        print(f"Ycen: {Ycen}")

        PA   = row['PA'][0]
        D_Mpc = row['D [Mpc]'][0]
        inc   = row['Inclination (deg)'][0]
        name_full = row['name'][0]

        # -------------------------------------------------------------
        # Query NED (with retry)
        # -------------------------------------------------------------
        max_retries = 3
        for attempt in range(max_retries):
            try:
                Ned_table = Ned.query_object(name_full)
                RA  = format_coord(float(Ned_table['RA'][0]))
                DEC = format_coord(float(Ned_table['DEC'][0]))

                if "Velocity" in Ned_table.colnames:
                    vsys = float(Ned_table["Velocity"][0])
                elif "Redshift" in Ned_table.colnames:
                    if not co32:
                        vsys = float(Ned_table["Redshift"][0]) * 299792.458
                    else:
                        vsys = float(Ned_table["Redshift"][0]) * 345796
                else:
                    vsys = 0.0
                break

            except (requests.exceptions.ConnectionError,
                    requests.exceptions.ReadTimeout,
                    RemoteServiceError) as e:
                print(f"⚠️  NED failed for {name} (attempt {attempt+1}/3): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    print("❌ Using fallback LLAMA coordinates")
                    RA  = format_coord(float(row['RA (deg)'][0]))
                    DEC = format_coord(float(row['DEC (deg)'][0]))
                    vsys = 0.0

        # -------------------------------------------------------------
        # Compute NRADII
        # -------------------------------------------------------------
        nkpc = 2.5
        R_kpc = nkpc * (206.265 / D_Mpc)
        NRADII = math.floor(R_kpc / (1.5 * BMAJ)) if BMAJ > 0 else 1 # RADSEP changed from 1 to 1.5
        LINEAR = 0.425  # ALMA typical

        # -------------------------------------------------------------
        # Write .par file
        # -------------------------------------------------------------
        parfile = os.path.join(outfolder, f"{name}.par")
        with open(parfile, 'w') as f:
            f.write(f"""
# BBarolo parameter file for {name}
FITSFILE    {cube_file}
THREADS     4

3DFIT       true
NRADII      {NRADII}
RADSEP      {(2.5*BMAJ):.3f}

XPOS        {Xcen}
YPOS        {Ycen}
INC         {inc}
PA          {PA}

VROT        200
VDISP       10
VRAD        0
VSYS        {vsys:.2f}

FREE        VROT VDISP PA INC

NORM        AZIM
MASK        SMOOTH&SEARCH
FACTOR      1.5 
BLANKCUT    2 

TWOSTAGE    false
REGTYPE     auto
FTYPE       2
WFUNC       2
LINEAR      {LINEAR}
SIDE        B
FLAGERRORS  false
BADOUT      true
NORMALCUBE  true
DISTANCE    {D_Mpc}
""")

        print(f"✅ Wrote {parfile}")

        # -------------------------------------------------------------
        # Append to execute script
        # -------------------------------------------------------------
        with open(execfile, 'a') as f:
            f.write(f"{BBAROLO_EXE} -p {name}.par\n")

    print("✔️ Finished writing parameter files and barolo_execute.sh.")


# ---------------------------------------------------------------------
# Run it
# ---------------------------------------------------------------------

if not co32:
    indir = "/data/c3040163/llama/alma/phangs_imaging_scripts-master/full_run_newkeys_all_arrays/reduction/derived/"
else:
    indir = "/data/c3040163/llama/alma/phangs_imaging_scripts-master/CO32_all_arrays/reduction/derived/"

write_barolo_params(
    indir,
    "/data/c3040163/llama/alma/barolo"
)

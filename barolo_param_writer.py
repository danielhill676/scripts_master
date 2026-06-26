from astropy.table import Table
from astropy.io import fits
import math
import os
from astroquery.ipac.ned import Ned
import time
import requests
from astroquery.exceptions import RemoteServiceError

# Load LLAMA main properties table
llamatab = Table.read(
    '/Users/administrator/Astro/LLAMA/llama_main_properties.fits',
    format='fits'
)

# Optional imaging list (not used directly in loop, but could filter)
imaging_list = [
    "ESO021", "ESO093", "ESO137", "ESO208", "MCG514", "MCG523", "MCG630", "NGC1079",
    "NGC1315", "NGC1365", "NGC1375", "NGC1947", "NGC2110", "NGC2775", "NGC2992", "NGC3081",
    "NGC3175", "NGC3351", "NGC3717", "NGC3749", "NGC3783", "NGC4224", "NGC4235", "NGC4254",
    "NGC4260", "NGC4593", "NGC5037", "NGC5182", "NGC5506", "NGC5728", "NGC5845", "NGC5921",
    "NGC6814", "NGC7172", "NGC718", "NGC7213", "NGC7582", "NGC7727"
]


def format_coord(value):
    """Ensure RA/DEC values have explicit + or - and add 'd' for degrees."""
    val_str = str(value)
    if not val_str.startswith('-'):
        val_str = '+' + val_str
    return val_str + 'd'

def write_barolo_params(infolder, outfolder):

    os.makedirs(outfolder, exist_ok=True)

    # Initialize barolo_execute.sh once per outfolder
    execfile = os.path.join(outfolder, 'barolo_execute.sh')
    if not os.path.exists(execfile):
        with open(execfile, 'w') as f:
            f.write("""
#!/bin/bash
shopt -s expand_aliases
source ~/.zshrc  # or ~/.bashrc if you use bash
            """)
        os.chmod(execfile, 0o755)  # make executable

    for file in os.listdir(infolder):
        if not file.endswith('.fits'):
            continue

        name = file.removesuffix('.subcube.fits')
        filepath = os.path.join(infolder, file)

        # Cross-match with LLAMA table
        row = llamatab[llamatab['id'] == name]
        if len(row) == 0:
            print(f"⚠️  Skipping {name}: not found in llama_main_properties.fits")
            continue

        # Load FITS header of cleaned cube
        header = fits.getheader(filepath)
        BMAJ = header.get("BMAJ", 0) * 3600  # degrees → arcsec

        name_full = row['name'][0]

        PA = row['PA'][0]
        D_Mpc = row['D [Mpc]'][0]
        inc = row['Inclination (deg)'][0]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                Ned_table = Ned.query_object(name_full)
                RA = format_coord(float(Ned_table['RA'][0]))
                DEC = format_coord(float(Ned_table['DEC'][0]))
                if "Velocity" in Ned_table.colnames:
                    vsys = float(Ned_table["Velocity"][0])
                elif "Redshift" in Ned_table.colnames:
                    vsys = float(Ned_table["Redshift"][0]) * 299792.458
                else:
                    vsys = 0.0
                break  # ✅ success, exit retry loop

            except (requests.exceptions.ConnectionError,
                    requests.exceptions.ReadTimeout,
                    RemoteServiceError) as e:
                print(f"⚠️  NED query failed for {name_full} "
                    f"(attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print("Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print("❌ All NED attempts failed for "
                        f"{name_full}.")

                    # Fallback: use coordinates already in llamatab if available
                    if 'RA (deg)' in llamatab.colnames and 'DEC (deg)' in llamatab.colnames:
                        RA = float(row['RA (deg)'][0])
                        DEC = float(row['DEC (deg)'][0])
                        print(f"Using fallback RA/DEC from llamatab "
                            f"for {name_full}.")
                    else:
                        print(f"Skipping {name_full} — no coordinates available.")
                        continue  # skip this galaxy if no coordinates at all




        # Calculate NRADII
        n_kpc = 1.0
        R = n_kpc * (206.265 / D_Mpc)  # 1 kpc in arcsec
        NRADII = math.floor(R / BMAJ) if BMAJ > 0 else 1
        LINEAR = 0.425  # typical for ALMA channelization

        # --- Write parameter file with new FITSFILE path ---
        parfile = os.path.join(outfolder, f"{name}.par")
        with open(parfile, 'w') as f:
            f.write(f"""
# BBarolo 3DFIT parameter file (simple model for inner 1 kpc)
# -------------------------
FITSFILE    {filepath}
THREADS     8

/////////// 3DFIT parameters /////////////
3DFIT       false
#NRADII      {NRADII}               # integer, computed (see notes)
#RADSEP      {BMAJ:.3f}             # set ≈ BMAJ (beam major FWHM)

# Geometry (FIX these to external/photometric values)
XPOS        {RA}
YPOS        {DEC}
INC         {inc}
PA          {PA}

# Kinematic initial guesses (sensible starting values)
VROT        200
VDISP       10
VRAD        0
VSYS        {vsys:.2f}

# Which parameters to fit (minimal)
FREE        VROT VDISP PA INC

# Normalization & mask (emphasis on kinematics)
NORM        LOCAL
MASK        SMOOTH&SEARCH
FACTOR      2
BLANKCUT    3

# Fitting controls and instrument params
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
///////////////////////////////////////////
""")

        # Append command to execution script
        with open(execfile, 'a') as f:
            f.write(f"\nbbarolo -p {name}.par")
        print(f"✅ Wrote {filepath}, {parfile}, and appended command to barolo_execute.sh")

os.system('rm -f /Users/administrator/Astro/LLAMA/ALMA/barolo/barolo_execute.sh')
# Run for active + inactive galaxies
write_barolo_params(
    '/Users/administrator/Astro/LLAMA/ALMA/AGN/subcubes',
    '/Users/administrator/Astro/LLAMA/ALMA/barolo'
)
write_barolo_params(
    '/Users/administrator/Astro/LLAMA/ALMA/CO32/AGN/subcubes',
    '/Users/administrator/Astro/LLAMA/ALMA/barolo'
)
write_barolo_params(
    '/Users/administrator/Astro/LLAMA/ALMA/inactive/subcubes',
    '/Users/administrator/Astro/LLAMA/ALMA/barolo'
)




galaxy_vsys = {
    "ESO021": 2950.0,
    "ESO093": 1831.0,
    "ESO137": 2843.0,
    "ESO208": 1085.0,
    "MCG514": 2972.0,
    "MCG523": 2559.0,
    "MCG630": 2323.0,
    "NGC1079": 1452.0,
    "NGC1315": 1615.0,
    "NGC1365": 1636.0,
    "NGC1375": 777.0,
    "NGC1947": 1100.0,
    "NGC2110": 2353.0,
    "NGC2775": 1350.0,
    "NGC2992": 2311.0,
    "NGC3081": 2443.0,
    "NGC3175": 1087.0,
    "NGC3351": 778.0,
    "NGC3717": 1733.0,
    "NGC3749": 2702.0,
    "NGC3783": 2917.0,
    "NGC4224": 2585.0,
    "NGC4235": 2263.0,
    "NGC4254": 2407.0,
    "NGC4260": 1776.0,
    "NGC4593": 2492.0,
    "NGC5037": 1857.0,
    "NGC5128": 547.0,
    "NGC5506": 1824.0,
    "NGC5728": 2754.0,
    "NGC5845": 1664.0,
    "NGC5921": 1480.0,
    "NGC6814": 1565.0,
    "NGC7172": 2616.0,
    "NGC718": 1733.0,
    "NGC7213": 1750.0,
    "NGC7582": 1622.0,
    "NGC7727": 1795.0,
}





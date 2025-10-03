from astropy.table import Table
from astropy.io import fits
import math
import os

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

def make_barolo_fits(infile, outfile):
    with fits.open(infile, memmap=True) as hdul:
        hdu = hdul[0]

        # Work with the header only (don’t load all data)
        header = hdu.header.copy()

        # Derive ndim safely (squeeze only changes shape, not data)
        ndim = hdu.data.squeeze().ndim  

        # Update NAXIS and clean out old NAXISj
        header["NAXIS"] = ndim
        for key in list(header.keys()):
            if key.startswith("NAXIS") and key != "NAXIS":
                del header[key]

        # Add back correct NAXISj entries
        shape = hdu.data.squeeze().shape
        for i in range(ndim):
            header[f"NAXIS{i+1}"] = shape[-(i+1)]

        # Ensure float32, but write lazily without expanding in RAM
        # Use scale_back to force BITPIX = -32
        hdu.header["BITPIX"] = -32  

        # Write only the header + reference to data
        hdu_scaled = fits.PrimaryHDU(data=hdu.data, header=header)
        hdu_scaled.scale("float32")  # avoid casting whole cube in RAM
        hdu_scaled.writeto(outfile, overwrite=True)

    print(f"✅ Wrote cleaned Barolo FITS: {outfile}")

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
            f.write("#!/bin/bash\n")

    for file in os.listdir(infolder):
        if not file.endswith('.fits'):
            continue

        name = file.removesuffix('.fits')
        filepath = os.path.join(infolder, file)

        # Cross-match with LLAMA table
        row = llamatab[llamatab['id'] == name]
        if len(row) == 0:
            print(f"⚠️  Skipping {name}: not found in llama_main_properties.fits")
            continue

        barolo_fits_path = infolder+'/barolo/'+file

        if name in ['NGC7582']:

            make_barolo_fits(filepath,barolo_fits_path)

        # Load FITS header of cleaned cube
        header = fits.getheader(barolo_fits_path)
        BMAJ = header.get("BMAJ", 0) * 3600  # degrees → arcsec

        PA = row['PA'][0]
        D_Mpc = row['D [Mpc]'][0]
        inc = row['Inclination (deg)'][0]

        RA = format_coord(row['RA (deg)'][0])
        DEC = format_coord(row['DEC (deg)'][0])

        # Calculate NRADII
        R = 206.265 / D_Mpc  # 1 kpc in arcsec
        NRADII = math.floor(R / BMAJ) if BMAJ > 0 else 1
        LINEAR = 0.425  # typical for ALMA channelization

        # --- Write parameter file with new FITSFILE path ---
        parfile = os.path.join(outfolder, f"{name}.par")
        with open(parfile, 'w') as f:
            f.write(f"""
# BBarolo 3DFIT parameter file (simple model for inner 1 kpc)
# -------------------------
FITSFILE    {barolo_fits_path}
THREADS     4

/////////// 3DFIT parameters /////////////
3DFIT       true
NRADII      {NRADII}               # integer, computed (see notes)
RADSEP      {BMAJ:.3f}             # set ≈ BMAJ (beam major FWHM)

# Geometry (FIX these to external/photometric values)
XPOS        {RA}
YPOS        {DEC}
INC         {inc}
PA          {PA}

# Kinematic initial guesses (sensible starting values)
VROT        200
VDISP       10
VRAD        0

# Which parameters to fit (minimal)
FREE        VROT VDISP

# Normalization & mask (emphasis on kinematics)
NORM        AZIM
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
            f.write(f"bbarolo -p {name}.par\n")
        print(f"✅ Wrote {barolo_fits_path}, {parfile}, and appended command to barolo_execute.sh")


# Run for active + inactive galaxies
write_barolo_params(
    '/Users/administrator/Astro/LLAMA/ALMA/AGN',
    '/Users/administrator/Astro/LLAMA/ALMA/barolo'
)
write_barolo_params(
    '/Users/administrator/Astro/LLAMA/ALMA/inactive',
    '/Users/administrator/Astro/LLAMA/ALMA/barolo'
)









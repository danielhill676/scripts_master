import os
import math
import numpy as np
import pandas as pd
from pymakeplots import pymakeplots
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from astropy import units as u

base_dir = "/Users/administrator/Astro/LLAMA/ALMA/pipeline_cubes"

AGN_table_dir = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/AGN/gas_analysis_summary_broad_1.5kpc.csv"
inactive_table_dir = "/Users/administrator/Astro/LLAMA/ALMA/gas_distribution_fits/inactive/gas_analysis_summary_broad_1.5kpc.csv"
fit_data_AGN = pd.read_csv(AGN_table_dir)
fit_data_inactive = pd.read_csv(inactive_table_dir)

outbase = "/Users/administrator/Astro/LLAMA/ALMA/barolo/mapsonly"
os.makedirs(outbase, exist_ok=True)

# Initialize barolo_execute.sh once per outfolder
execfile = os.path.join(outbase, 'barolo_execute.sh')
if not os.path.exists(execfile):
    with open(execfile, 'w') as f:
        f.write("""
#!/bin/bash
shopt -s expand_aliases
source ~/.zshrc  # or ~/.bashrc if you use bash
        """)
    os.chmod(execfile, 0o755)  # make executable

for name in os.listdir(base_dir):
    subdir = os.path.join(base_dir, name)

    if not os.path.isdir(subdir):
        continue
        
    outsubdir = os.path.join(outbase, name)

    # if os.path.exists(outsubdir+'/_allplots.pdf'):
    #     continue

    print(f"Processing {name}")

    file_pbcorr = os.path.join(subdir, f"{name}_12m_co21_pbcorr_trimmed.fits")
    file_pb = os.path.join(subdir, f"{name}_12m_co21_trimmed_pb.fits")

    if not (os.path.exists(file_pbcorr) and os.path.exists(file_pb)):
        print(f"{name}: missing files")
        continue



    if name in fit_data_AGN["Galaxy"].values:
        table = fit_data_AGN
    elif name in fit_data_inactive["Galaxy"].values:
        table = fit_data_inactive
    else:
        print(f"{name} not found in tables")
        continue

    cube, hdr_cube = fits.getdata(file_pbcorr, header=True)
    pb, hdr_pb = fits.getdata(file_pb, header=True)
    if cube.shape != pb.shape:
        print('mismatched cube and pb')
        continue 

    
    row = table[
    (table["Galaxy"] == name) &
    (table["resolution_source"] == "native")
]

    RA = row["RA (deg)"].iloc[0]
    DEC = row["DEC (deg)"].iloc[0]
    D = row["D_Mpc"].iloc[0]

    pixelscale_deg_x = 

    size_arcsec = 206.265 * 1.5 / D
    radius_pix_x = size_arcsec / pixscale_arcsec_x
    radius_pix_x = size_arcsec / pixscale_arcsec_x

    # --- Write parameter file with new FITSFILE path ---
    parfile = os.path.join(outbase, f"{name}.par")
    with open(parfile, 'w') as f:
        f.write(f"""
# BBarolo 3DFIT parameter file (simple model for inner 1 kpc)
# -------------------------
FITSFILE    {file}
THREADS     8

# Moment maps
TOTALMAP true
VELOCITYMAP true
DISPERSIONMAP true
MAPTYPE MOMENT
RMSMAP true
SNMAP true

# Normalization & mask (emphasis on kinematics)
NORM        LOCAL
MASK        SMOOTH&SEARCH
FACTOR      2
BLANKCUT    3

///////////////////////////////////////////
""")
            # Append command to execution script
    with open(execfile, 'a') as f:
        f.write(f"\nbbarolo -p {name}.par")
    print(f"✅ Wrote {file}, {parfile}, and appended command to barolo_execute.sh")

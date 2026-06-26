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

ranges = {
    "ESO021": [51, 114],
    "ESO093": [77, 115],
    "ESO137": [68, 125],
    "ESO208": [50, 136],
    "MCG514": [80, 120],
    "MCG523": [62, 120],
    "MCG630": [65, 92],
    "NGC718": [79, 112],
    "NGC1079": [74, 113],
    "NGC1315": [0, 0],
    #"NGC1365": [50, 93],
    "NGC1365": [118, 161],
    "NGC1375": [0, 0],
    "NGC1947": [72, 138],
    "NGC2110": [49, 94],
    "NGC2775": [0, 0],
    "NGC2992": [50, 96],
    "NGC3081": [57, 91],
    "NGC3175": [83, 116],
    "NGC3351": [20, 50],
    "NGC3717": [69, 126],
    "NGC3749": [63, 131],
    "NGC3783": [84, 106],
    "NGC4224": [66, 125],
    "NGC4235": [65, 124],
    "NGC4254": [20, 46],
    "NGC4260": [76, 122],
    "NGC4593": [49, 96],
    "NGC5037": [66, 133],
    "NGC5506": [60, 103],
    "NGC5728": [54, 107],
    "NGC5845": [83, 119],
    "NGC5921": [80, 118],
    #"NGC6814": [137, 162],
    "NGC6814": [87, 112],
    "NGC7172": [37, 104],
    "NGC7213": [50, 90],
    "NGC7582": [1, 105],
    "NGC7727": [74, 132],
}

fit_data_AGN = pd.read_csv(AGN_table_dir)
fit_data_inactive = pd.read_csv(inactive_table_dir)

cwd = os.getcwd()
outbase = "/Users/administrator/Astro/LLAMA/ALMA/pymakeplots/"

for name in os.listdir(base_dir):
    subdir = os.path.join(base_dir, name)

    if not os.path.isdir(subdir):
        continue
        
    outsubdir = os.path.join(outbase, name)

    # if os.path.exists(outsubdir+'/_allplots.pdf'):
    #     continue

    if name not in['NGC6814']:
        continue

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

    if row.empty:
        print(f"No matching row for {name}")
        continue

    RA = row["RA (deg)"].iloc[0]
    DEC = row["DEC (deg)"].iloc[0]
    D = row["D_Mpc"].iloc[0]
    PA = row["PA (deg)"].iloc[0]

    theta = math.degrees(math.atan(1500 / (D * 1e6)))
    theta_arcsec = theta * 3600



    plotter = pymakeplots(cube=file_pbcorr, pb=file_pb)



    plotter.obj_ra = RA
    plotter.obj_dec = DEC
    plotter.gal_distance=D
    if not ranges[name][0] == 0 and not ranges[name][1] == 0:
        plotter.chans2do = ranges[name]
    else:
        continue
    plotter.posang = PA
    plotter.imagesize=[theta_arcsec, theta_arcsec]

    os.makedirs(outsubdir, exist_ok=True)

    os.chdir(outsubdir)

    print("plotting...")
    plotter.make_all(pdf=True, fits=True)

os.chdir(cwd)

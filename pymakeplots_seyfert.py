import os
import math
import numpy as np
import pandas as pd
from pymakeplots import pymakeplots
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from astropy import units as u

base_dir = "/data/c3040163/llama/alma/phangs_imaging_scripts-master/full_run_newkeys_all_arrays/reduction/postprocess"

AGN_table_dir = "/data/c3040163/llama/alma/gas_analysis_results/AGN/gas_analysis_summary_broad_1.5kpc.csv"
inactive_table_dir = "/data/c3040163/llama/alma/gas_analysis_results/inactive/gas_analysis_summary_broad_1.5kpc.csv"

fit_data_AGN = pd.read_csv(AGN_table_dir)
fit_data_inactive = pd.read_csv(inactive_table_dir)

cwd = os.getcwd()
outbase = "/data/c3040163/llama/alma/gas_analysis_results/pymakeplots/"

for name in os.listdir(base_dir):
    subdir = os.path.join(base_dir, name)

    if not os.path.isdir(subdir):
        continue

    print(f"Processing {name}")

    file_pbcorr = os.path.join(subdir, f"{name}_12m_co21_pbcorr_trimmed.fits")
    file_pb = os.path.join(subdir, f"{name}_12m_co21_trimmed_pb.fits")

    if not (os.path.exists(file_pbcorr) and os.path.exists(file_pb)):
        print(f"{name}: missing files")
        continue

    header = fits.getheader(file_pbcorr)

    nchan = header["NAXIS3"]
    mid_chan = nchan // 2

    ref_freq = 230.538*10e9 * u.Hz
    chan_width = header["CDELT3"] * u.Hz

    dv = chan_width.to(u.km/u.s, equivalencies=u.doppler_radio(ref_freq)).value

    vwidth = 1000  # km/s
    chan_offset = int(np.round(vwidth / dv))

    chan_min = max(0, mid_chan - chan_offset)
    chan_max = min(nchan - 1, mid_chan + chan_offset)

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
    plotter.chans2do = [chan_min, chan_max]
    plotter.posang = PA
    plotter.imagesize=[theta_arcsec, theta_arcsec]

    outsubdir = os.path.join(outbase, name)
    os.makedirs(outsubdir, exist_ok=True)

    os.chdir(outsubdir)

    print("plotting...")
    plotter.make_all(pdf=True, fits=True)

os.chdir(cwd)

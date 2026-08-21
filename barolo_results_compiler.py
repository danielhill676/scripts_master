#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table


# ==========================================================================================
# RUN NAME
runname = 'phangsmask_cenfroz_axisfree_zfree'
# =========================================================================================
# RUN NUMBER
runn = 4
# MACHINE
seyfert = True
# ==========================================================================================

# Input directory containing {name} subdirectories
outerdir = f"/Users/administrator/Astro/LLAMA/ALMA/barolo/{runname}" if not seyfert else f"/data/c3040163/llama/alma/barolo/{runname}" 

llamatab = Table.read('/data/c3040163/llama/llama_main_properties.fits',format='fits') if seyfert else Table.read('/Users/administrator/Astro/LLAMA/llama_main_properties.fits',format='fits')

# Output file
output_csv = outerdir+f"/{runname}_fit{runn}.csv"


def process_rings_file(
    filepath,
    D_Mpc,
    pixel_scale_arcsec,
    threshold=0.5,
    R_kpc=1.3,
    npoints=180
):
    """
    Read rings_final2.txt and calculate mean fitted parameters using
    only rings for which at least `threshold` fraction of the projected
    ring circumference lies inside the 1.5-kpc image footprint.

    Parameters
    ----------
    filepath : str
        Path to rings_final2.txt.

    D_Mpc : float
        Galaxy distance in Mpc.

    pixel_scale_arcsec : float
        Image pixel scale in arcsec/pixel.

    threshold : float
        Minimum fraction of ring circumference inside the image.
        0.5 = 50%, 0.8 = 80%, etc.

    R_kpc : float
        Radius of the image region in kpc. Default = 1.5 kpc.

    npoints : int
        Number of points sampled around each ring.
    """

    columns = [
        "RAD_Kpc",
        "RAD_arcs",
        "VROT",
        "DISP",
        "INC",
        "PA",
        "Z0_pc",
        "Z0_arcs",
        "SIG_E20",
        "XPOS",
        "YPOS",
        "VSYS",
        "VRAD",
        "FITOK"
    ]

    try:
        data = pd.read_csv(
            filepath,
            comment="#",
            sep=r"\s+",
            names=columns
        )

    except Exception as e:
        print(f"Failed reading {filepath}: {e}")
        return None

    # ------------------------------------------------------
    # Image dimensions from distance and physical radius
    # ------------------------------------------------------

    R_arcsec = R_kpc * (206.265 / D_Mpc)

    diameter_arcsec = 2 * R_arcsec

    nx = int(round(diameter_arcsec / pixel_scale_arcsec))
    ny = nx

    # ------------------------------------------------------
    # Keep only successful fits
    # ------------------------------------------------------

    data = data[data["FITOK"] == 1].copy()

    if len(data) == 0:
        return None

    # ------------------------------------------------------
    # Sample points around each projected ring
    # ------------------------------------------------------

    theta = np.linspace(
        0,
        2 * np.pi,
        npoints,
        endpoint=False
    )

    valid_rings = []

    for _, ring in data.iterrows():

        R = ring["RAD_arcs"]
        inc = np.deg2rad(ring["INC"])
        PA = np.deg2rad(ring["PA"])

        cx = ring["XPOS"]
        cy = ring["YPOS"]

        # --------------------------------------------------
        # Project circular ring into an ellipse
        # --------------------------------------------------

        a = R
        b = R * np.cos(inc)

        x = a * np.cos(theta)
        y = b * np.sin(theta)

        # --------------------------------------------------
        # Rotate ellipse by PA
        # --------------------------------------------------

        x_rot = (
            x * np.cos(PA)
            - y * np.sin(PA)
        )

        y_rot = (
            x * np.sin(PA)
            + y * np.cos(PA)
        )

        # --------------------------------------------------
        # Convert arcsec -> pixels
        # --------------------------------------------------

        x_pix = cx + x_rot / pixel_scale_arcsec
        y_pix = cy + y_rot / pixel_scale_arcsec

        # --------------------------------------------------
        # Determine which points are inside image
        # --------------------------------------------------

        inside = (
            (x_pix >= 0)
            & (x_pix < nx)
            & (y_pix >= 0)
            & (y_pix < ny)
        )

        intersection_fraction = np.mean(inside)

        # --------------------------------------------------
        # Keep ring if enough is inside
        # --------------------------------------------------

        if intersection_fraction >= threshold:
            valid_rings.append(ring)

    # ------------------------------------------------------
    # No valid rings
    # ------------------------------------------------------

    if len(valid_rings) == 0:
        print(
            f"    No rings satisfy the "
            f"{threshold * 100:.0f}% intersection threshold."
        )
        return None

    # ------------------------------------------------------
    # Valid rings
    # ------------------------------------------------------

    data_valid = pd.DataFrame(valid_rings)

    n_valid = len(data_valid)
    max_R_kpc = data_valid["RAD_Kpc"].max()

    print(
        f"    Valid rings: {n_valid} out of {len(data)} | "
        f"Maximum R: {max_R_kpc:.3f} kpc"
    )

    # ------------------------------------------------------
    # Calculate means
    # ------------------------------------------------------

    mean_inc = np.mean(data_valid["INC"])
    mean_pa = np.mean(data_valid["PA"])
    mean_vsys = np.mean(data_valid["VSYS"])
    mean_vrot = np.mean(data_valid["VROT"])
    mean_disp = np.mean(data_valid["DISP"])
    mean_z = np.mean(data_valid["Z0_pc"])
    mean_vrad = np.mean(data_valid["VRAD"])

    return (
        mean_inc,
        mean_pa,
        mean_vsys,
        mean_vrot
    )


results = []

# Loop through outerdir/{name}
for name in sorted(os.listdir(outerdir)):

    galaxy_dir = os.path.join(outerdir, name)

    if not os.path.isdir(galaxy_dir):
        continue

    rings_file = os.path.join(galaxy_dir, "rings_final2.txt")

    if not os.path.exists(rings_file):
        print(f"No rings_final2.txt found for {name}")
        continue
    try:
        mom0_file_path = f"/data/c3040163/llama/alma/pipeline_m0/{name}/{name}_12m_co21_strict_mom0.fits" if seyfert else f"/Users/administrator/Astro/LLAMA/ALMA/pipeline_m0/{name}/{name}_12m_co21_strict_mom0.fits"
        with fits.open(mom0_file_path) as hdul:
            header = hdul[0].header.copy()
    except:
        try:
            mom0_file_path = f"/data/c3040163/llama/alma/pipeline_m0/{name}/{name}_12m_co32_strict_mom0.fits" if seyfert else f"/Users/administrator/Astro/LLAMA/ALMA/pipeline_m0/{name}/{name}_12m_co32_strict_mom0.fits"
            with fits.open(mom0_file_path) as hdul:
                header = hdul[0].header.copy()
        except:
            print(f'no m0 file found for {name}')
            continue

    print(f'\nprocessing {name}')

    pixel_scale_arcsec = abs(header["CDELT2"]) * 3600
    D_Mpc = llamatab[llamatab['id'] == name]['D [Mpc]'][0]


    result = process_rings_file(rings_file, D_Mpc , pixel_scale_arcsec)


    if result is None:
        print(f"No valid FITOK rows for {name}")
        continue

    mean_inc, mean_pa, mean_vsys, mean_vrot = result

    results.append({
        "name": name,
        "mean_INC_deg": round(mean_inc,2),
        "mean_PA_deg": round(mean_pa,2),
        "mean_vsys": round(mean_vsys,2),
        "mean_vrot": round(mean_vrot,2)
    })

    # print(
    #     f"{name}: INC={mean_inc:.3f} deg, PA={mean_pa:.3f} deg"
    # )


# Save results
results_df = pd.DataFrame(results)

results_df.to_csv(output_csv, index=False)

print(f"\nSaved results to {output_csv}")
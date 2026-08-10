#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd


# Input directory containing {name} subdirectories
outerdir = "/Users/administrator/Astro/LLAMA/ALMA/barolo/phangsmask"

# Output file
output_csv = outerdir+"/phangsmask_fit1.csv"


def process_rings_file(filepath):
    """
    Read rings_final2.txt, filter FITOK == 1,
    and return mean INC and PA.
    """

    # Column names matching the file structure
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
            delim_whitespace=True,
            names=columns
        )

    except Exception as e:
        print(f"Failed reading {filepath}: {e}")
        return None

    # Keep only successful fits
    data = data[data["FITOK"] == 1]

    if len(data) == 0:
        return None

    mean_inc = np.mean(data["INC"])
    mean_pa = np.mean(data["PA"])
    mean_vsys = np.mean(data["VSYS"])
    mean_vrot = np.mean(data["VROT"])

    return mean_inc, mean_pa, mean_vsys, mean_vrot


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

    result = process_rings_file(rings_file)

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

    print(
        f"{name}: INC={mean_inc:.3f} deg, PA={mean_pa:.3f} deg"
    )


# Save results
results_df = pd.DataFrame(results)

results_df.to_csv(output_csv, index=False)

print(f"\nSaved results to {output_csv}")
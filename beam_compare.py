import os
import glob
import csv
from astropy.io import fits
import analysisUtils as au
from astropy.table import Table
from casatools import quanta, msmetadata, measures
import re
from astroquery.ned import Ned
import warnings

def normalize_field_name(s):
    """
    Robust normalization of astronomical field names:
    - Uppercase letters
    - Convert each numeric block to integer (removes leading zeros)
    - Remove separators after splitting
    """
    s = s.upper()

    # Split into sequences of letters and numbers
    parts = re.findall(r'\d+|[^\d\s_\-\.]+', s)
    # parts example: ['MCG', '06', '30', '015']

    for i, p in enumerate(parts):
        if p.isdigit():
            parts[i] = str(int(p))  # convert numeric block to int to strip leading zeros

    # Recombine
    normalized = ''.join(parts)
    return normalized



# === CONFIGURATION ===
# Top-level directory containing subdirectories named {name}
llamatab = Table.read('/data/c3040163/llama/llama_main_properties.fits', format='fits')
base_dir = "/data/c3040163/llama/alma/phangs_imaging_scripts-master/full_run_newkeys_all_arrays/reduction/imaging"
output_csv = os.path.join(os.getcwd(), "beam_summary.csv")

# === TABLE HEADER ===
results = [("Name", "MS_Directory", "SynthesizedBeam", "FITS_File", "BMAJ", "BMIN", "Average_Beam", "Increase_in_Beam")]

# === LOOP THROUGH SUBDIRECTORIES ===
for name in sorted(os.listdir(base_dir)):
    name_path = os.path.join(base_dir, name)
    if not os.path.isdir(name_path):
        continue

    # --- find all .ms directories under this name ---
    ms_dirs = sorted(glob.glob(os.path.join(name_path, "*.ms")))
    for vis in ms_dirs:
        try:
            print(f"Processing {vis} ...")

            # Get target field name from llamatab
            field_entry = llamatab[llamatab['id'] == name]['name'][0]
            field_norm = normalize_field_name(field_entry)
            print(field_norm, "normalized target field name")

            # Open MS metadata
            msmd = msmetadata()
            msmd.open(vis)

            # Get all field names from the MS
            ms_field_names = msmd.fieldnames()

            # Normalize them the same way and find the match
            fname_match = None
            print('printing field names in MS:')
            for f in ms_field_names:
                f_norm = normalize_field_name(f)
                print(f_norm)
                if f_norm == field_norm:
                    fname_match = f  # preserve original capitalization/format
                    break

            if fname_match is None:
                print(f"No direct field match for {field_entry} in {vis}. Trying NED aliases...")

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")  # suppress minor warnings
                        ned_result = Ned.get_table(field_entry, table='names')

                    if ned_result is not None and len(ned_result) > 0:
                        # Extract all Name / ObjectName identifiers from NED table
                        aliases = [str(ned_result['Object Name'][i]) for i in range(len(ned_result))]
                        print(f"Found {len(aliases)} aliases in NED for {field_entry}")

                        # Try each alias as a possible match
                        print('printing aliases:')
                        for alt in aliases:
                            alt_norm = normalize_field_name(alt)
                            print(alt_norm)
                            for f in ms_field_names:
                                f_norm = normalize_field_name(f)
                                if f_norm == alt_norm:
                                    fname_match = f
                                    print(f"Matched via NED alias: {alt} → {fname_match}")
                                    break
                            if fname_match:
                                break
                    else:
                        print(f"No aliases found in NED for {field_entry}")

                except Exception as e:
                    print(f"NED lookup failed for {field_entry}: {e}")

            # --- if still no match, skip this MS ---
            if fname_match is None:
                print(f"⚠️ No matching field (or alias) found for {field_entry} in {vis}. Skipping.")
                msmd.close()
                continue


            print(f"Matched field: {fname_match}")

            # Run beam estimation using the *original* MS field name
            synth_beam = au.estimateSynthesizedBeam(vis, useL80method=True, field=fname_match)

            msmd.close()

        except Exception as e:
            print(f"Error estimating beam for {vis}: {e}")
            synth_beam = None


        # --- find corresponding FITS file (exact stem match) ---
        fits_file = os.path.splitext(vis)[0] + ".fits"
        if not os.path.exists(fits_file):
            print(f"No exact FITS match for {vis}, skipping.")
            continue  # skip to next .ms


        bmaj = None
        if fits_file:
            try:
                with fits.open(fits_file) as hdul:
                    bmaj = float(hdul[0].header.get("BMAJ", None))*3600
                    bmin = float(hdul[0].header.get("BMIN", None))*3600
                    average_beam = (bmaj + bmin) / 2 if bmaj and bmin else None
            except Exception as e:
                print(f"Error reading BMAJ from {fits_file}: {e}")

        # --- add to results ---
        vis_name = os.path.basename(vis)
        fits_name = os.path.basename(fits_file) if fits_file else None
        increase = f'{((average_beam/float(synth_beam))-1):.2%}' if bmaj and synth_beam else None
        results.append((name, vis_name, synth_beam, fits_name, bmaj, bmin, average_beam, increase))

# === WRITE OUTPUT TABLE ===
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(results)

print(f"\n✅ Beam summary table written to: {output_csv}")
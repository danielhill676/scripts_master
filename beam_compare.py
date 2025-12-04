import os
import glob
import csv
from astropy.io import fits
import analysisUtils as au
from astropy.table import Table
from casatools import quanta, msmetadata, measures, image as iatool
import re
from astroquery.ned import Ned
import warnings

co32 = True

def normalize_field_name(s):
    """
    Robust normalization of astronomical field names:
    - Uppercase letters
    - Convert each numeric block to integer (removes leading zeros)
    - Remove separators after splitting
    """
    s = s.upper()
    parts = re.findall(r'\d+|[^\d\s_\-\.]+', s)
    for i, p in enumerate(parts):
        if p.isdigit():
            parts[i] = str(int(p))
    normalized = ''.join(parts)
    return normalized

# === CONFIGURATION ===
llamatab = Table.read('/data/c3040163/llama/llama_main_properties.fits', format='fits')
base_dir = "/data/c3040163/llama/alma/phangs_imaging_scripts-master/full_run_newkeys_all_arrays/reduction/imaging"
if co32:
    base_dir = "/data/c3040163/llama/alma/phangs_imaging_scripts-master/CO32_all_arrays/reduction/imaging"
postprocess_dir = "/data/c3040163/llama/alma/phangs_imaging_scripts-master/full_run_newkeys_all_arrays/reduction/postprocess"
if co32:
    postprocess_dir = "/data/c3040163/llama/alma/phangs_imaging_scripts-master/CO32_all_arrays/reduction/postprocess"
output_csv = os.path.join(os.getcwd(), "beam_summary.csv")
if co32:
    output_csv = os.path.join(os.getcwd(), "beam_summary_co32.csv")

# === TABLE HEADER ===
results = [("Name", "MS_Directory", "SynthesizedBeam", "FITS_File", "BMAJ_arcsec", "Weighting")]

# === LOOP THROUGH SUBDIRECTORIES ===
for name in sorted(os.listdir(base_dir)):
    name_path = os.path.join(base_dir, name)
    if not os.path.isdir(name_path):
        continue

    ms_dirs = sorted(glob.glob(os.path.join(name_path, "*12m_co21.ms")))
    if co32:
        ms_dirs = sorted(glob.glob(os.path.join(name_path, "*12m_co32.ms")))
    for vis in ms_dirs:
        try:
            print(f"\nProcessing {vis} ...")

            # Get target field name from llamatab
            field_entry = llamatab[llamatab['id'] == name]['name'][0]
            field_norm = normalize_field_name(field_entry)

            # Open MS metadata
            msmd = msmetadata()
            msmd.open(vis)
            ms_field_names = msmd.fieldnames()

            # Normalize and match
            fname_match = None
            for f in ms_field_names:
                if normalize_field_name(f) == field_norm:
                    fname_match = f
                    break

            # If no match, try NED aliases
            if fname_match is None:
                print(f"No direct field match for {field_entry} in {vis}. Trying NED aliases...")
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ned_result = Ned.get_table(field_entry, table='names')

                    if ned_result is not None and len(ned_result) > 0:
                        aliases = [str(ned_result['Object Name'][i]) for i in range(len(ned_result))]
                        for alt in aliases:
                            alt_norm = normalize_field_name(alt)
                            for f in ms_field_names:
                                if normalize_field_name(f) == alt_norm:
                                    fname_match = f
                                    print(f"Matched via NED alias: {alt} → {fname_match}")
                                    break
                            if fname_match:
                                break
                except Exception as e:
                    print(f"NED lookup failed for {field_entry}: {e}")

            if fname_match is None:
                print(f"⚠️ No matching field (or alias) found for {field_entry} in {vis}. Skipping.")
                msmd.close()
                continue

            print(f"Matched field: {fname_match}")
            synth_beam = au.estimateSynthesizedBeam(vis, useL80method=True, field=fname_match)
            msmd.close()

        except Exception as e:
            print(f"Error estimating beam for {vis}: {e}")
            synth_beam = None

        # --- FITS file handling in postprocess directory ---
        image_stem = os.path.splitext(os.path.basename(vis))[0]
        casa_image = os.path.join(postprocess_dir, name, image_stem + "_pbcorr_trimmed.image")
        fits_file = os.path.join(postprocess_dir, name, image_stem + "_pbcorr_trimmed.fits")

        bmaj = None
        weighting = None

        # Convert CASA image to FITS if necessary
        if os.path.exists(casa_image) and not os.path.exists(fits_file):
            ia = iatool()
            try:
                print(f"Converting {casa_image} to FITS...")
                ia.open(casa_image)
                ia.tofits(fits_file)
            except Exception as e:
                print(f"Failed to convert {casa_image} to FITS: {e}")
            finally:
                ia.done()
            

        # Read BMAJ and weighting from FITS
        if os.path.exists(fits_file):
            try:
                with fits.open(fits_file) as hdul:
                    bmaj = float(hdul[0].header.get("BMAJ", 0)) * 3600  # arcsec
                    print(f"BMAJ: {bmaj} arcsec")

                    for card in hdul[0].header.cards:
                        if card.keyword == "HISTORY":
                            match = re.search(r'weighting\s*=\s*"([^"]+)"', str(card.value))
                            if match:
                                weighting = match.group(1)
                                print(f"Weighting: {weighting}")
                                break

            except Exception as e:
                print(f"Error reading {fits_file}: {e}")
        else:
            print(f"FITS file not found: {fits_file}")

        # --- append results ---
        results.append((name,
                        os.path.basename(vis),
                        synth_beam,
                        os.path.basename(fits_file) if os.path.exists(fits_file) else None,
                        bmaj,
                        weighting))

# === WRITE OUTPUT TABLE ===
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(results)

print(f"\n✅ Beam summary table written to: {output_csv}")

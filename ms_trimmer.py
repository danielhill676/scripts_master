# Script: select_fields_by_distance.py
from casatools import quanta, msmetadata, measures
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
import os

# CASA tools
qa = quanta()
msmd = msmetadata()
me = measures()

# Read your target table
llamatab = Table.read('/data/c3040163/llama/llama_main_properties.fits', format='fits')

# ---------------- helpers ----------------
def deg_to_sexagesimal(ra_deg, dec_deg, frame="J2000"):
    ra_q = qa.quantity(float(ra_deg), 'deg')
    ra_hms = qa.formxxx(ra_q, format='hms', prec=9)
    dec_q = qa.quantity(float(dec_deg), 'deg')
    dec_dms = qa.formxxx(dec_q, format='dms', prec=8)
    return ra_hms, dec_dms, frame

def ms_frame_from_phase(phase0):
    """
    Try to infer MS frame from a phasecenter measure using me.show().
    Returns an astropy frame string: 'icrs' or 'fk5' (J2000) as best guess.
    """
    try:
        s = me.show(phase0)  # returns a TEXT description
        sl = s.lower()
        if 'icrs' in sl:
            return 'icrs'
        if 'j2000' in sl or 'fk5' in sl:
            return 'fk5'
        if 'fk4' in sl or 'b1950' in sl:
            return 'fk4'
        # fallback: try to detect 'ra/dec' string presence
        return 'icrs'
    except Exception:
        # If anything goes wrong, default to ICRS (safe fallback)
        return 'icrs'

# ---------------- user params ----------------
name = 'NGC3351'
row = llamatab[llamatab['id'] == name]
if len(row) == 0:
    raise RuntimeError(f"No entry '{name}' found in llama table")

D_Mpc = float(row['D [Mpc]'][0])

input_ms = 'trimmed_calibrated_final.ms'
output_ms = 'trimmed_' + input_ms

# ensure these are plain floats
target_ra = float(row['RA (deg)'][0])    # deg (catalog assumed J2000/FK5)
target_dec = float(row['DEC (deg)'][0])  # deg

max_sep_kpc = 1.0
max_sep_arcsec = (max_sep_kpc / D_Mpc) * 206.265  # arcsec

# ---------------- open MS ----------------
msmd.open(input_ms)
nfields = msmd.nfields()
if nfields == 0:
    raise RuntimeError("MS has no fields according to msmetadata().nfields()")

# ---------------- detect MS frame ----------------
phase0 = msmd.phasecenter(0)   # direction measure (dictionary-like)
ms_frame_astropy = ms_frame_from_phase(phase0)
print(f"Detected MS frame (astropy): {ms_frame_astropy}")

# ---------------- build SkyCoord for target ----------------
# assume your catalog RA/Dec are J2000 (FK5) — that's the usual case
# If your catalog is in a different frame change 'fk5' below accordingly.
catalog_frame = 'fk5'   # J2000
target_coord = SkyCoord(ra=target_ra*u.deg, dec=target_dec*u.deg, frame=catalog_frame, equinox='J2000')

# transform the target into the MS frame for consistent comparisons
try:
    target_in_ms = target_coord.transform_to(ms_frame_astropy)
except Exception:
    # fallback: if astropy can't transform for some reason, assume MS frame equals catalog
    target_in_ms = target_coord
print(f"Target (in MS frame): RA={target_in_ms.ra.deg:.8f} deg  Dec={target_in_ms.dec.deg:.8f} deg")

# ---------------- loop fields, compute separations with Astropy ----------------
all_field_ras = []
all_field_decs = []
selected_fields = []
selected_field_ras = []
selected_field_decs = []

name_norm = name.upper().replace("_", "")

for field_id in range(nfields):
    # get field name and normalize
    fname = msmd.namesforfields([field_id])[0]
    fname_norm = fname.upper().replace("_", "")

    if fname_norm != name_norm:
        continue

    # msmd.phasecenter returns rad values under m0/m1 -> convert to degrees
    phase = msmd.phasecenter(field_id)
    f_ra_deg = np.degrees(float(phase['m0']['value']))
    f_dec_deg = np.degrees(float(phase['m1']['value']))

    # store all fields with this name
    all_field_ras.append(f_ra_deg)
    all_field_decs.append(f_dec_deg)

    # create SkyCoord in ms frame
    field_coord = SkyCoord(ra=f_ra_deg*u.deg, dec=f_dec_deg*u.deg, frame=ms_frame_astropy)

    # compute separation robustly using astropy
    sep_arcsec = target_in_ms.separation(field_coord).arcsec

    print(f"Field {field_id} ({fname}): separation = {sep_arcsec:.3f} arcsec")

    if sep_arcsec <= max_sep_arcsec:
        selected_fields.append(field_id)
        selected_field_ras.append(f_ra_deg)
        selected_field_decs.append(f_dec_deg)

print(f"Selected fields with Name={name}: {selected_fields}")

if len(selected_fields) == 0:
    msmd.close()
    raise ValueError("No fields found within the specified distance.")

# ---------------- optional split ----------------
field_str = ','.join(str(fid) for fid in selected_fields)
# try calling split() if available (CASA builtin). If not, print the command to run manually.
try:
    split  # noqa: F821 (refer to builtin split in CASA shell)
    # If running inside CASA shell, split should be a builtin task
    os.system('rm -rf ' + output_ms)  # remove existing output MS if any
    split(vis=input_ms, outputvis=output_ms, field=field_str, datacolumn='data')
    print(f"Created new MS: {output_ms}")
except NameError:
    print("CASA 'split' builtin not available in this Python environment.")
    print("Run the following in the CASA shell to create the trimmed MS:")
    print(f"split(vis='{input_ms}', outputvis='{output_ms}', field='{field_str}', datacolumn='data')")

# --- Plotting ---
fig, ax = plt.subplots(figsize=(6, 6))

matched_ras = []
matched_decs = []
matched_ids = []
matched_names = []

for fid in range(msmd.nfields()):
    fname = msmd.namesforfields([fid])[0].upper().replace("_", "")
    if fname != name:   # only keep fields with the same target name
        continue

    ph = msmd.phasecenter(fid)
    ra_deg = np.degrees(ph['m0']['value'])
    dec_deg = np.degrees(ph['m1']['value'])

    matched_ras.append(ra_deg)
    matched_decs.append(dec_deg)
    matched_ids.append(fid)
    matched_names.append(fname)

# Convert target into MS frame
target_icrs = me.direction('ICRS', f"{target_ra}deg", f"{target_dec}deg")
phase0 = msmd.phasecenter(0)
ms_frame = phase0['refer']  # 'ICRS' or 'J2000'
target_in_ms = me.measure(target_icrs, ms_frame)
target_ra_deg = np.degrees(target_in_ms['m0']['value'])
target_dec_deg = np.degrees(target_in_ms['m1']['value'])

# Plot all matched fields (grey)
ax.scatter(matched_ras, matched_decs, c='grey', alpha=0.5, label=f'Fields: {name}')

# Highlight selected fields (blue)
selected_ras_deg = [matched_ras[matched_ids.index(fid)] for fid in selected_fields]
selected_decs_deg = [matched_decs[matched_ids.index(fid)] for fid in selected_fields]
ax.scatter(selected_ras_deg, selected_decs_deg, c='blue', label='Selected')

# Annotate matched fields
for i, fname in enumerate(matched_names):
    ax.text(matched_ras[i], matched_decs[i], str(matched_ids[i]), fontsize=8)

# Plot target coordinate
ax.scatter(target_ra_deg, target_dec_deg, c='red', marker='*', s=200, label='Target')

# Draw selection circle
radius_deg = max_sep_arcsec / 3600.0
circle = plt.Circle((target_ra_deg, target_dec_deg), radius_deg, 
                    color='red', fill=False, linestyle='--', 
                    label=f'{max_sep_kpc} kpc')
ax.add_patch(circle)

# Labels and styling
ax.set_xlabel("RA (deg)")
ax.set_ylabel("Dec (deg)")
ax.set_title(f"Fields matching {name}")
ax.invert_xaxis()  # RA increases to the left
ax.set_aspect('equal', adjustable='datalim')  # Keep circle round
ax.legend()
plt.show()

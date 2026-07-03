import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

# -----------------------------
# Load your CSV
# Expected columns:
# name, ra, dec, v1, v2 (extra cols ok)
# -----------------------------
df = pd.read_csv("/Users/administrator/Astro/LLAMA/ALMA/comp_samples/phangs_radec.csv", header=None)

df.columns = ["Name", "RA_str", "DEC_str", "col4", "col5"]  # adjust if needed


# -----------------------------
# Convert RA/DEC → degrees
# -----------------------------
def to_deg(ra_str, dec_str):
    try:
        c = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
        return c.ra.deg, c.dec.deg
    except Exception as e:
        print(f"Failed: {ra_str}, {dec_str} -> {e}")
        return None, None


ra_deg = []
dec_deg = []

for ra, dec in zip(df["RA_str"], df["DEC_str"]):
    r, d = to_deg(ra, dec)
    ra_deg.append(r)
    dec_deg.append(d)


# -----------------------------
# Add new columns
# -----------------------------
df["RA_deg"] = ra_deg
df["DEC_deg"] = dec_deg


# -----------------------------
# Save output
# -----------------------------
df.to_csv("/Users/administrator/Astro/LLAMA/ALMA/comp_samples/phangs_radec_deg.csv", index=False)

print("Done → /Users/administrator/Astro/LLAMA/ALMA/comp_samples/phangs_radec_deg.csv")
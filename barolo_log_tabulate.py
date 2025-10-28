import re
import pandas as pd

log_path = "/Users/administrator/Astro/LLAMA/ALMA/barolo/barolo3.log"
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    log_text = f.read()


# Define the starting string after which to read the log
start_string = "(base) administrator@MSP-MAC-522839 barolo % ./barolo_execute.sh"

# Only keep log content after the last occurrence of start_string
if start_string in log_text:
    _, _, log_text = log_text.rpartition(start_string)
else:
    print("Warning: start string not found in log. Parsing entire file.")

# Split the log into chunks by galaxies
chunks = re.split(r"FITS file to be analysed.*=", log_text)[1:]

records = []

for chunk in chunks:
    # Galaxy name (from the FITS filename)
    match_name = re.search(r"\s+(\S+\.fits)", chunk)
    galaxy = match_name.group(1).replace(".fits", "") if match_name else None

    # Number of rings and ring width
    match_rings = re.search(r"Fitting\s+#(\d+).*rings of width\s+([\d\.]+)", chunk)
    num_rings = int(match_rings.group(1)) if match_rings else None
    ring_width = float(match_rings.group(2)) if match_rings else None

    # Failed rings
    failed_rings = len(re.findall(r"Not enough pixels to fit in ring", chunk))

    # Convergence failures
    convergence_failures = len(re.findall(r"Can not achieve convergence in ring", chunk))

    # Source count (mask-building step)
    match_source = re.search(r"Final object count\s*=\s*(\d+)", chunk)
    source_count = int(match_source.group(1)) if match_source else None


    records.append({
        "Galaxy": galaxy,
        "Num_Rings": num_rings,
        "Ring_Width_arcsec": ring_width,
        "Rings_Failed": failed_rings,
        "Convergence_Failures": convergence_failures,
        "Source_Count": source_count
    })

# Convert to DataFrame
df = pd.DataFrame(records)

# Save to CSV
csv_path = "/Users/administrator/Astro/LLAMA/ALMA/barolo/fit_summary.csv"
df.to_csv(csv_path, index=False)

print(f"Saved summary table with source count to: {csv_path}")
print(df.head(10))  # show first 10 rows as a preview

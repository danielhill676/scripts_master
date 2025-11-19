import re
import pandas as pd

seyfert = True

basedir = "/Users/administrator/Astro/LLAMA/ALMA/barolo/"
if seyfert:
    basedir = '/data/c3040163/llama/alma/barolo/'
log_path = f"{basedir}barolo1.log"
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    log_text = f.read()


# Define the starting string after which to read the log
start_string = "(base) administrator@MSP-MAC-522839 barolo % ./barolo_execute.sh"
if seyfert:
    start_string = ""

# Only trim the log if a non-empty start_string is given
if start_string:
    if start_string in log_text:
        _, _, log_text = log_text.rpartition(start_string)
    else:
        print("Warning: start string not found in log. Parsing entire file.")
# else: skip trimming entirely


# Split the log into chunks by galaxies
chunks = re.split(r"FITS file to be analysed.*=", log_text)[1:]

records = []

for chunk in chunks:
    # Galaxy name (from the FITS filename)
    match_name = re.search(r"\s+(\S+\.fits)", chunk)
    if seyfert:
        match_name = re.search(r"\s+(\S+\_12m_co21.fits)", chunk)
    galaxy = match_name.group(1).replace("_12m_co21.fits", "") if match_name else None

    # Number of rings and ring width
    match_rings = re.search(r"Fitting\s+#(\d+).*rings of width\s+([\d\.]+)", chunk)
    num_rings = int(match_rings.group(1)) if match_rings else None
    ring_width = float(match_rings.group(2)) if match_rings else None

    # Failed rings
    failed_rings = len(re.findall(r"Not enough pixels to fit in ring", chunk))

    # Convergence failures
    convergence_failures = len(re.findall(r"Can not achieve convergence in ring", chunk))

    # successful rings
    successful_rings = round((num_rings - failed_rings - convergence_failures) / num_rings,1) if num_rings is not None else None

    # Segfault detection
    num_segfault = len(re.findall(r"Segmentation fault", chunk))
    if num_segfault > 0:
        segfault = True
        successful_rings = 0
    else:
        segfault = False

    # Plot error detection
    num_plot_errors = len(re.findall(r"Something went wrong! Check plotting scripts in the output folder", chunk))
    if num_plot_errors > 0:
        plot_error = True
    else:
        plot_error = False

    # no source detection

    num_nosource = len(re.findall(r"No sources detected in the datacube", chunk))
    if num_nosource > 0:
        no_source = True
        successful_rings = 0
    else:
        no_source = False

    # Source count (mask-building step)
    match_source = re.search(r"Final object count\s*=\s*(\d+)", chunk)
    source_count = int(match_source.group(1)) if match_source else None
    


    records.append({
        "Galaxy": galaxy,
        "Num_Rings": num_rings,
        "Ring_Width_arcsec": ring_width,
        "segfault": segfault,
        "no_source": no_source,
        "Rings_Failed": failed_rings,
        "Convergence_Failures": convergence_failures,
        "Successful_Rings": successful_rings,
        "Source_Count": source_count,
        "Plot_Error": plot_error
    })

# Convert to DataFrame
df = pd.DataFrame(records)

# Save to CSV
csv_path = "/Users/administrator/Astro/LLAMA/ALMA/barolo/fit_summary.csv"
if seyfert:
    csv_path = '/data/c3040163/llama/alma/barolo/fit_summary_seyfert.csv'
df.to_csv(csv_path, index=False)

print(f"Saved summary table with source count to: {csv_path}")
print(df.head(10))  # show first 10 rows as a preview

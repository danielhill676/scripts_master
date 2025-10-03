from astropy.io import fits
import pandas as pd
from tabulate import tabulate

hdul = fits.open('/Users/administrator/Astro/LLAMA/ALMA/uids_table.fits')
data = hdul[1].data
print(tabulate(data))
df = pd.DataFrame(data.byteswap().newbyteorder())
df.to_csv('/Users/administrator/Astro/LLAMA/ALMA/array_uids.csv', index=False)
hdul.close()
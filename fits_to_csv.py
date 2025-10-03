import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

# uncomment for 3d fits file

# # Load the FITS file
# hdulist = fits.open('/Users/administrator/Documents/bayesian/astro/NGC4254_velocitymap.fits')
# scidata = hdulist[0].data
# print("Shape of scidata:", scidata.shape)

# # Get the dimensions of the data
# f_dim, x_dim, y_dim = scidata.shape[1:4]
# print(f"Dimensions: x = {x_dim}, y = {y_dim}, z = {f_dim}")

# # Create mesh grids for the coordinates (x, y, f)
# x, y, f = np.indices((x_dim, y_dim, f_dim))

# # Flatten the coordinates and values to create a 2D DataFrame
# x = x.flatten()
# y = y.flatten()
# f = f.flatten()
# values = scidata.flatten()

# # Create a DataFrame with the coordinates and the values
# stacked = pd.DataFrame({
#     'x': x,
#     'y': y,
#     'f': f,
#     'value': values
# })

# # Save the DataFrame to a CSV
# stacked.to_csv('/Users/administrator/Documents/bayesian/7213.csv', index=False)

# # Print the shape of the resulting DataFrame
# print(stacked.shape)


import numpy
from astropy.io import fits

# ask for an input file 
filename = '/Users/administrator/Documents/bayesian/astro/NGC4254_velocitymap.fits'

# ask for an output file name
output = '/Users/administrator/Documents/bayesian/astro/NGC4254_velocitymap.csv'

# Open the given fits file
hdulist = fits.open(filename)
scidata = hdulist[0].data
scidata = scidata.squeeze()
print(scidata.shape)
# save your new file as a csv
numpy.savetxt(output, scidata, fmt='%.6f', delimiter=',')
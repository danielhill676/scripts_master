import pyproffit
import matplotlib.pyplot as plt
import math
import os
import time
import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from spectral_cube import SpectralCube
from astropy.table import Table
from regions import CircleSkyRegion # type: ignore
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm
from astropy.visualization.wcsaxes import add_scalebar
from astropy.visualization.wcsaxes import add_beam
from astropy.stats import sigma_clipped_stats
from astropy.io import fits

os.chdir('/Users/administrator/Astro/LLAMA/ALMA/AGN_images/')
dat=pyproffit.Data(imglink='NGC5728_moment_0.fits')

# data = pyproffit.data(
#     imglink='NGC5728_moment_0.fits',  # The image data to fit
#     data_units='Jy/beam km/s',  # Set the data units
#     x_unit='arcsec',  # Set the x-unit to arcseconds
#     y_unit='Jy/beam km/s',  # Set the y-unit to Jy/arcsec^2
# )


prof=pyproffit.Profile(dat,center_choice='centroid',maxrad=0.1,binsize=0.7)
prof.SBprofile()
# prof.Plot()
# print(prof.profile)
# help(prof)
powlaw=pyproffit.Model(pyproffit.PowerLaw)
beta = pyproffit.Model(pyproffit.BetaModel)

# fitobj=pyproffit.Fitter(model=powlaw, profile=prof, alpha=0.7, norm=-2.,pivot=2,bkg=-4)

fitobj=pyproffit.Fitter(model=beta, profile=prof, beta=0.7, norm=-2.,rc=2,bkg=-4)

fitobj.Migrad(fixed=[0,0,0,0])

# prof.Plot(model=powlaw)

prof.Plot(model=beta)

plt.show()

print('attempting fitobj.minuit.minos()')

if fitobj.minuit.fmin.is_valid:
	fitobj.minuit.minos()
else:
	print("Fit did not converge properly. Skipping minos step.")

# print('\nattempting fitobj.minuit.draw_mncontour')
# try:
# 	fitobj.minuit.draw_mncontour('alpha', 'pivot', cl=(0.68, 0.9, 0.99))
# except ValueError as e:
# 	print(f"Error drawing contour: {e}")
# plt.show()

print('\nattempting fitobj.minuit.draw_mncontour')
try:
	if fitobj.minuit.fmin.is_valid and fitobj.minuit.np_matrix().size > 0:
		fitobj.minuit.draw_mncontour('beta', 'rc', cl=(0.68, 0.9, 0.99))
	else:
		print("Fit results are not valid or parameter matrix is empty. Skipping contour drawing.")
except ValueError as e:
	print(f"Error drawing contour: {e}")
plt.show()

# powlaw - 381 beta - 583 bad! 

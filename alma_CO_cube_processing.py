import sys
import os
import warnings

import numpy as np

from scipy.integrate import trapz

import matplotlib.pyplot as plt

from spectral_cube import SpectralCube

from astropy.table import Table, join
from astropy.utils.exceptions import AstropyWarning
from astropy.io import fits
from astropy import units as u
from astropy import wcs
from astropy.coordinates import SkyCoord
# from astropy.convolution import Gaussian1DKernel
from astropy.nddata.utils import block_reduce
from astropy.stats import sigma_clipped_stats

#from voronoi.voronoi_2d_binning import voronoi_2d_binning

from lmfit.models import GaussianModel


""" Process and make measurements on ALMA CO 2-1 cubes of LLAMA galaxies
	
"""

def  CubeMeasurements1(cube_vel, snrcut=3.0):
	""" First iteration of spectral cube measurements.
		Input is a SpectralCube in velocity units. 
		Single component gaussian fit of block added spectrum.
		Block adding is done to minimally Nyquist sample the beam minor axis.
		Fit is accepted if fitted line is > SNR provided
		Returns line flux map, velocity map, FWHM map with errors as a tuple.
	"""

	# Collapse the cube to a line map. 
	vindex, = np.where((velocity > -1000*u.km/u.s) & (velocity < 1000*u.km/u.s))
	linemap_full = trapz(np.flipud(cube_vel.unmasked_data[vindex,:,:].value),\
						 np.flipud(cube_vel.spectral_axis.value[vindex]),axis=0)
	# Rebin the map to Nyquist sample the beam minor width. 
	oversamp_factor = np.int(np.floor(cube.header['BMIN']*3600/pixscale/2))
	linemap_new = block_reduce(linemap_full,oversamp_factor,func=np.sum) # binned map
	# Old code. Kept here for testing
	#oversamp_factor = 1
	#linemap_new = linemap_full.copy()

	print('CubeMeas1')
	print('Initial field size: {0:4d}, {1:4d} pixels'.format(linemap_full.shape[1],linemap_full.shape[0]))
	print('Final field size: {0:4d}, {1:4d} pixels'.format(linemap_new.shape[1],linemap_new.shape[0]))
	print('Resampling factor: x{0:1d}'.format(oversamp_factor))

	# Initialise arrays to store velocity and line width (with errors)
	linemap  = np.full((2,linemap_new.shape[1],linemap_new.shape[0]),np.nan)
	velmap   = np.full((2,linemap_new.shape[1],linemap_new.shape[0]),np.nan)
	widthmap = np.full((2,linemap_new.shape[1],linemap_new.shape[0]),np.nan)

	# 	Initialise the model:
	# 	   a) Single 1D gaussian profile

	gauss_component1 = GaussianModel(prefix='gcomp1_',nan_policy='omit')
	linemodel = gauss_component1
	lineparams = linemodel.make_params()

	# Apply reasonable bounds to the parameters
	lineparams['gcomp1_amplitude'].min = 0.0  # Amplitude is always positive
	lineparams['gcomp1_center'].min = -1000.0 # Line center is within 1000 km/s ...
	lineparams['gcomp1_center'].max =  1000.0 # ... of systemic
	lineparams['gcomp1_sigma'].min = 2.0*np.abs(cube_vel.header['CDELT3'])/2.35  #  Line width > Nyquist sampled velocity resolution
	lineparams['gcomp1_sigma'].max = 2000.0/2.35 # Line FWHM < 1000 km/s

	print('Currently processing 1D spectra in row: ')
	# Iterate over rebinned pixel ranges
	for ix in range(int(linemap_new.shape[1])):
	
		print(ix)
		for iy in range(int(linemap_new.shape[0])):
		
			# If the lineflux is non-finite or is not at least n sigma, don't perform any fits
			if (np.isfinite(linemap_new[iy,ix]) == False):
				continue
		
# 		If the pixel > radius of primary beam from the pointing center of the map, then don't perform any fits
# 		pix_coord = wcs.utils.pixel_to_skycoord(4*ix+2,4*iy+2,astrom,origin=0)
# 		Note, this approach is heavy-handed and can be slow, but is more accurate.
# 		 If upgrading the code for agility, replace this.
# 		if (pix_coord.separation(center_coord).arcsec > primary_beam_width/2):
# 			continue

#			spec1d = cube_vel[:,iy,ix].value
			spec1d = cube_vel[:,oversamp_factor*iy:oversamp_factor*(iy+1),\
								oversamp_factor*ix:oversamp_factor*(ix+1)].sum((1,2)).value
			if np.count_nonzero(np.logical_not(np.isfinite(spec1d))) > 0:
				continue  # Only process spaxels with valid data (before masking)
			# Get noise statistics from sigma-clipping.
			noise = sigma_clipped_stats(spec1d)[2] 
		
			# Initialise with some reasonable starting values for the parameters
			lineparams['gcomp1_amplitude'].value = 5.0*noise*50.0  # 5sigma
			lineparams['gcomp1_center'].value = 0.0 # Systemic velocity
			lineparams['gcomp1_sigma'].value = 50.0 # 50 km/s linewidth (120 km/s FWHM)
			
			# Enclose in try block, since sometimes LMFIT fails. Failed fits are disregarded.
			try:
				fit = gauss_component1.fit(spec1d,params=lineparams,x=velocity.value)
	
				if fit.params['gcomp1_height'].value > 3.0*noise:
					linemap[0,iy,ix]  = fit.best_values['gcomp1_amplitude'] 
					velmap[0,iy,ix]   = fit.best_values['gcomp1_center']
					widthmap[0,iy,ix] = fit.best_values['gcomp1_sigma']*2.35
					if fit.covar is None:
						linemap[1,iy,ix]  = np.nan
						velmap[1,iy,ix]   = np.nan
						widthmap[1,iy,ix] = np.nan
					else:	
						linemap[1,iy,ix]  = np.sqrt(fit.covar[0,0])
						velmap[1,iy,ix]   = np.sqrt(fit.covar[1,1])
						widthmap[1,iy,ix] = np.sqrt(fit.covar[2,2])*2.35
			except:
				print('Fitting error for spectrum at spaxel: {0:4d}, {1:4d}'.format(ix,iy))
				continue
	
	return (linemap,velmap,widthmap,oversamp_factor)	


warnings.simplefilter('ignore', AstropyWarning)
warnings.simplefilter('ignore', UserWarning)

Iteration = 1

llamatable = Table.read('../llama_main_properties.fits',format='fits')

almadataroot = '/obs/r2/rosario/data/LLAMA/ALMA/'
llalmacat = Table.read(almadataroot+'LL-ALMA_database_catalog.ascii',format='ascii.commented_header')
sys.exit()

for tableindex in range(len(llamatable)):

	# Properties of the galaxy
	objname  = llamatable['id'][tableindex]        # ID from LLAMA table 
	redshift = llamatable['redshift'][tableindex]  # redshift from LLAMA table 
	d_l      = llamatable['D [Mpc]'][tableindex]   # luminosity distance

#	if objname != 'MCG514':
#		continue

	catindex,   = np.where(llalmacat['id'] == objname)
	if len(catindex) == 0: continue
	if llalmacat['ALMA cube filename'].mask[catindex[0]]: continue

	# fit_lines_voronoi   = True
	# fit_lines_simplebin = True

	# Properties of the observation
	cubefile = almadataroot+llalmacat['ALMA cube filename'][catindex[0]]
	cotrans = llalmacat['ALMA CO line'][catindex[0]]
	if cotrans == '2-1':
		rest_freq = 230.53800*u.GHz
	elif cotrans == '3-2':
		rest_freq = 345.79599*u.GHz
	else:
		print('Cannot process CO '+cotrans+' yet.')
		continue
	# Skip processing this way if the file is too large (> 4 GB)
#	if llalmacat['ALMA cube filesize (Gb)'][catindex[0]] > 4.0: 
#		print('Skipping processing of cube for '+objname+' because the file is too large.')
#		continue 

	if Iteration == 1:
		outfile = objname+'_ALMA_CO'+cotrans+'_outputs_iter1.fits'
	
	# Since these fits take time, do not overwrite an existing fit.
	# If a refit is needed, manually remove the fitting output file from this directory.
	if os.path.isfile(outfile): continue
	print('Fitting '+objname)

	line_freq = rest_freq/(1.0+redshift) # Line central frequency in GHz
	primary_beam_width = 1.13*((2.9979e8/(line_freq.value*1e9))/12.0)*(180.0*3600.0/np.pi)

	try:
		# Read in the cube
		cube = SpectralCube.read(cubefile)
		cube.allow_huge_operations = True
		# Has shape of (spectrum, Y, X)

		astrom = wcs.WCS(cube.header).celestial  #  the astrometry of the cube
		center_coord = SkyCoord(astrom.wcs.crval[0]*u.deg, astrom.wcs.crval[1]*u.deg)
		pixscale = np.sqrt(astrom.pixel_scale_matrix[0,0]**2.0 + astrom.pixel_scale_matrix[1,0]**2.0)*3600.0
		beamarea = 1.133*(cube.header['BMAJ']*cube.header['BMIN'])*(3600**2)
		jypixconv = beamarea/(pixscale**2)

#		cube *= (beamarea/(pixscale**2)/u.pix)
	except:
		print('Skipping processing of cube for '+objname+' because of header/file issues.')
		continue


	# The cube in velocity space units.
	cube_vel = cube.with_spectral_unit(u.km/u.s,velocity_convention='radio',rest_value=line_freq)
	velocity = cube_vel.spectral_axis.to(u.km/u.s)

	if Iteration == 1:
		fitoutputs = CubeMeasurements1(cube_vel)
		oversamp_factor = fitoutputs[3]
	

	# Generate a WCS instance for the block_reduced map
	outwcs = astrom.deepcopy()
	outwcs.wcs.cdelt = astrom.wcs.cdelt*oversamp_factor
	outwcs.wcs.crpix = astrom.wcs.crpix/oversamp_factor

	# Write a multi-HDU fits file storing continuum map, line map, velocity map, widthmap

	wcs_header = outwcs.to_header()

	primhdu = fits.PrimaryHDU()
	primhdu.header.append(('TARGETNAME',objname))
	primhdu.header.append(('REDSHIFT',redshift,'Redshift of target used for systemic velocity'))
	primhdu.header.append(('CUBEFILE',os.path.basename(cubefile),'Filename of ALMA cube'))

	# line map
	# Multiply by the ratio of the area of the beam (from the header) and the area of the pixel in the resampled map.
	# This to convert from Jy/beam (native) to Jy/pixel (useful for further flux measurements).
	linehdu = fits.ImageHDU(data=fitoutputs[0]*jypixconv/(oversamp_factor**2))
	linehdu.header.append(('LINE','CO ('+cotrans+')','Emission line'))
	linehdu.header.append(('QUANTITY','Line','Integrated line flux'))
	linehdu.header.extend(wcs_header.cards)

	# peak velocity map
	velhdu = fits.ImageHDU(data=fitoutputs[1])
	velhdu.header.append(('LINE','CO ('+cotrans+')','Emission line'))
	velhdu.header.append(('QUANTITY','Velocity','Peak velocity of fitted gaussian (km/s)'))
	velhdu.header.extend(wcs_header.cards)

	# FWHM map
	widthhdu = fits.ImageHDU(data=fitoutputs[2])
	widthhdu.header.append(('LINE','CO ('+cotrans+')','Emission line'))
	widthhdu.header.append(('QUANTITY','FWHM','FWHM of fitted gaussian (km/s)'))
	widthhdu.header.extend(wcs_header.cards)

	outhdu = fits.HDUList(hdus=[primhdu,linehdu,velhdu,widthhdu])
	outhdu.writeto(outfile,overwrite=True)

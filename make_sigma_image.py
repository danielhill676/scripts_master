import numpy as np
from astropy.io import fits
import astropy.units as u
from astropy.table import Table
import os
from multiprocessing import Pool


def compute_sigma_chunk(start_idx, end_idx, memmap_data, mask):
    """
    Compute the standard deviation of a chunk of data for a given spectral range.
    """
    chunk_data = memmap_data[:, start_idx:end_idx, :, :]
    chunk_masked_data = np.ma.masked_array(chunk_data, mask=mask)
    sigma_chunk = np.ma.std(chunk_masked_data, axis=1) * 500
    return sigma_chunk


def model_fit(folder,restf,dv,outfolder):

    for name in os.listdir(folder):
            name_short = name.removesuffix('.fits')

            if name == '.DS_Store':    
                continue
            if name != 'NGC7172.fits':
                continue                 #Commment or uncomment this block for testing

            print(f"running {name}")       
            with fits.open(f'{folder}/{name}') as hdul:

                for i in range(len(llamatab)):
                        if llamatab['id'][i] == f'{name_short}':
                            RA = llamatab['RA (deg)'][i]
                            DEC = llamatab['DEC (deg)'][i]
                            D = llamatab['D [Mpc]'][i]
                            z = llamatab['redshift'][i]

                data = hdul[0].data
                header = hdul[0].header

                n_spectral = data.shape[1]
                min_freq = header['CRVAL3'] * u.Hz
                delta_freq = header['CDELT3'] * u.Hz
                pixel_x = header['CDELT2']
                pixel_y = header['CDELT1']

                observed_frequencies = (min_freq + np.arange(n_spectral) * delta_freq)
                mask_max = restf/(1+z) * (1 + dv / c)
                mask_min = restf/(1+z) * (1 - dv / c)

                line_mask = (observed_frequencies >= mask_min) & (observed_frequencies <= mask_max)
                print('linemask shape', line_mask.shape)

                expanded_line_mask = line_mask[np.newaxis, :, np.newaxis, np.newaxis]
                expanded_line_mask = np.broadcast_to(expanded_line_mask, data.shape)
                print('expanded linemask shape', expanded_line_mask.shape)

                masked_data = np.ma.masked_array(data, mask=expanded_line_mask)
                print('masked data', masked_data.shape)

                print('creating sigma image')

                data = np.array(data, dtype=np.float32)
                npy_file = f'{outfolder}/data_files/{name_short}.npy'
                if os.path.exists(npy_file):
                    os.remove(npy_file)
                np.save(npy_file, data)
                print(f"Data saved as {npy_file}")

                memmap_data = np.memmap(npy_file, dtype=data.dtype, mode='r', shape=data.shape)
                sigma_image = np.zeros(masked_data.shape[1:], dtype=np.float32)

                num_workers = 8
                chunk_size = data.shape[1] // num_workers  # Split the data into chunks
                tasks = []
                for i in range(num_workers):
                    start_idx = i * chunk_size
                    end_idx = (i + 1) * chunk_size if i < num_workers - 1 else data.shape[1]
                    tasks.append((start_idx, end_idx, memmap_data, expanded_line_mask))

                with Pool(processes=num_workers) as pool:
                    sigma_chunks = pool.starmap(compute_sigma_chunk, tasks)
                
                sigma_image = np.concatenate(sigma_chunks, axis=1)
                sigma_image = sigma_image.filled(np.nan).squeeze()
                print(f"Sigma image shape: {sigma_image.shape}")

                # sigma_image /= (masked_data.shape[1] // chunk_size)
                # print('Sigma image shape:', sigma_image.shape)
                

                # sigma_image = sigma_image.filled(np.nan)
                # sigma_image = sigma_image.squeeze()
                # print('sigma image shape',sigma_image.shape)

                bad_pixel_mask = np.any(~np.isfinite(data), axis=1) | ~np.isfinite(sigma_image)

                # Convert boolean mask to integer type
                bad_pixel_mask = bad_pixel_mask.astype(np.int16).squeeze()
                print('badpix shape',bad_pixel_mask.shape)
                
                # Save the sigma image
                sigma_hdu = fits.PrimaryHDU(sigma_image)
                sigma_path = f'{outfolder}/{name_short}.sigma.fits'
                sigma_hdu.writeto(sigma_path, overwrite=True)
                print(f"Sigma image saved to {sigma_path}")

                if os.path.exists(npy_file):
                    os.remove(npy_file)
                    print(f"Data file {npy_file} removed")

                # Save the bad pixel mask
                bad_pixel_hdu = fits.PrimaryHDU(bad_pixel_mask)
                bad_pixel_path = f'{outfolder}/{name_short}.bad_pixels.fits'
                bad_pixel_hdu.writeto(bad_pixel_path, overwrite=True)
                print(f"Bad pixel mask saved to {bad_pixel_path}")

                if np.isnan(bad_pixel_mask).any():
                    print("Bad pixel mask contains NaNs")
                else:
                    print("Bad pixel mask does not contain NaNs")


            with open(f'/Users/administrator/Astro/LLAMA/ALMA/{outfolder}/{name_short}_galfit_input','w') as f:
                f.write(f"""
            ================================================================================
            # IMAGE and GALFIT CONTROL PARAMETERS
            A) {name_short}_moment_0_galfit.fits            # Input data image (FITS file)
            B) {name_short}_imblock.fits       # Output data image block
            C) {name_short}.sigma.fits                # Sigma image name (made from data if blank or "none")
            D) none   #        # Input PSF image and (optional) diffusion kernel
            E) 1                   # PSF fine sampling factor relative to data
            F) {name_short}.bad_pixels.fits                # Bad pixel mask (FITS image or ASCII coord list)
            G) none                # File with parameter constraints (ASCII file)
            H)  0 {int(sigma_image.shape[1])} 0 {int(sigma_image.shape[0])}   # Image region to fit (xmin xmax ymin ymax)
            # I)       # Size of the convolution box (x y)
            J) 25             # Magnitude photometric zeropoint
            K) {pixel_x}  {pixel_y}        # Plate scale (dx dy)   [arcsec per pixel]
            O) regular             # Display type (regular, curses, both)
            P) 0                   # Options: 0=normal run; 1,2=make model/imgblock & quit

            # Exponential function

            0) expdisk            # Object type
            1) {int(0.5*sigma_image.shape[1])}  {int(0.5*sigma_image.shape[0])}  0 0     # position x, y        [pixel]
            3) 33       1       # total magnitude
            4) 80       1       #     Rs               [Pixels]
            9) 1        1       # axis ratio (b/a)
            10) 90         1       # position angle (PA)  [Degrees: Up=0, Left=90]
            Z) 0                 #  Skip this model in output image?  (yes=1, no=0)')
            """)

            if os.path.exists(f'/Users/administrator/Astro/LLAMA/ALMA/{outfolder}/galfit_exceute.sh'):
                os.remove(f'/Users/administrator/Astro/LLAMA/ALMA/{outfolder}/galfit_exceute.sh')


            with open(f'/Users/administrator/Astro/LLAMA/ALMA/{outfolder}/galfit_exceute.sh','w'):
                pass
            with open(f'/Users/administrator/Astro/LLAMA/ALMA/{outfolder}/galfit_exceute.sh','a') as f:
                f.write(f"""
                \ngalfit {name_short}_galfit_input
                """)

#NOTE: test_hdu = fits.PrimaryHDU(np.squeeze(np.full((50,100),0.1))) # this has x width 100 and y height 50 and value 0.1


CO2_1 = 230.538 * 1e9 * u.Hz
CO3_2 = 345.796 * 1e9 * u.Hz
dv = 250 * u.km / u.s
c = 299792.458 * u.km / u.s
llamatab = Table.read('/Users/administrator/Astro/LLAMA/llama_main_properties.fits',format='fits')

folder_AGN = '/Users/administrator/Astro/LLAMA/ALMA/AGN/'
outfolder_AGN = '/Users/administrator/Astro/LLAMA/ALMA/AGN_images'
folder_3_2 = "/Users/administrator/Astro/LLAMA/ALMA/CO32/AGN"

folder_inactive = '/Users/administrator/Astro/LLAMA/ALMA/inactive/'
outfolder_inactive = '/Users/administrator/Astro/LLAMA/ALMA/inactive_images'

model_fit(folder_AGN,CO2_1,dv,outfolder_AGN)
model_fit(folder_inactive,CO2_1,dv,outfolder_inactive)
model_fit(folder_3_2,CO3_2,dv,outfolder_AGN)
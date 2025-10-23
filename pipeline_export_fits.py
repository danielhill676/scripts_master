#!/usr/bin/env python
import os
from casatasks import exportfits as casa_exportfits

# --- Patch casaStuff for your function ---
casaStuff = type('casaStuff', (), {'exportfits': casa_exportfits})

# --- Include your function definition here (unchanged except logger/casaStuff will now work) ---
def export_imaging_to_fits(
        image_root=None,
        imaging_method='tclean',
        bitpix=-32,
        just_image=False):
    """
    Export the products associated with a CASA imaging run to FITS.
    """

    exts = ['alpha', 'alpha.error',
            'beta', 'beta.error',
            'image.tt0', 'image.tt1', 'image.tt2',
            'model.tt0', 'model.tt1', 'model.tt2',
            'residual.tt0', 'residual.tt1', 'residual.tt2',
            'mask.tt0', 'mask.tt1', 'mask.tt2',
            'pb.tt0', 'pb.tt1', 'pb.tt2',
            'psf.tt0', 'psf.tt1', 'psf.tt2',
            'weight.tt0', 'weight.tt1', 'weight.tt2',
            'image', 'model', 'residual', 'mask', 'pb', 'psf', 'weight']

    ext_map = {}

    if imaging_method == 'tclean':
        for ext in exts:
            if ext == 'image':
                ext_map['.%s' % ext] = '.fits'
            elif 'image' in ext and 'tt0' in ext:
                ext_map['.%s' % ext] = '.fits'
            elif 'image' in ext:
                ext_map['.%s' % ext] = '%s.fits' % ext.replace('image', '').replace('.', '_')
            elif 'tt0' in ext:
                ext_map['.%s' % ext] = '_%s.fits' % ext.replace('.tt0', '').replace('.', '_')
            else:
                ext_map['.%s' % ext] = '_%s.fits' % ext.replace('.', '_')
    elif imaging_method == 'sdintimaging':
        cube_exts = ['sd.cube', 'int.cube', 'joint.cube', 'joint.multiterm']
        for cube_ext in cube_exts:
            for ext in exts:
                if ext == 'image':
                    ext_map['.%s.%s' % (cube_ext, ext)] = '_%s.fits' % cube_ext.replace('.', '_')
                elif 'image' in ext and 'tt0' in ext:
                    ext_map['.%s.%s' % (cube_ext, ext)] = '_%s.fits' % cube_ext.replace('.', '_')
                elif 'image' in ext:
                    ext_map['.%s.%s' % (cube_ext, ext)] = '_%s_%s.fits' % (cube_ext.replace('.', '_'),
                                                                           ext.replace('image', '').replace('.', '_'))
                elif 'tt0' in ext:
                    ext_map['.%s.%s' % (cube_ext, ext)] = '_%s_%s.fits' % (cube_ext.replace('.', '_'),
                                                                           ext.replace('.tt0', '').replace('.', '_'))
                else:
                    ext_map['.%s.%s' % (cube_ext, ext)] = '_%s_%s.fits' % (cube_ext.replace('.', '_'),
                                                                           ext.replace('.', '_'))

    for this_ext in ext_map.keys():
        if just_image and ((this_ext != '.tt0') and this_ext != '.image'):
            continue

        this_casa_ext = this_ext
        this_fits_ext = ext_map[this_ext]

        casa_image = image_root + this_casa_ext
        if not os.path.isdir(casa_image):
            continue
        fits_image = image_root + this_fits_ext

        print('exportfits from ' + casa_image + ' to ' + fits_image)
        casaStuff.exportfits(imagename=casa_image,
                             fitsimage=fits_image,
                             velocity=True, overwrite=True, dropstokes=True,
                             dropdeg=True, bitpix=bitpix)

    return ()


# --- Main scanning script ---
def export_all_images(base_dir, skiplist=None):
    endings = ['co21.image', 'co32.image', 'cont.image']

    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            if d not in skiplist:
                for end in endings:
                    image_path = os.path.join(root, d, end)
                    if os.path.isdir(image_path):
                        fits_path = image_path.replace('.image', '.fits')
                        if not os.path.exists(fits_path):
                            print(f"Exporting {image_path} → {fits_path}")
                            export_imaging_to_fits(
                                image_root=image_path[:-6],  # strip ".image"
                                just_image=True
                            )
                        else:
                            print(f"Skipping existing FITS: {fits_path}")

# --- Run ---
skiplist = ['NGC4593','NGC2775']
base_dir = '/data/c3040163/llama/alma/phangs_imaging_scripts-master/full_run_newkeys_all_arrays/reduction/imaging'
export_all_images(base_dir, skiplist=skiplist)

#!/usr/bin/env python
import os
from casatasks import exportfits as casa_exportfits

# ---- CONFIG ----
base_dir = '/data/c3040163/llama/alma/phangs_imaging_scripts-master/full_run_newkeys_all_arrays/reduction/imaging'
skip_targets = {'NGC4593', 'NGC2775'}
bitpix = -32
# ----------------

SUFFIXES = [
    # co21 set
    'co21.image', 'co21.mask', 'co21.weight', 'co21.model', 'co21.pb', 'co21.psf', 'co21.residual',
    # cont set (including multi-term tt variants)
    'cont.image', 'cont.mask', 'cont.weight', 'cont.model', 'cont.pb', 'cont.psf', 'cont.residual',
    'cont.alpha', 'cont.alpha.error',
    'cont.image.tt0', 'cont.image.tt1',
    'cont.model.tt0', 'cont.model.tt1',
    'cont.pb.tt0',
    'cont.psf.tt0', 'cont.psf.tt1', 'cont.psf.tt2',
    'cont.residual.tt0', 'cont.residual.tt1',
    'cont.weight.tt0', 'cont.weight.tt1', 'cont.weight.tt2'
]


def fits_name_from_casa_dirname(dirname):
    """Apply naming rules for FITS output."""
    name = dirname.replace('.', '_')
    name = name.replace('_image', '')  # drop '_image'
    name = name.replace('_tt0', '')    # drop '_tt0'
    return name + '.fits'


def export_one(casa_dirpath, fits_path):
    """Call CASA exportfits safely."""
    try:
        print(f'Exporting: {casa_dirpath}  ->  {fits_path}')
        casa_exportfits(
            imagename=casa_dirpath,
            fitsimage=fits_path,
            velocity=True,
            overwrite=True,
            dropstokes=True,
            dropdeg=True,
            bitpix=bitpix
        )
        print('  ✅ done')
    except Exception as e:
        print(f'  ❌ exportfits failed for {casa_dirpath}: {e}')


def main():
    if not os.path.isdir(base_dir):
        raise SystemExit(f'Base directory not found: {base_dir}')

    for target_name in sorted(os.listdir(base_dir)):
        target_path = os.path.join(base_dir, target_name)
        if not os.path.isdir(target_path):
            continue
        if target_name in skip_targets:
            print(f"\nSkipping target (in skiplist): {target_name}")
            continue

        print(f"\n=== Processing target: {target_name} ===")

        # Collect directory entries (depth 1)
        dirnames = [d for d in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, d))]
        found_suffixes = set()

        for entry in sorted(dirnames):
            for suf in SUFFIXES:
                if entry.endswith(suf):
                    found_suffixes.add(suf)
                    casa_path = os.path.join(target_path, entry)
                    fits_path = os.path.join(target_path, fits_name_from_casa_dirname(entry))
                    if os.path.exists(fits_path):
                        print(f"Skipping existing FITS: {fits_path}")
                    else:
                        export_one(casa_path, fits_path)
                    break

        # Report missing expected suffixes
        missing_suffixes = [s for s in SUFFIXES if s not in found_suffixes]
        if missing_suffixes:
            print(f"  ⚠️  Missing CASA products for {target_name}:")
            for m in missing_suffixes:
                print(f"     - {m}")
        else:
            print(f"  ✅ All expected CASA products found for {target_name}")


if __name__ == '__main__':
    main()

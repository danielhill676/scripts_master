#!/usr/bin/env python3
"""
Export PHANGS CASA imaging dirs to FITS (depth=1).

Rules implemented:
 - Look only one level down: base_dir/<TARGET>/
 - For each target, check immediate directory entries (no recursion).
 - Expected CASA "suffixes" (e.g. 'co21.image', 'cont.image', 'cont.image.tt1', etc.)
 - A suffix is considered present if:
     * a CASA directory exists whose name endswith the suffix (or suffix + .ttX for cont)
     * OR a previously exported FITS exists that matches the naming convention
 - Export using casatasks.exportfits when a CASA dir exists and FITS is absent.
 - Print explicit missing suffixes per target.
"""

import os
import sys
from casatasks import exportfits

co32 = True  # set to True to process CO(3-2) products instead of CO(2-1)

# ----------------- USER CONFIG -----------------
base_dir = '/data/c3040163/llama/alma/phangs_imaging_scripts-master/full_run_newkeys_all_arrays/reduction/imaging'
if co32:
    base_dir = '/data/c3040163/llama/alma/phangs_imaging_scripts-master/CO32_all_arrays/reduction/imaging' 
skip_targets = set([])  # example skiplist; adjust if needed
# ------------------------------------------------
line = 'co21'
if co32:
    line = 'co32'

# Suffix list required (base suffixes)
LINE_SUFFIXES = [
    f'{line}.image', f'{line}.mask', f'{line}.weight', f'{line}.model', f'{line}.pb', f'{line}.psf', f'{line}.residual'
]
CONT_SUFFIXES = [
    'cont.alpha.error', 'cont.alpha',
    'cont.image.tt0', 'cont.image.tt1',
    'cont.model.tt0', 'cont.model.tt1',
    'cont.pb.tt0',
    'cont.psf.tt0', 'cont.psf.tt1', 'cont.psf.tt2',
    'cont.residual.tt0', 'cont.residual.tt1',
    'cont.weight.tt0', 'cont.weight.tt1', 'cont.weight.tt2'
]

# Combined expected suffixes for reporting
EXPECTED_SUFFIXES = LINE_SUFFIXES + CONT_SUFFIXES

# helper functions ---------------------------------------------------------

def fits_name_from_casa_dir(casa_dirname):
    """
    Convert a CASA directory name (e.g. NGC5845_12m_cont.image.tt1 or NGC5845_12m_co21.image)
    to the target FITS filename according to the rules:
      - replace '.' -> '_'
      - remove the literal substring '_image' (so .image variants don't add '_image')
      - remove trailing '_tt0' (so tt0 files do not add '_tt0'), but keep '_tt1'/_tt2
    Return filename (no path).
    """
    fn = casa_dirname.replace('.', '_')
    # remove the _image substring if present
    fn = fn.replace('_image', '')
    # remove _tt0 (but keep other ttN)
    fn = fn.replace('_tt0', '')
    return fn + '.fits'

def probable_fits_candidates_for_suffix(target_name, suffix):
    """
    Generate one or two plausible FITS names that could already exist and
    should be treated as evidence the suffix is present.
    Eg suffix 'cont.image.tt1' -> target_name + '_cont_tt1.fits' (per user examples)
       suffix 'co21.weight' -> target_name + '_co21_weight.fits'
       suffix 'cont.alpha.error' -> target_name + '_cont_alpha_error.fits'
    We'll return a small set of guessed names to check for existence.
    """
    base = suffix.replace('.', '_')
    # special handling to match the user's naming conventions:
    # - .image files: they don't want '_image' in final FITS; for tt1 we add '_tt1'
    candidates = []
    if suffix.startswith('cont.image'):
        # cont.image.tt1 -> _cont_tt1.fits ; cont.image.tt0 -> _cont.fits
        if '.tt1' in suffix:
            candidates.append(f"{target_name}_cont_tt1.fits")
        elif '.tt2' in suffix:
            candidates.append(f"{target_name}_cont_tt2.fits")
        else:
            # plain cont.image or tt0 -> just target_cont.fits
            candidates.append(f"{target_name}_cont.fits")
    elif suffix.startswith('cont.') and ('alpha' in suffix):
        # cont.alpha.error -> target_cont_alpha_error.fits
        candidates.append(f"{target_name}_{base}.fits")
    else:
        # general mapping: target_<suffix with dots->underscores>.fits
        candidates.append(f"{target_name}_{base}.fits")
    # Also be liberal: also check a more general candidate (some runs produce other variants)
    candidates.append(f"{target_name}_{base}_tt0.fits")
    return list(dict.fromkeys(candidates))  # unique order-preserving

def export_one(casa_path, fits_path):
    """Run exportfits for a CASA dir -> fits path."""
    print(f"Exporting: {casa_path} -> {fits_path}")
    try:
        exportfits(imagename=casa_path, fitsimage=fits_path,
                   velocity=True, overwrite=True, dropstokes=True, dropdeg=True, bitpix=-32)
    except Exception as e:
        print(f"  ❌ exportfits FAILED for {casa_path} -> {fits_path} : {e}")

# main loop ----------------------------------------------------------------

def main():
    if not os.path.isdir(base_dir):
        print(f"Base directory not found: {base_dir}", file=sys.stderr)
        return 1

    targets = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    if not targets:
        print("No target directories found under base_dir.")
        return 0

    for target in targets:
        if target in skip_targets:
            print(f"\nSkipping target (in skiplist): {target}")
            continue

        targdir = os.path.join(base_dir, target)
        print(f"\n=== Target: {target} ===")
        # list immediate entries (files & dirs) only
        entries = os.listdir(targdir)
        dir_entries = [d for d in entries if os.path.isdir(os.path.join(targdir, d))]
        file_entries = [f for f in entries if os.path.isfile(os.path.join(targdir, f))]
        dirset = set(dir_entries)
        fileset = set(file_entries)

        found = set()  # suffixes we've satisfied

        # Check each expected suffix
        for suf in EXPECTED_SUFFIXES:
            satisfied = False

            # 1) Does any CASA dir endwith this suffix OR (for cont) .tt variants?
            #    We check immediate dir entries only.
            #    A dir like 'NGC2775_12m_cont.image.tt1' should satisfy 'cont.image.tt1' and 'cont.image'
            for d in dir_entries:
                # direct match
                if d.endswith(suf):
                    satisfied = True
                    # remember the actual directory that satisfied it
                    matched_dir = d
                    break
                # if suffix is cont.<x> (base cont type) allow tt variants to satisfy base cont suffix
                if suf.startswith('cont.') and d.endswith(suf + '.tt0'):
                    satisfied = True
                    matched_dir = d
                    break
                if suf.startswith('cont.') and d.endswith(suf + '.tt1'):
                    satisfied = True
                    matched_dir = d
                    break
                if suf.startswith('cont.') and d.endswith(suf + '.tt2'):
                    satisfied = True
                    matched_dir = d
                    break
                # also allow base cont.image to be satisfied by cont.image.tt1/tt0 etc:
                if suf.startswith('cont.') and suf.count('.') == 1:
                    # suf like 'cont.image' : check if dir endswith 'cont.image.ttX'
                    if d.endswith(suf + '.tt0') or d.endswith(suf + '.tt1') or d.endswith(suf + '.tt2'):
                        satisfied = True
                        matched_dir = d
                        break

            if satisfied:
                found.add(suf)
                # If the CASA dir existed, attempt export if FITS absent
                # find the matching CASA dir to use for export (prefer exact match, else any matching)
                match_dir_for_export = None
                for d in dir_entries:
                    if d.endswith(suf):
                        match_dir_for_export = d
                        break
                if match_dir_for_export is None:
                    # choose first dir that endswith the base cont.* or .tt
                    for d in dir_entries:
                        if suf.startswith('cont.') and (d.endswith(suf) or d.endswith(suf + '.tt1') or d.endswith(suf + '.tt0') or d.endswith(suf + '.tt2')):
                            match_dir_for_export = d
                            break

                if match_dir_for_export:
                    casa_path = os.path.join(targdir, match_dir_for_export)
                    fits_name = fits_name_from_casa_dir(match_dir_for_export)
                    fits_path = os.path.join(targdir, fits_name)
                    if os.path.exists(fits_path):
                        print(f"Skipping existing FITS: {fits_path}")
                    else:
                        export_one(casa_path, fits_path)

                continue  # next suffix

            # 2) Not matched by CASA dir — check for existing FITS that indicate exported product
            #    Use plausible FITS names and also any file that contains suffix->underscores
            candidates = probable_fits_candidates_for_suffix(target, suf)
            for cand in candidates:
                if cand in fileset:
                    satisfied = True
                    break
            # also looser check: any existing fits that contains the suffix pattern
            if not satisfied:
                for f in file_entries:
                    if f.endswith('.fits') and (suf.replace('.', '_') in f):
                        satisfied = True
                        break
                    # treat cont_tt presence as evidence of cont.image existence
                    if suf.startswith('cont.') and ('cont_tt' in f or f.startswith(target + '_cont_')):
                        # only accept if file starts with target (reduce false positives)
                        if f.startswith(target):
                            satisfied = True
                            break

            if satisfied:
                found.add(suf)
                # nothing to export (no CASA dir), FITS already present
                print(f"Found existing FITS for {suf} (no CASA dir) for {target}")
                continue

            # if we get here suffix is not satisfied
            # do not print missing immediately — collect and print after loop
            # but mark nothing

        # Report missing suffixes explicitly (those not in found)
        missing = [s for s in EXPECTED_SUFFIXES if s not in found]
        if missing:
            print(f"\n  ⚠️  Missing CASA/FITS products for {target}:")
            for m in missing:
                print(f"     - {m}")
        else:
            print(f"  ✅ All expected CASA/FITS products found for {target}")

    return 0

if __name__ == '__main__':
    rc = main()
    sys.exit(rc)







######3

#!/usr/bin/env python3
"""
Export and import PHANGS CASA imaging dirs to/from FITS (depth=1).

Adds importfits: if a FITS exists but the CASA directory is missing, 
it creates the CASA image using the same name conversion.
"""

import os
import sys
from casatasks import exportfits, importfits

co32 = True  # set to True to process CO(3-2) products instead of CO(2-1)

# ----------------- USER CONFIG -----------------
base_dir = '/data/c3040163/llama/alma/phangs_imaging_scripts-master/full_run_newkeys_all_arrays/reduction/imaging'
if co32:
    base_dir = '/data/c3040163/llama/alma/phangs_imaging_scripts-master/CO32_all_arrays/reduction/imaging' 
skip_targets = set([])  # example skiplist; adjust if needed
# ------------------------------------------------
line = 'co21'
if co32:
    line = 'co32'

# Suffix list required (base suffixes)
LINE_SUFFIXES = [
    f'{line}.image', f'{line}.mask', f'{line}.weight', f'{line}.model', f'{line}.pb', f'{line}.psf', f'{line}.residual'
]
CONT_SUFFIXES = [
    'cont.alpha.error', 'cont.alpha',
    'cont.image.tt0', 'cont.image.tt1',
    'cont.model.tt0', 'cont.model.tt1',
    'cont.pb.tt0',
    'cont.psf.tt0', 'cont.psf.tt1', 'cont.psf.tt2',
    'cont.residual.tt0', 'cont.residual.tt1',
    'cont.weight.tt0', 'cont.weight.tt1', 'cont.weight.tt2'
]

# Combined expected suffixes for reporting
EXPECTED_SUFFIXES = LINE_SUFFIXES + CONT_SUFFIXES


# helper functions ---------------------------------------------------------

def fits_name_from_casa_dir(casa_dirname):
    """Convert CASA directory name to FITS name (same as before)."""
    fn = casa_dirname.replace('.', '_')
    fn = fn.replace('_image', '')
    fn = fn.replace('_tt0', '')
    return fn + '.fits'


def probable_fits_candidates_for_suffix(target_name, suffix):
    """Guess possible FITS names for a given suffix."""
    base = suffix.replace('.', '_')
    candidates = []
    if suffix.startswith('cont.image'):
        if '.tt1' in suffix:
            candidates.append(f"{target_name}_cont_tt1.fits")
        elif '.tt2' in suffix:
            candidates.append(f"{target_name}_cont_tt2.fits")
        else:
            candidates.append(f"{target_name}_cont.fits")
    elif suffix.startswith('cont.') and ('alpha' in suffix):
        candidates.append(f"{target_name}_{base}.fits")
    else:
        candidates.append(f"{target_name}_{base}.fits")
    candidates.append(f"{target_name}_{base}_tt0.fits")
    return list(dict.fromkeys(candidates))  # unique, order-preserving


def export_one(casa_path, fits_path):
    """Run exportfits for a CASA dir -> FITS path."""
    print(f"Exporting: {casa_path} -> {fits_path}")
    try:
        exportfits(imagename=casa_path, fitsimage=fits_path,
                   velocity=True, overwrite=True, dropstokes=True, dropdeg=True, bitpix=-32)
    except Exception as e:
        print(f"  ❌ exportfits FAILED for {casa_path} -> {fits_path} : {e}")


def import_one(fits_path, casa_path):
    """Run importfits for a FITS file -> CASA image."""
    print(f"Importing: {fits_path} -> {casa_path}")
    try:
        importfits(fitsimage=fits_path, imagename=casa_path,
                   overwrite=True, zeroblanks=True, defaultaxes=True)
    except Exception as e:
        print(f"  ❌ importfits FAILED for {fits_path} -> {casa_path} : {e}")


# main loop ----------------------------------------------------------------

def main():
    if not os.path.isdir(base_dir):
        print(f"Base directory not found: {base_dir}", file=sys.stderr)
        return 1

    targets = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    if not targets:
        print("No target directories found under base_dir.")
        return 0

    for target in targets:
        if target in skip_targets:
            print(f"\nSkipping target (in skiplist): {target}")
            continue

        targdir = os.path.join(base_dir, target)
        print(f"\n=== Target: {target} ===")
        entries = os.listdir(targdir)
        dir_entries = [d for d in entries if os.path.isdir(os.path.join(targdir, d))]
        file_entries = [f for f in entries if os.path.isfile(os.path.join(targdir, f))]
        fileset = set(file_entries)

        found = set()

        # Check each expected suffix
        for suf in EXPECTED_SUFFIXES:
            satisfied = False
            matched_dir = None

            # 1) CASA dir exists?
            for d in dir_entries:
                if d.endswith(suf) or (
                    suf.startswith('cont.') and any(d.endswith(suf + tt) for tt in ('.tt0', '.tt1', '.tt2'))
                ):
                    satisfied = True
                    matched_dir = d
                    break

            if matched_dir:
                found.add(suf)
                casa_path = os.path.join(targdir, matched_dir)
                fits_name = fits_name_from_casa_dir(matched_dir)
                fits_path = os.path.join(targdir, fits_name)

                if os.path.exists(fits_path):
                    print(f"Skipping existing FITS: {fits_path}")
                else:
                    export_one(casa_path, fits_path)
                continue

            # 2) FITS exists but no CASA dir
            candidates = probable_fits_candidates_for_suffix(target, suf)
            fits_found = None
            for cand in candidates:
                if cand in fileset:
                    fits_found = cand
                    satisfied = True
                    break

            if fits_found:
                fits_path = os.path.join(targdir, fits_found)
                casa_name = fits_found.replace('.fits', '')
                # revert naming: underscores back to dots, remove target prefix
                if casa_name.startswith(target + '_'):
                    casa_name = casa_name[len(target) + 1:]
                casa_name = casa_name.replace('_', '.')
                casa_path = os.path.join(targdir, f"{target}_{casa_name}")

                if os.path.exists(casa_path):
                    print(f"Found existing FITS for {suf} (CASA already present) for {target}")
                else:
                    import_one(fits_path, casa_path)
                found.add(suf)
                continue

            # 3) no match
            continue

        # Report missing suffixes explicitly
        missing = [s for s in EXPECTED_SUFFIXES if s not in found]
        if missing:
            print(f"\n  ⚠️  Missing CASA/FITS products for {target}:")
            for m in missing:
                print(f"     - {m}")
        else:
            print(f"  ✅ All expected CASA/FITS products found for {target}")

    return 0


if __name__ == '__main__':
    rc = main()
    sys.exit(rc)

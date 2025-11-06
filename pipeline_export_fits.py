def _fits_name_from_casa_dirname(dirname):
    """Apply naming rules for FITS output (same as before)."""
    name = dirname.replace('.', '_')
    name = name.replace('_image', '')  # drop '_image'
    name = name.replace('_tt0', '')    # drop '_tt0'
    return name + '.fits'

def _casa_dir_candidates_for_suffix(target_name, suffix):
    """
    Return a list of possible CASA directory name substrings that
    should count as matching 'suffix'. This implements the
    tt-variant rule: cont.image will match cont.image, cont.image.tt0, cont.image.tt1, ...
    Also makes co21.* match exact endings only.
    """
    if suffix.startswith('cont.'):
        # base 'cont.image' should match 'cont.image', 'cont.image.tt0', 'cont.image.tt1', etc.
        base = suffix  # e.g. 'cont.image' or 'cont.residual'
        return [base, base + '.tt0', base + '.tt1', base + '.tt2']
    else:
        # For line products (co21, co32 etc) we require direct match or likely tt variants are not used
        return [suffix]

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

        # depth=1 subdirectories only
        dirnames = [d for d in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, d))]
        found_suffixes = set()

        # Build a set of dirnames for quick membership tests
        dirset = set(dirnames)

        # Also precompute existing FITS filenames in target_path
        fits_files = set(os.listdir(target_path))

        for suf in SUFFIXES:
            # generate candidate CASA dir substrings that should satisfy 'suf'
            candidates = _casa_dir_candidates_for_suffix(target_name, suf)
            matched = False

            # Check for CASA directory matches (endswith candidate)
            for d in dirset:
                for cand in candidates:
                    if d.endswith(cand):
                        matched = True
                        break
                if matched:
                    break

            # If not matched by CASA dir look for an already-existing FITS file
            if not matched:
                # Build expected FITS filename(s) for this suffix.
                # There are multiple possible FITS names because our export naming rules convert dots to underscores
                # Special-case: .image -> no '_image' in FITS, and .tt0 -> drop _tt0.
                # We'll try a few likely candidates:
                basename = None
                # Try common naming produced by export_imaging_to_fits:
                # e.g., NGC2775_12m_cont_alpha_error.fits  or NGC2775_12m_cont_tt1.fits  or NGC2775_12m_co21_weight.fits
                # Construct prefix from target files already present: try to find any file starting with target_name and containing suffix parts
                for f in fits_files:
                    if f.endswith('.fits') and f.startswith(target_name):
                        # quick heuristic: if suffix parts (replace '.'->'_') appear in filename treat as match
                        if suffix.replace('.', '_') in f:
                            matched = True
                            break
                        # Accept tt variants as fulfilling base cont.image
                        if suffix.startswith('cont.') and ('cont_' + suffix.split('.',1)[1] + '_tt' in f or 'cont_tt' in f):
                            matched = True
                            break

            if matched:
                found_suffixes.add(suf)
                # Export if needed: find the CASA dir that matched (prefer CASA dir export if no FITS exists)
                # We only export when CASA dir exists and corresponding FITS is absent.
                # Find CASA dir path:
                matching_dir = None
                for d in dirset:
                    for cand in _casa_dir_candidates_for_suffix(target_name, suf):
                        if d.endswith(cand):
                            matching_dir = d
                            break
                    if matching_dir:
                        break

                if matching_dir:
                    casa_path = os.path.join(target_path, matching_dir)
                    fits_path = os.path.join(target_path, _fits_name_from_casa_dirname(matching_dir))
                    if os.path.exists(fits_path):
                        print(f"Skipping existing FITS: {fits_path}")
                    else:
                        export_one(casa_path, fits_path)
                else:
                    # no CASA dir matched, but FITS exists -> nothing to export
                    expected_fits_guess = os.path.join(target_path, target_name + '_' + suf.replace('.', '_') + '.fits')
                    if any(f.startswith(target_name) and suffix.replace('.', '_') in f for f in fits_files):
                        print(f"Found FITS for {suf}, skipping export.")
                    else:
                        # nothing to do - will print missing later if truly absent
                        pass

        # Report missing expected suffixes (after checking for CASA dirs and existing FITS)
        missing_suffixes = [s for s in SUFFIXES if s not in found_suffixes]
        if missing_suffixes:
            print(f"  ⚠️  Missing CASA products for {target_name}:")
            for m in missing_suffixes:
                print(f"     - {m}")
        else:
            print(f"  ✅ All expected CASA products found for {target_name}")

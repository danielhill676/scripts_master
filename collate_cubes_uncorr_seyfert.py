from pathlib import Path
import shutil

# Base directories
src_base = Path('/data/c3040163/llama/alma/phangs_imaging_scripts-master/full_run_newkeys_all_arrays/reduction/imaging/')
dst_base = Path('/data/c3040163/llama/alma/pipeline_cubes/uncorrected/')

# Loop through subdirectories
for subdir in src_base.iterdir():
    if subdir.is_dir():
        name = subdir.name
        
        # Construct source file path
        src_cube = subdir / f"{name}_12m_co21.fits"
        
        if src_cube.exists():
            # Construct destination directory and file path
            dst_dir = dst_base / name
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            dst_cube = dst_dir / f"{name}_12m_co21.fits"
            
            # Copy file
            if not dst_cube.exists():
                shutil.copy2(src_cube, dst_cube)
                print(f"Copied: {src_cube} -> {dst_cube}")
        else:
            print(f"Missing: {src_cube}")
          
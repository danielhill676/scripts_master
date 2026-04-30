from pathlib import Path
import shutil

# Base directories
src_base = Path('/data/c3040163/llama/alma/phangs_imaging_scripts-master/full_run_newkeys_all_arrays/reduction/imaging/')
dst_base = Path('/data/c3040163/llama/alma/pipeline_cont/')

# Loop through subdirectories
for subdir in src_base.iterdir():
    if subdir.is_dir():
        name = subdir.name
        
        # Construct source file path
        src_file = subdir / f"{name}_12m_cont.fits"
        
        if src_file.exists():
            # Construct destination directory and file path
            dst_dir = dst_base / name
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            dst_file = dst_dir / f"{name}_12m_cont.fits"
            
            # Copy file
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {src_file} -> {dst_file}")
        else:
            print(f"Missing: {src_file}")

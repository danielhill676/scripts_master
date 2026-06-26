from pathlib import Path
import shutil

# Base directories
src_base = Path('/data/c3040163/llama/alma/phangs_imaging_scripts-master/full_run_newkeys_all_arrays/reduction/postprocess/')
dst_base = Path('/data/c3040163/llama/alma/pipeline_cubes/')

# Loop through subdirectories
for subdir in src_base.iterdir():
    if subdir.is_dir():
        name = subdir.name
        
        # Construct source file path
        src_pb = subdir / f"{name}_12m_co21_trimmed_pb.fits"
        src_cube = subdir / f"{name}_12m_co21_pbcorr_trimmed.fits"
        
        if src_cube.exists() and src_pb.exists():
            # Construct destination directory and file path
            dst_dir = dst_base / name
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            dst_pb = dst_dir / f"{name}_12m_co21_trimmed_pb.fits"
            dst_cube = dst_dir / f"{name}_12m_co21_pbcorr_trimmed.fits"
            
            # Copy file
            if not dst_cube.exists():
                shutil.copy2(src_cube, dst_cube)
                print(f"Copied: {src_cube} -> {dst_cube}")
            if not dst_pb.exists():
                shutil.copy2(src_pb, dst_pb)
                print(f"Copied: {src_pb} -> {dst_pb}")
        elif not src_cube.exists():
            print(f"Missing: {src_cube}")
        elif not src_pb.exists():
            print(f"Missing: {src_pb}")
        else:
            print(f"Missing: {src_cube} and {src_pb}")            

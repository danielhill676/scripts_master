import os
import tarfile
import re

# Root directory containing the data
ROOT_DIR = '/data/c3040163/llama/alma/raw'

# Regex to match valid 'array' directory names
ARRAY_DIR_PATTERN = re.compile(r'^(12m|7m ACA|TP)_\d+$')

def is_valid_array_dir(dirname):
    """Check if the directory name matches the array pattern."""
    return bool(ARRAY_DIR_PATTERN.match(dirname))

def untar_file(tar_path, extract_path):
    """Extract a tar file to the given path."""
    try:
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=extract_path)
        print(f"✓ Extracted: {tar_path}")
    except Exception as e:
        print(f"✗ Failed to extract {tar_path}: {e}")

def process_tar_files():
    for name in os.listdir(ROOT_DIR):
        name_path = os.path.join(ROOT_DIR, name)
        if not os.path.isdir(name_path):
            continue
        
        for subdir in os.listdir(name_path):
            array_path = os.path.join(name_path, subdir)
            if not os.path.isdir(array_path):
                continue
            if not is_valid_array_dir(subdir):
                continue
            
            for item in os.listdir(array_path):
                if item.endswith('.tar'):
                    tar_path = os.path.join(array_path, item)
                    untar_file(tar_path, array_path)

if __name__ == "__main__":
    process_tar_files()

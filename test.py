import os
import glob
import csv
from astropy.io import fits

from astropy.table import Table

import re
from astroquery.simbad import Simbad
import warnings

import re

def normalize_field_name(s):
    """
    Robust normalization of astronomical field names:
    - Uppercase letters
    - Convert each numeric block to integer (removes leading zeros)
    - Remove separators after splitting
    """
    s = s.upper()

    # Split into sequences of letters and numbers
    parts = re.findall(r'\d+|[^\d\s_\-\.]+', s)
    # parts example: ['MCG', '06', '30', '015']

    for i, p in enumerate(parts):
        if p.isdigit():
            parts[i] = str(int(p))  # convert numeric block to int to strip leading zeros

    # Recombine
    normalized = ''.join(parts)
    return normalized



print(normalize_field_name("MCG-06-30-015"))
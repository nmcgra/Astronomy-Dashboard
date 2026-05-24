#!/usr/bin/env python3
"""
read_headers.py — Print key header info for every FITS file in the current folder.
Useful for quickly checking exposure times, temperatures, and frame types.

Usage:
  python3 read_headers.py
  python3 read_headers.py /path/to/folder
"""

import sys
from pathlib import Path

def read_fits_header(path):
    header = {}
    with open(path, "rb") as f:
        while True:
            block = f.read(2880)
            if not block:
                break
            for i in range(36):
                record = block[i*80:(i+1)*80].decode("ascii", errors="replace")
                keyword = record[:8].strip()
                rest    = record[8:].strip()
                if keyword == "END":
                    return header
                if rest.startswith("="):
                    value_comment = rest[1:].strip()
                    in_quote, slash_pos = False, None
                    for j, ch in enumerate(value_comment):
                        if ch == "'": in_quote = not in_quote
                        if ch == "/" and not in_quote:
                            slash_pos = j
                            break
                    value = (value_comment[:slash_pos] if slash_pos else value_comment).strip().strip("'").strip()
                    try:    value = int(value)
                    except: 
                        try: value = float(value)
                        except: pass
                    header[keyword] = value
    return header

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
fits_files = sorted(list(folder.glob("*.fit")) + list(folder.glob("*.fits")))

if not fits_files:
    print("No .fit or .fits files found.")
    sys.exit(1)

keys = ["EXPTIME", "IMAGETYP", "CCD-TEMP", "GAIN", "XBINNING", "DATE-OBS", "INSTRUME", "NAXIS1", "NAXIS2"]

print(f"\nFound {len(fits_files)} FITS files in '{folder}'\n")
print(f"{'Filename':<55} {'EXPTIME':>8} {'TEMP':>6} {'GAIN':>5} {'BIN':>4} {'DATE-OBS'}")
print("-" * 110)

for path in fits_files:
    hdr = read_fits_header(path)
    exp   = hdr.get("EXPTIME",  hdr.get("EXPOSURE", "?"))
    temp  = hdr.get("CCD-TEMP", "?")
    gain  = hdr.get("GAIN",     "?")
    binn  = hdr.get("XBINNING", "?")
    date  = hdr.get("DATE-OBS", "?")
    itype = hdr.get("IMAGETYP", "")
    name  = path.name
    if itype:
        name += f"  [{itype}]"
    print(f"  {name:<53} {str(exp):>8}s {str(temp):>6}°C {str(gain):>5} {str(binn):>4}   {date}")

print()

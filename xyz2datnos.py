#!/usr/bin/env python3
"""
Convert many multi-frame XYZ files into one .dat.nos file,
taking ONLY the *last* frame from each file.

Usage (run in the folder with your XYZ_2.2.1_interp36.*.xyz files):

    python xyz2datnos.py
    # or customize:
    python xyz2datnos.py --pattern "XYZ_2.2.1_interp36.*.xyz" --output "XYZ_2.2.1.dat.nos"

Notes:
- Assumes standard XYZ format: first line = N (atoms), second line = comment, then N lines (elem x y z).
- All last frames across files must share the same atom count N.
- Writes floats as scientific notation with high precision.
"""

import argparse
import glob
import os
import re
import sys

def natural_key(filename):
    """
    Sort key that extracts the numeric index after 'interp36.' in names like:
    'XYZ_2.2.1_interp36.91.xyz' -> 91
    Falls back to the full filename if no match.
    """
    m = re.search(r'interp36\.(\d+)\.xyz$', os.path.basename(filename))
    if m:
        return (int(m.group(1)), filename)
    return (float('inf'), filename)

def read_last_xyz_frame(path):
    """
    Return the last frame as a list of (x,y,z) floats, or raise on parse error.
    Memory-efficient: streams through the file and keeps only the most recent frame.
    """
    last_frame = None
    N_current = None
    with open(path, "r") as fh:
        while True:
            # Find next frame's atom count
            line = fh.readline()
            if not line:
                break  # EOF
            line = line.strip()
            if line == "":
                continue
            try:
                n = int(line)
            except ValueError:
                raise ValueError(f"[{path}] Expected integer atom count, got: {line!r}")

            # Skip comment line
            comment = fh.readline()
            if not comment:
                raise EOFError(f"[{path}] Unexpected EOF while reading comment line")

            frame = []
            for _ in range(n):
                atom_line = fh.readline()
                if not atom_line:
                    raise EOFError(f"[{path}] Unexpected EOF while reading atom lines")
                parts = atom_line.split()
                if len(parts) < 4:
                    raise ValueError(f"[{path}] Bad XYZ atom line (need >=4 columns): {atom_line!r}")
                x, y, z = map(float, parts[-3:])
                frame.append((x, y, z))

            # Keep only the last one seen
            last_frame = frame
            N_current = n

    if last_frame is None:
        raise ValueError(f"[{path}] No frames found")
    return last_frame, N_current

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="XYZ_2.2.1_interp36.*.xyz",
                   help="Glob pattern for input XYZ files (default: %(default)s)")
    p.add_argument("--output", default="XYZ_2.2.1.dat.nos",
                   help="Output .dat.nos filename (default: %(default)s)")
    args = p.parse_args()

    files = sorted(glob.glob(args.pattern), key=natural_key)
    if not files:
        print(f"No files match pattern: {args.pattern}", file=sys.stderr)
        sys.exit(1)

    N_expected = None
    written = 0

    with open(args.output, "w") as outfh:
        for fn in files:
            last_frame, N = read_last_xyz_frame(fn)
            if N_expected is None:
                N_expected = N
            elif N != N_expected:
                raise ValueError(
                    f"Atom count mismatch: expected {N_expected}, got {N} in file {fn}"
                )
            for x, y, z in last_frame:
                outfh.write(f"{x:.16e} {y:.16e} {z:.16e}\n")
            written += 1
            print(f"Wrote last frame from {os.path.basename(fn)} (N={N})")

    print("\nDone.")
    print(f"Output: {args.output}")
    print(f"Beads per configuration (Nbeads): {N_expected}")
    print(f"Total configurations (files) written: {written}")
    print(f"Total rows in .dat.nos: {written * (N_expected or 0)}")

if __name__ == "__main__":
    main()


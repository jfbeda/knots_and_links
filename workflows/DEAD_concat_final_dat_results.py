#!/usr/bin/env python3
import argparse
import glob
import os
import sys

def read_final_file(path):
    """Reads a FINAL_xyz.dat file and returns a list of (x,y,z) floats."""
    coords = []
    with open(path, "r") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Bad line in {path}: {line!r}")
            coords.append(tuple(map(float, parts)))
    return coords, len(coords)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", required=True,
                   help="Glob pattern to FINAL_*.dat files")
    p.add_argument("--output", required=True,
                   help="Output .dat.nos file")
    args = p.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"No files match pattern: {args.pattern}", file=sys.stderr)
        sys.exit(1)

    N_expected = None
    total_written = 0

    with open(args.output, "w") as outfh:
        for fn in files:
            coords, N = read_final_file(fn)

            if N_expected is None:
                N_expected = N
            elif N != N_expected:
                raise ValueError(
                    f"Atom count mismatch: expected {N_expected}, got {N} in file {fn}"
                )

            for x, y, z in coords:
                outfh.write(f"{x:.16e} {y:.16e} {z:.16e}\n")

            total_written += 1
            print(f"Wrote {os.path.basename(fn)} (N={N})")

    print("\nDone.")
    print(f"Output written to: {args.output}")
    print(f"Beads per configuration: {N_expected}")
    print(f"Total configurations: {total_written}")
    print(f"Total rows written: {total_written * N_expected}")

if __name__ == "__main__":
    main()

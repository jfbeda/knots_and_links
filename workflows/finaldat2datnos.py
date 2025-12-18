#!/usr/bin/env python3

# Call using
# python finaldat2datnos.py --infolder <name> --outfolder <name>

import glob
import os
import re
import argparse
import shutil
from datetime import datetime


# Matches patterns like FINAL_2.2.1.3.dat → prefix FINAL_2.2.1, index 3
FILE_RE = re.compile(r'^(FINAL_\d+\.\d+\.\d+)\.(\d+)\.dat$')

def parse_filename(filename):
    """Extract prefix and numeric index from FINAL_X.Y.Z.i.dat."""
    base = os.path.basename(filename)
    m = FILE_RE.match(base)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))



# def extract_xyz_from_lammps(path):
#     atoms = []
#     in_atoms = False

#     with open(path) as f:
#         for line in f:
#             if line.strip().startswith("Atoms"):
#                 in_atoms = True
#                 next(f)  # skip blank line
#                 continue

#             if in_atoms:
#                 line = line.strip()
#                 if line == "":
#                     break
#                 parts = line.split()
#                 atom_id = int(parts[0])
#                 x, y, z = map(float, parts[3:6])
#                 atoms.append((atom_id, x, y, z))

#     atoms.sort(key=lambda t: t[0])

#     # return only XYZ in sorted order
#     return [(x, y, z) for (_, x, y, z) in atoms]

def extract_xyz_from_lammps(path):
    atoms = []
    in_atoms = False

    # Box bounds
    xlo = xhi = ylo = yhi = zlo = zhi = None

    with open(path) as f:
        for line in f:
            line_stripped = line.strip()

            # --- Read box bounds ---
            if line_stripped.endswith("xlo xhi"):
                xlo, xhi = map(float, line_stripped.split()[:2])
            elif line_stripped.endswith("ylo yhi"):
                ylo, yhi = map(float, line_stripped.split()[:2])
            elif line_stripped.endswith("zlo zhi"):
                zlo, zhi = map(float, line_stripped.split()[:2])

            # --- Enter Atoms section ---
            elif line_stripped.startswith("Atoms"):
                in_atoms = True
                next(f)  # skip blank line
                continue

            # --- Read atoms ---
            elif in_atoms:
                if line_stripped == "":
                    break

                parts = line_stripped.split()

                atom_id = int(parts[0])
                x, y, z = map(float, parts[3:6])

                # Image flags (may or may not be present)
                if len(parts) >= 9:
                    ix, iy, iz = map(int, parts[6:9])
                else:
                    ix = iy = iz = 0

                atoms.append((atom_id, x, y, z, ix, iy, iz))

    # Sanity check
    if None in (xlo, xhi, ylo, yhi, zlo, zhi):
        raise ValueError(f"Box bounds not found in {path}")

    # Box lengths
    Lx = xhi - xlo
    Ly = yhi - ylo
    Lz = zhi - zlo

    # Sort by atom ID
    atoms.sort(key=lambda t: t[0])

    # Apply periodic shifts and return XYZ
    xyz = []
    for _, x, y, z, ix, iy, iz in atoms:
        xu = x + ix * Lx
        yu = y + iy * Ly
        zu = z + iz * Lz
        xyz.append((xu, yu, zu))

    return xyz



# ---------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--infolder", required=True,
                   help="Folder containing FINAL_X.Y.Z.i.dat files")
    p.add_argument("--outfolder", required=True,
                   help="Folder where XYZ_X.Y.Z.dat.nos outputs will be written")
    p.add_argument(
    "--archive-name",
    default=None,
    help="Name of archive directory under Archive/. "
         "If omitted, a timestamped name is used."
)

    args = p.parse_args()

    infolder = args.infolder
    outfolder = args.outfolder

    if not os.path.isdir(infolder):
        raise SystemExit(f"Input folder does not exist: {infolder}")

    os.makedirs(outfolder, exist_ok=True)

    # Find all FINAL_*.dat files in input folder
    search_path = os.path.join(infolder, "FINAL_*.dat")
    all_files = glob.glob(search_path)

    groups = {}

    # Group by prefix
    for fn in all_files:
        prefix, idx = parse_filename(fn)
        if prefix is None:
            continue
        groups.setdefault(prefix, []).append((idx, fn))

    if not groups:
        raise SystemExit("No FINAL_X.Y.Z.i.dat files found.")

    # Process each group independently
    for prefix, filelist in groups.items():
        print(f"\nProcessing group: {prefix}")

        # Sort by numeric index
        filelist.sort(key=lambda x: x[0])

        # Build output filename
        outname = prefix.replace("FINAL_", "XYZ_") + ".dat.nos"
        outpath = os.path.join(outfolder, outname)

        all_xyz = []
        N_expected = None

        for idx, fn in filelist:
            coords = extract_xyz_from_lammps(fn)

            if N_expected is None:
                N_expected = len(coords)
            elif len(coords) != N_expected:
                raise ValueError(
                    f"Atom count mismatch in {fn}: expected {N_expected}, got {len(coords)}"
                )

            all_xyz.extend(coords)
            print(f"  Read {len(coords)} atoms from {fn}")

        # Write out the combined xyz data
        with open(outpath, "w") as out:
            for x, y, z in all_xyz:
                out.write(f"{x:.16e} {y:.16e} {z:.16e}\n")

        print(f"  → Wrote {len(all_xyz)} coordinates to {outpath}")
        print(f"  Frames merged: {len(filelist)}   Atoms per frame: {N_expected}")


    knot_data_dir = infolder    # KNOT_data
    archive_root = "Archive"

    # Build archive name
    if args.archive_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"KNOT_data_FINAL_{timestamp}"
    else:
        archive_name = args.archive_name

    archive_dir = os.path.join(archive_root, archive_name)

    if os.path.exists(archive_dir):
        print(f"Overwriting existing archive: {archive_dir}")
        shutil.rmtree(archive_dir)

    os.makedirs(archive_dir)


    # Find FINAL files inside KNOT_data
    final_files = glob.glob(os.path.join(knot_data_dir, "FINAL_*.dat"))

    if not final_files:
        print("Warning: no FINAL_*.dat files found to archive.")
    else:
        for fn in final_files:
            dst = os.path.join(archive_dir, os.path.basename(fn))
            shutil.copy2(fn, dst)

        print(f"\nArchived {len(final_files)} FINAL files → {archive_dir}")
    
    print("\nDone.")

# -------------------------

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import glob
import subprocess
from pathlib import Path

# This file is in: knots_and_links/workflows/
# BASE_DIR = knots_and_links/
BASE_DIR = Path(__file__).resolve().parent.parent

# Input .dat files (one subfolder per length)
LAMMPS_DAT_ROOT = BASE_DIR / "3_lammps_normalized_coordinates_variable_length"

# LAMMPS executable and input script
LMP_BINARY = BASE_DIR / "lammps_simulation" / "lmp_serial"
LAMMPS_INPUT = BASE_DIR / "lammps_simulation" / "runbulk.lam"

# Number of runs per .dat (runid 0..NUM_RUNS-1)
NUM_RUNS = 10


def run_lammps_for_all_lengths():
    """Run LAMMPS for every .dat file in each subfolder of 3_lammps_normalized_coordinates_variable_length."""
    
    # Find all subfolders like: 3_lammps_normalized_coordinates_51, _75, _100, ...
    subdirs = sorted(
        d
        for d in glob.glob(str(LAMMPS_DAT_ROOT / "3_lammps_normalized_coordinates_*"))
        if os.path.isdir(d)
    )

    if not subdirs:
        print(f"No subdirectories found under {LAMMPS_DAT_ROOT}")
        return

    for subdir in subdirs:
        print(f"\n=== Running simulations for directory: {subdir} ===")
        
        dat_files = sorted(glob.glob(os.path.join(subdir, "*.dat")))
        if not dat_files:
            print(f"  No .dat files found in {subdir}, skipping.")
            continue

        # Relative path from BASE_DIR, e.g.:
        # "3_lammps_normalized_coordinates_variable_length/3_lammps_normalized_coordinates_51"
        simdir_rel = os.path.relpath(subdir, BASE_DIR)

        for dat_path in dat_files:
            base = os.path.splitext(os.path.basename(dat_path))[0]
            print(f"  Running simulations for base: {base}")

            for i in range(NUM_RUNS):
                seed = 10000 + i

                cmd = [
                    str(LMP_BINARY),
                    "-in", str(LAMMPS_INPUT),
                    "-var", "simdir", simdir_rel,
                    "-var", "simname", base,
                    "-var", "runid", str(i),
                    "-var", "seed", str(seed),
                ]

                # cwd=BASE_DIR ensures runbulk.lam sees the correct relative paths
                subprocess.run(cmd, check=True, cwd=str(BASE_DIR))


if __name__ == "__main__":
    run_lammps_for_all_lengths()

#!/usr/bin/env python3

# Call using:
# python -m workflows.3_to_4_variable_num_points_run_all

import os
import glob
import subprocess
from pathlib import Path
import shutil

NUM_RUNS = 3

BASE_DIR = Path(__file__).resolve().parent.parent

# Input folders containing the .dat files for each polymer length
INPUT_ROOT = BASE_DIR / "3_lammps_normalized_coordinates_variable_length"

# Final XYZ output (per length)
XYZ_ROOT = BASE_DIR / "4_xyz_files_variable_length"

# Delete and recreate output root
if os.path.exists(XYZ_ROOT):
    import shutil
    shutil.rmtree(XYZ_ROOT)
os.makedirs(XYZ_ROOT, exist_ok=True)

# LAMMPS executable + script
LMP_BINARY = BASE_DIR / "lammps_simulation" / "lmp_serial"
LAMMPS_INPUT = BASE_DIR / "lammps_simulation" / "runbulk.lam"

# Python converter
CONVERTER = BASE_DIR / "workflows" / "finaldat2datnos.py"


def run_all():
    """Run LAMMPS + concatenate FINAL_* files for each length folder."""

    # Find folders like: 3_lammps_normalized_coordinates_51, 75, 100, etc.
    length_dirs = sorted(
        d for d in glob.glob(str(INPUT_ROOT / "3_lammps_normalized_coordinates_*"))
        if os.path.isdir(d)
    )

    if not length_dirs:
        print("No length directories found.")
        return

    for length_dir in length_dirs:
        length_dir = Path(length_dir)
        length_name = length_dir.name     # e.g., "3_lammps_normalized_coordinates_51"
        length = length_name.split("_")[-1]  # "51"

        print(f"\n=== Processing length {length} ===")

        # Output folder for XYZ files for this length
        xyz_out_dir = XYZ_ROOT / f"4_xyz_files_{length}"
        xyz_out_dir.mkdir(parents=True, exist_ok=True)

        # LAMMPS output folder is ALWAYS the same
        lmp_out_dir = BASE_DIR / "lammps_simulation" / "KNOT_data"

        # Clean KNOT_data before each length
        if lmp_out_dir.exists():
            shutil.rmtree(lmp_out_dir)
        lmp_out_dir.mkdir(parents=True)

        # Find all input .dat files for this length
        input_dats = sorted(glob.glob(str(length_dir / "*.dat")))
        if not input_dats:
            print("No .dat files found, skipping.")
            continue

        # -------- Run LAMMPS on all input files --------
        for dat_path in input_dats:
            simname = Path(dat_path).stem
            print(f"  Running LAMMPS for: {simname}")

            for i in range(NUM_RUNS):
                seed = 10000 + i
                cmd = [
                    str(LMP_BINARY),
                    "-in", str(LAMMPS_INPUT),
                    "-var", "simdir", str(length_dir.relative_to(BASE_DIR)),
                    "-var", "simname", simname,
                    "-var", "runid", str(i),
                    "-var", "seed", str(seed),
                ]
                subprocess.run(cmd, check=True, cwd=BASE_DIR)

        print("  LAMMPS runs finished; converting FINAL_*.dat → XYZ_*.dat.nos")

        # -------- Convert FINAL_*.dat into XYZ_*.dat.nos --------
        subprocess.run(
            [
                "python",
                str(CONVERTER),
                "--infolder", str(lmp_out_dir),
                "--outfolder", str(xyz_out_dir),
                "--archive-name", f"KNOT_data_{length}_points"
            ],
            check=True,
        )



if __name__ == "__main__":
    run_all()

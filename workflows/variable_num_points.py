#!/usr/bin/env python3
import os
import glob
import shutil
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

LAMMPS_DAT_ROOT = BASE_DIR / "3_lammps_normalized_coordinates_variable_length"
LMP_BINARY = BASE_DIR / "lammps_simulation" / "lmp_serial"
LAMMPS_INPUT = BASE_DIR / "lammps_simulation" / "runbulk.lam"

KNOT_DATA = BASE_DIR / "lammps_simulation" / "KNOT_data"
OUTPUT_ROOT = BASE_DIR / "4_xyz_files_variable_length"

CONCAT_SCRIPT = BASE_DIR / "workflows" / "concat_final_dat_results.py"

NUM_RUNS = 1000


def run_lammps_for_folder(subdir):
    dat_files = sorted(glob.glob(os.path.join(subdir, "*.dat")))
    if not dat_files:
        print(f"  No .dat files in {subdir}")
        return []

    simdir_rel = os.path.relpath(subdir, BASE_DIR)
    bases = set()

    for dat_path in dat_files:
        base = os.path.splitext(os.path.basename(dat_path))[0]
        bases.add(base)
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

            subprocess.run(cmd, check=True, cwd=str(BASE_DIR))

    return sorted(bases)


def concat_results_for_folder(x_value, bases):
    outdir = OUTPUT_ROOT / f"4_xyz_files_variable_length_{x_value}"
    outdir.mkdir(parents=True, exist_ok=True)

    for base in bases:
        pattern = str(KNOT_DATA / f"FINAL_{base}.*.dat")
        outpath = outdir / f"{base}.dat.nos"

        print(f"  Concatenating for base {base} → {outpath}")

        cmd = [
            "python",
            str(CONCAT_SCRIPT),
            "--pattern", pattern,
            "--output", str(outpath),
        ]

        subprocess.run(cmd, check=True, cwd=str(BASE_DIR))


def clear_knot_data():
    if KNOT_DATA.exists():
        shutil.rmtree(KNOT_DATA)
    KNOT_DATA.mkdir()


def run_all():
    subdirs = sorted(
        d for d in glob.glob(str(LAMMPS_DAT_ROOT / "3_lammps_normalized_coordinates_*"))
        if os.path.isdir(d)
    )

    if not subdirs:
        print(f"No subdirectories found under {LAMMPS_DAT_ROOT}")
        return

    for subdir in subdirs:
        print(f"\n=== Processing directory: {subdir} ===")

        # Extract e.g. 51 from folder name
        x_value = os.path.basename(subdir).split("_")[-1]

        # Run LAMMPS for this x
        bases = run_lammps_for_folder(subdir)

        # Concatenate FINAL_* for this x
        concat_results_for_folder(x_value, bases)

        # Clear KNOT_data for next x
        clear_knot_data()
        print(f"  Cleared KNOT_data for next iteration.\n")


if __name__ == "__main__":
    run_all()

#!/usr/bin/env python3

# Call using:
# python -m workflows.3_to_4_variable_temperatures_run

import os
import glob
import subprocess
from pathlib import Path
import shutil

NUM_RUNS = 1000

temperatures = [0.3, 0.5, 0.7]

BASE_DIR = Path(__file__).resolve().parent.parent

# Input folders containing the .dat files for each polymer length
INPUT_ROOT = BASE_DIR / "3_lammps_normalized_coordinates"

# Final XYZ output (per length)
XYZ_ROOT = BASE_DIR / "4_xyz_files_variable_temperature"

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
    
    for T in temperatures:
        out_name = f"4_xyz_files_temperature_{T}"
        out_path = os.path.join(XYZ_ROOT, out_name)
        os.makedirs(out_path, exist_ok=False) # Make the out-subfolder. If it already exists something is wrong as we tried to delete the out folder

        print(f"\n=== Processing temperature {T} ===")

        # LAMMPS output folder is ALWAYS the same
        lmp_out_dir = BASE_DIR / "lammps_simulation" / "KNOT_data"

        # Clean KNOT_data before each length
        if lmp_out_dir.exists():
            shutil.rmtree(lmp_out_dir)
        lmp_out_dir.mkdir(parents=True)

        # Find all input .dat files for this length
        input_dats = sorted(glob.glob(str(INPUT_ROOT / "*.dat")))

        assert input_dats, "No .dat files found, skipping."

        # -------- Run LAMMPS on all input files --------
        for dat_path in input_dats:
            simname = Path(dat_path).stem
            print(f"  Running LAMMPS for: {simname}")

            for i in range(NUM_RUNS):
                final_file = lmp_out_dir / f"FINAL_{simname}.{i}.dat"

                if final_file.exists():
                    continue  # already done, skip

                seed = 10000 + i
                print(str(Path(out_path).relative_to(BASE_DIR)))
                cmd = [
                    str(LMP_BINARY),
                    "-in", str(LAMMPS_INPUT),
                    "-var", "simdir", str(INPUT_ROOT.relative_to(BASE_DIR)),
                    "-var", "simname", simname,
                    "-var", "runid", str(i),
                    "-var", "seed", str(seed),
                    "-var", "T", str(T),
                    ]
                subprocess.run(cmd, check=True, cwd=BASE_DIR)
                print("ho")

        print("  LAMMPS runs finished; converting FINAL_*.dat → XYZ_*.dat.nos")

        # -------- Convert FINAL_*.dat into XYZ_*.dat.nos --------
        subprocess.run(
            [
                "python",
                str(CONVERTER),
                "--infolder", str(lmp_out_dir),
                "--outfolder", str(out_path)
            ],
            check=True,
        )



if __name__ == "__main__":
    run_all()

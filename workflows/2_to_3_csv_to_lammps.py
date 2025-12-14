
# Call using:
# python -m workflows.2_to_3_csv_to_lammps

import os
import subprocess


def main():
    input_root = "2_normalized_coordinates"
    output_root = "3_lammps_normalized_coordinates"

    # Delete and recreate output root
    if os.path.exists(output_root):
        import shutil
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)


    # Process CSVs
    for fname in os.listdir(input_root):
        if fname.endswith(".csv"):
            in_file = os.path.join(input_root, fname)
            base = os.path.splitext(fname)[0]
            out_file = os.path.join(output_root, f"{base}.dat")

            # Run csv2lammps.py
            subprocess.run([
                "python", "csv2lammps.py", in_file,
                "-o", out_file,
                "--atom-types", "4"
            ], check=True)


if __name__ == "__main__":
    main()

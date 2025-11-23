import os
import subprocess


def main():
    input_root = "2_normalized_coordinates_variable_length"
    output_root = "3_lammps_normalized_coordinates_variable_length"

    # Delete and recreate output root
    if os.path.exists(output_root):
        import shutil
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)

    # Iterate over subfolders like 2_normalized_coordinates_51, etc.
    for sub in os.listdir(input_root):
        sub_path = os.path.join(input_root, sub)
        if not os.path.isdir(sub_path):
            continue

        # Create matching output folder name
        out_sub = sub.replace("2_normalized_coordinates_", "3_lammps_normalized_coordinates_")
        out_path = os.path.join(output_root, out_sub)
        os.makedirs(out_path, exist_ok=True)

        # Process CSVs
        for fname in os.listdir(sub_path):
            if fname.endswith(".csv"):
                in_file = os.path.join(sub_path, fname)
                base = os.path.splitext(fname)[0]
                out_file = os.path.join(out_path, f"{base}.dat")

                # Run csv2lammps.py
                subprocess.run([
                    "python", "csv2lammps.py", in_file,
                    "-o", out_file,
                    "--atom-types", "4"
                ], check=True)


if __name__ == "__main__":
    main()

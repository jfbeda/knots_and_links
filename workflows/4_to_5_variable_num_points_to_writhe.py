import os
import subprocess


def main():
    input_root = "4_xyz_files_variable_length"
    output_root = "5_writhe_files_variable_length"

    # Delete and recreate output root
    if os.path.exists(output_root):
        import shutil
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)

    # Loop through subfolders like 4_xyz_files_51, 75, etc.
    for sub in os.listdir(input_root):
        sub_path = os.path.join(input_root, sub)
        if not os.path.isdir(sub_path):
            continue

        # Determine number of points from folder name
        # Expecting folder names like: 4_xyz_files_51
        try:
            num_points = int(sub.split("_")[-1])
        except ValueError:
            print(f"Skipping folder {sub}: could not parse number of points")
            continue

        # Create corresponding output folder
        out_sub = sub.replace("4_xyz_files_", "5_writhe_files_")
        out_path = os.path.join(output_root, out_sub)
        os.makedirs(out_path, exist_ok=True)

        # Run datnos2writhe.py
        subprocess.run([
            "python", "datnos2writhe.py",
            "--in-dir", sub_path,
            "--out-dir", out_path,
            "--subknots", "2",
            "--points", str(num_points),
            "--pattern", "*.dat.nos"
        ], check=True)


if __name__ == "__main__":
    main()

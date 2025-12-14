# Call using:
# python -m workflows.4_to_4_interpolate_xyz

# Given an input folder that contains a set of 0.2.1.dat.nos, 2.2.1.dat.nos, etc with a fixed number of points per subknot (say 60). Then it returns a folder "4_xyz_files_variable_length" that has subfolders 4_xyz_files_60, 4_xyz_files_120, 4_xyz_files_180 or what not, that contains the same knot data, but interppolated with a different number of points. 

# Not only does this code interpolate the knot, but it also scales all coordinates by the same factor such that the average interparticle spacing in each knot remains the same as before (probably an average length of about 1)



import os
import subprocess


def main():
    input_root = "4_xyz_files"
    output_root = "4_xyz_files_variable_length"

    # Delete and recreate output root
    if os.path.exists(output_root):
        import shutil
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)

    points_per_subknot_list = [51, 60, 65, 70]
    
# Call using:
# python -m workflows.4_to_4_interpolate_xyz

# Given an input folder that contains a set of 0.2.1.dat.nos, 2.2.1.dat.nos, etc with a fixed number of points per subknot (say 60). Then it returns a folder "4_xyz_files_variable_length" that has subfolders 4_xyz_files_60, 4_xyz_files_120, 4_xyz_files_180 or what not, that contains the same knot data, but interppolated with a different number of points. 

# Not only does this code interpolate the knot, but it also scales all coordinates by the same factor such that the average interparticle spacing in each knot remains the same as before (probably an average length of about 1)


import os
import subprocess
import interpolate
from pathlib import Path
from dataprocessor import *
from interpolate import *


def main():
    original_num_points = 51 # HACK I Really shouldn't hard code this
    num_subknots_per_link = 2

    input_root = f"4_xyz_files_{original_num_points}"
    output_root = "4_xyz_files_variable_length"

    # Delete and recreate output root
    if os.path.exists(output_root):
        import shutil
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)

    points_per_subknot_list = [51, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125]

    for new_num_points in points_per_subknot_list:
        output_folder = os.path.join(output_root, f"4_xyz_files_{new_num_points}")
        os.makedirs(output_folder, exist_ok=False) #If the output folder already exists something is wrong as we should have deleted it  

        for file in os.listdir(input_root):
            if file[-8:] == ".dat.nos":
                print(f"Interpolating {file} to {new_num_points} points")
                out_file = os.path.join(output_folder, file)

                original_link_coordinates = load_links(os.path.join(input_root, file), num_subknots_per_link, original_num_points, 3)
                num_links = original_link_coordinates.shape[0]
                new_link_coordinates = np.zeros(shape = (num_links, new_num_points * num_subknots_per_link, 3))
                for i in range(num_links):
                    # Resample the first subknot
                    new_link_coordinates[i,:new_num_points] = smooth_and_resample_3d(original_link_coordinates[i, :original_num_points], new_num_points, 1)

                    # Resample the second subknot
                    new_link_coordinates[i,new_num_points:] = smooth_and_resample_3d(original_link_coordinates[i, original_num_points:], new_num_points, 1)

                # Reshape the interpolated data to 
                new_datnos_file = new_link_coordinates.reshape(-1, 3)
                # assert np.allclose(new_datnos_file.reshape(-1, 2*new_num_points, 3),new_link_coordinates), "ah"

                with open(out_file, "w") as out:
                    for j in range(len(new_datnos_file)):
                        out.write(f"{new_datnos_file[j][0]:.16e} {new_datnos_file[j][1]:.16e} {new_datnos_file[j][2]:.16e}\n")
    

if __name__ == "__main__":
    main()
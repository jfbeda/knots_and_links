
# python -m workflows.1_to_2_variable_num_points_extended

import os
from knots_and_links import SetofLinks


def main():
    input_folder = "1_main_coordinates"
    base_output = "2_normalized_coordinates_variable_length"

    if os.path.exists(base_output):
        import shutil
        shutil.rmtree(base_output)
    os.makedirs(base_output, exist_ok=True)

    # target_lengths = [51, 75, 100, 125, 150]
    # target_lengths = [60, 70, 80, 90, 110] 
    # target_lengths = [55, 65, 85, 95, 105]
    target_lengths = [55]

    for n in target_lengths:
        output_folder = os.path.join(base_output, f"2_normalized_coordinates_{n}")
        os.makedirs(output_folder, exist_ok=True)

        mylinks = SetofLinks.from_folder_name(input_folder)
        mylinks.normalize(desired_num_points_per_subknot=n)
        mylinks.to_folder_name(output_folder)


if __name__ == "__main__":
    main()

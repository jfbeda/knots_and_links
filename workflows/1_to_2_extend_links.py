
# python -m workflows.1_to_2_extend_links

import os
from knots_and_links import SetofLinks


def main():
    input_folder = "1_main_coordinates"
    output_folder = "2_normalized_coordinates"

    if os.path.exists(output_folder):
        import shutil
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    target_length = 60

    mylinks = SetofLinks.from_folder_name(input_folder)
    mylinks.normalize(desired_num_points_per_subknot=target_length)
    mylinks.to_folder_name(output_folder)


if __name__ == "__main__":
    main()

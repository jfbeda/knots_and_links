
# python -m workflows.2_to_2_reflect_link

import os
from knots_and_links import SetofLinks, Knot, Link


def main():
    input_folder = "2_normalized_coordinates"
    output_folder = "2_normalized_coordinates_reflected"

    if os.path.exists(output_folder):
        import shutil
        shutil.rmtree(output_folder)

    os.makedirs(output_folder, exist_ok=False)
    
    old_links = SetofLinks.from_folder_name(input_folder)
    new_links_list = []
    for old_link in old_links.links:
        knot_dictionary = dict()
        for knotname in old_link.subknots.keys():
            old_knot = old_link.subknots[knotname]
            knot_dictionary[knotname] = Knot(name = f"{old_knot.name}" + " reflected", coords = -old_knot.coords)
        new_link = Link(subknots = knot_dictionary, name = f"{old_link.name}" + " reflected")
        new_links_list.append(new_link)
    
    new_links = SetofLinks(new_links_list)
    new_links.to_folder_name(output_folder)

if __name__ == "__main__":
    main()

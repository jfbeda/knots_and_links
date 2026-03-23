# #!/usr/bin/env python3

# Call using:

# python find_split_knots.py

# REMEMBER to adjust the root folder

import os
import numpy as np

# -----------------------------
# USER PARAMETERS
# -----------------------------

# ROOT_FOLDER = "0_good_data/variable_length_1000_runs/4_xyz_files_variable_length"
ROOT_FOLDER = "0_good_data/variable_temperature_1000_runs_(60_points_per_subknot)/4_xyz_files_variable_temperature"

NUM_SUBKNOTS = 2                 # number of subknots per link
POINTS_PER_SUBKNOT = None        # inferred per file
NUM_DIMENSIONS = 3
THRESHOLD_SQ = 2.0               # squared distance threshold

# -----------------------------
# Helper: load_links
# -----------------------------
def load_links(file_path, num_subknots, points_per_subknot, num_dimensions):
    """
    Loads a .dat.nos file into shape:
      (num_links, num_subknots * points_per_subknot, num_dimensions)

    Assumes raw XYZ per line.
    """
    data = np.loadtxt(file_path)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != num_dimensions:
        raise ValueError(
            f"{file_path}: expected {num_dimensions} columns, got {data.shape[1]}"
        )

    total_points = data.shape[0]

    if points_per_subknot is None:
        # infer number of links from folder convention: 1000 knots
        NUM_LINKS = 1000
        denom = NUM_LINKS * num_subknots

        if total_points % denom != 0:
            raise ValueError(
                f"{file_path}: total_points={total_points} not divisible by "
                f"NUM_LINKS*num_subknots={denom}"
            )

        points_per_subknot = total_points // denom

    pts_per_link = num_subknots * points_per_subknot
    num_links = total_points // pts_per_link

    if total_points != num_links * pts_per_link:
        raise ValueError(f"{file_path}: inconsistent point count")

    return data.reshape(num_links, pts_per_link, num_dimensions)

# -----------------------------
# Subknot-aware split detector
# -----------------------------
def split_knot_indices(link_coordinates, threshold_sq, points_per_subknot, num_subknots):
    """
    Returns a list of knot indices whose *subknots* are split.
    """
    num_links = link_coordinates.shape[0]
    bad_knots = []

    for i in range(num_links):
        coords = link_coordinates[i]   # (pts_per_link, 3)

        for s in range(num_subknots):
            start = s * points_per_subknot
            end   = (s + 1) * points_per_subknot
            subknot = coords[start:end]

            # consecutive jumps inside subknot only
            diffs = np.diff(subknot, axis=0)
            dist_sq = np.sum(diffs**2, axis=1)

            if np.any(dist_sq > threshold_sq):
                bad_knots.append(i)
                break  # no need to check other subknots

    return bad_knots

# -----------------------------
# Main scan
# -----------------------------
def main():
    bad_files = {}

    for subdir in sorted(os.listdir(ROOT_FOLDER)):
        subdir_path = os.path.join(ROOT_FOLDER, subdir)
        if not os.path.isdir(subdir_path):
            continue

        for fname in sorted(os.listdir(subdir_path)):
            if not fname.endswith(".dat.nos"):
                continue

            fpath = os.path.join(subdir_path, fname)

            try:
                links = load_links(
                    fpath,
                    NUM_SUBKNOTS,
                    POINTS_PER_SUBKNOT,
                    NUM_DIMENSIONS
                )

                points_per_subknot = links.shape[1] // NUM_SUBKNOTS

                bad_indices = split_knot_indices(
                    links,
                    THRESHOLD_SQ,
                    points_per_subknot,
                    NUM_SUBKNOTS
                )

                if bad_indices:
                    bad_files[fpath] = bad_indices

            except Exception as e:
                print(f"ERROR reading {fpath}: {e}")
                bad_files[fpath] = ["READ_ERROR"]

    # -----------------------------
    # Report
    # -----------------------------
    if bad_files:
        print("\nFiles containing split subknots:\n")
        for fpath, indices in bad_files.items():
            print(f"{fpath}")
            print(f"  Split knot indices: {indices}\n")
    else:
        print("\nNo split subknots detected.")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import os
import numpy as np

# -----------------------------
# USER PARAMETERS
# -----------------------------

ROOT_FOLDER = "0_good_data/variable_length_1000_runs/4_xyz_files_variable_length"
NUM_SUBKNOTS = 2                 # adjust if needed
POINTS_PER_SUBKNOT = None        # None if variable per folder
NUM_DIMENSIONS = 3
THRESHOLD_SQ = 2.0

# -----------------------------
# Helper: load_links
# -----------------------------
def load_links(file_path, num_subknots, points_per_subknot, num_dimensions):
    """
    Loads a .dat.nos file into shape:
      (num_links, num_subknots*points_per_subknot, num_dimensions)

    Assumes raw XYZ per line.
    """
    data = np.loadtxt(file_path)
    total_points = data.shape[0]

    if points_per_subknot is None:
        points_per_subknot = total_points // num_subknots

    pts_per_link = num_subknots * points_per_subknot
    num_links = total_points // pts_per_link

    return data.reshape(num_links, pts_per_link, num_dimensions)

# -----------------------------
# Split-knot detector
# -----------------------------
def has_split_knot(link_coordinates, threshold_sq):
    """
    Returns True if any consecutive point jump exceeds threshold_sq.
    """
    # diff shape: (num_links, pts-1, dim)
    diffs = np.diff(link_coordinates, axis=1)
    dist_sq = np.sum(diffs**2, axis=2)
    return np.any(dist_sq > threshold_sq)

# -----------------------------
# Main scan
# -----------------------------
def main():
    bad_files = []

    for subdir in sorted(os.listdir(ROOT_FOLDER)):
        subdir_path = os.path.join(ROOT_FOLDER, subdir)
        if not os.path.isdir(subdir_path):
            continue

        for fname in os.listdir(subdir_path):
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

                if has_split_knot(links, THRESHOLD_SQ):
                    bad_files.append(fpath)

            except Exception as e:
                print(f"ERROR reading {fpath}: {e}")
                bad_files.append(fpath)

    # -----------------------------
    # Report
    # -----------------------------
    if bad_files:
        print("\nFiles containing split knots:\n")
        for f in bad_files:
            print(f)
    else:
        print("\nNo split knots detected.")

if __name__ == "__main__":
    main()

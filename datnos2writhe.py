#!/usr/bin/env python3
"""
datnos2writhe.py
Convert .dat.nos files (containing many configurations) into writhe matrices.

Default behavior matches your original script:
- Reads all *.dat.nos files from --in-dir (default: ml_coords)
- Writes to --out-dir (default: ml_writhe_coords)
- Assumes 2 subknots, 51 points each, 3 dimensions
- Output filename transformation:
    "XYZ" -> "WRITHE", drop ".nos" and drop "_small"
    e.g., XYZ_6.2.1.dat.nos  -> WRITHE_6.2.1.dat

Usage examples:
  python datnos2writhe.py
  python datnos2writhe.py --in-dir KNOT_data --out-dir writhe_out
  python datnos2writhe.py --subknots 2 --points 51 --dims 3
  python datnos2writhe.py --pattern "*.dat.nos" --no-rename
"""

import argparse
import os
from pathlib import Path
import numpy as np

# Your modules (as in the original script)
from dataprocessor import *   # noqa: F401,F403
from writhe import *          # noqa: F401,F403

def transform_name(original_name: str, no_rename: bool = False) -> str:
    """
    Replicate your original renaming:
      - replace "XYZ" with "WRITHE"
      - remove ".nos"
      - remove "_small"
      Keep the rest of the name as-is.
    If no_rename is True, use "<stem>.writhe.txt".
    """
    stem = original_name
    if no_rename:
        base = Path(original_name).stem  # drop last suffix only
        return f"{base}.writhe.txt"

    new_name = stem.replace("XYZ", "WRITHE").replace(".nos", "").replace("_small", "")
    return new_name

def process_file(filepath: Path,
                 out_dir: Path,
                 num_subknots: int,
                 points_per_subknot: int,
                 num_dimensions: int,
                 no_rename: bool):
    # Import functions from your modules
    # load_links(...) and compute_writhe_matrix(...)
    link_coordinates = load_links(str(filepath), num_subknots, points_per_subknot, num_dimensions)
    print(f"Loaded {filepath.name} -> shape {link_coordinates.shape}")

    num_links = link_coordinates.shape[0]
    total_points = num_subknots * points_per_subknot

    writhe_matrices = np.zeros((num_links, total_points, total_points), dtype=float)

    # Build block writhe matrix per configuration
    for i in range(num_links):
        first_subknot  = link_coordinates[i][:points_per_subknot]
        second_subknot = link_coordinates[i][points_per_subknot:]

        # Blocks:
        # [ A  B^T ]
        # [ B   C  ]
        A = compute_writhe_matrix(first_subknot,  first_subknot)
        B = compute_writhe_matrix(second_subknot, first_subknot)   # lower-left
        C = compute_writhe_matrix(second_subknot, second_subknot)

        W = writhe_matrices[i]
        W[:points_per_subknot, :points_per_subknot] = A
        W[points_per_subknot:, :points_per_subknot] = B
        W[:points_per_subknot, points_per_subknot:] = B.T
        W[points_per_subknot:, points_per_subknot:] = C

    out_name = transform_name(filepath.name, no_rename=no_rename)
    savepath = out_dir / out_name

    # Flatten each matrix to rows of length total_points (as in your original)
    np.savetxt(savepath, writhe_matrices.reshape(-1, writhe_matrices.shape[-1]))
    print(f"Saved writhe data to {savepath}")

def main():
    print("hi")
    p = argparse.ArgumentParser(description="Compute writhe matrices from .dat.nos files.")
    p.add_argument("--in-dir", default="ml_coords",
                   help="Directory to read .dat.nos files from (default: %(default)s)")
    p.add_argument("--out-dir", default="ml_writhe_coords",
                   help="Directory to write output files to (default: %(default)s)")
    p.add_argument("--pattern", default="*.dat.nos",
                   help="Glob pattern within --in-dir to select files (default: %(default)s)")
    p.add_argument("--subknots", type=int, default=2,
                   help="Number of subknots (default: %(default)s)")
    p.add_argument("--points", type=int, default=51,
                   help="Points per subknot (default: %(default)s)")
    p.add_argument("--dims", type=int, default=3,
                   help="Number of spatial dimensions (default: %(default)s)")
    p.add_argument("--no-rename", action="store_true",
                   help="Do not apply XYZ->WRITHE/.nos/_small transformation; use '<stem>.writhe.txt' instead.")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.pattern))
    if not files:
        print(f"No files match: {in_dir}/{args.pattern}")
        return

    for fp in files:
        try:
            process_file(
                filepath=fp,
                out_dir=out_dir,
                num_subknots=args.subknots,
                points_per_subknot=args.points,
                num_dimensions=args.dims,
                no_rename=args.no_rename,
            )
        except Exception as e:
            print(f"[ERROR] {fp.name}: {e}")

    print("Done.")

if __name__ == "__main__":
    main()

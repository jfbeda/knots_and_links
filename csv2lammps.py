#!/usr/bin/env python3
"""
csv2lammpslinks.py — convert KnotPlot-generated CSVs (single knot or multi-component links)
into a LAMMPS "data" file suitable for simulations.

It supports files that look like:
    Component 1 of 2:
    x y z
    x y z
    ...
    Component 2 of 2:
    x y z
    ...

and also plain CSV/space-delimited XYZ with no headers (single component).

Output "Atoms" section format:
    atom_id  molecule_id  atom_type  x  y  z

Bonds connect successive points within each component and (by default) close the ring.
Angles are defined for successive triplets (also closed for rings).

Usage:
    python csv2lammpslinks.py input.csv -o output.dat [--no-ring] [--mass 1.0]
                                       [--atom-type 1] [--atom-types 1]
                                       [--box xmin xmax ymin ymax zmin zmax]
                                       [--padding 5.0] [--precision 6]
                                       [--title "LAMMPS data file: ..."]

Notes:
- By default we assume each component is a closed ring (common for knots/links).
- molecule_id is 1..K for K components.
- If you want different atom types per component, you can use --component-types
  to supply a comma-separated list (e.g., "1,2,1"). If not given, all atoms use --atom-type.
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import List, Tuple

def parse_knotplot_csv(path: Path) -> List[List[Tuple[float, float, float]]]:
    """
    Parse a KnotPlot-style CSV/XYZ that may have multiple components.
    Returns a list of components; each component is a list of (x,y,z) floats.
    """
    components: List[List[Tuple[float, float, float]]] = []
    current: List[Tuple[float, float, float]] = []

    with path.open('r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                # empty line -> skip (but don't flush; KnotPlot usually uses headers)
                continue
            # Component headers like "Component 1 of 2:"
            if line.lower().startswith("component ") and line.endswith(":"):
                if current:
                    components.append(current)
                    current = []
                continue
            # Comments
            if line.startswith("#") or line.startswith("//"):
                continue
            # Try to parse 3 floats separated by spaces or commas
            # Replace commas with spaces and split
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                # Not a data line -> ignore
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                current.append((x, y, z))
            except ValueError:
                # Ignore lines that aren't coordinates
                continue

    if current:
        components.append(current)

    # If the file had no component headers, treat whole file as one component
    if not components:
        raise ValueError("No coordinate data found in file.")
    return components

def counts_for_components(sizes: List[int], ring: bool) -> Tuple[int, int, int]:
    num_atoms = sum(sizes)
    num_bonds = sum(n if ring else max(0, n - 1) for n in sizes)
    def angles_for(n: int) -> int:
        if n < 2:
            return 0
        if ring:
            return n
        return max(0, n - 2)
    num_angles = sum(angles_for(n) for n in sizes)
    return num_atoms, num_bonds, num_angles

def compute_box(components: List[List[Tuple[float,float,float]]], padding: float) -> Tuple[float,float,float,float,float,float]:
    all_pts = [p for comp in components for p in comp]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    zs = [p[2] for p in all_pts]
    xmin, xmax = min(xs) - padding, max(xs) + padding
    ymin, ymax = min(ys) - padding, max(ys) + padding
    zmin, zmax = min(zs) - padding, max(zs) + padding
    return xmin, xmax, ymin, ymax, zmin, zmax

def fmt_float(x: float, precision: int) -> str:
    return f"{x:.{precision}f}".rstrip('0').rstrip('.') if precision is not None else f"{x}"

def build_lammps_data(components: List[List[Tuple[float,float,float]]],
                      ring: bool,
                      atom_types: int,
                      mass: float,
                      default_atom_type: int,
                      component_types: List[int] | None,
                      box: Tuple[float,float,float,float,float,float] | None,
                      precision: int,
                      title: str) -> str:
    sizes = [len(c) for c in components]
    N_atoms, N_bonds, N_angles = counts_for_components(sizes, ring)

    if component_types is not None and len(component_types) != len(components):
        raise ValueError(f"--component-types length {len(component_types)} doesn't match number of components {len(components)}.")

    if box is None:
        xmin, xmax, ymin, ymax, zmin, zmax = compute_box(components, padding=5.0)
    else:
        xmin, xmax, ymin, ymax, zmin, zmax = box

    lines: List[str] = []
    lines.append(f"{title}\n")
    lines.append(f"{N_atoms} atoms")
    lines.append(f"{N_bonds} bonds")
    lines.append(f"{N_angles} angles\n")
    lines.append(f"{atom_types} atom types")
    lines.append("1 bond types")
    lines.append("1 angle types\n")
    lines.append(f"{fmt_float(xmin, precision)} {fmt_float(xmax, precision)} xlo xhi ")
    lines.append(f"{fmt_float(ymin, precision)} {fmt_float(ymax, precision)} ylo yhi ")
    lines.append(f"{fmt_float(zmin, precision)} {fmt_float(zmax, precision)} zlo zhi\n")
    lines.append("Masses \n")
    for t in range(1, atom_types + 1):
        lines.append(f" {t} {fmt_float(mass, precision)}")
    lines.append("\nAtoms \n")

    # Atoms
    atom_id = 1
    atom_records: List[str] = []
    for comp_idx, comp in enumerate(components, start=1):
        atype = component_types[comp_idx - 1] if component_types is not None else default_atom_type
        for (x, y, z) in comp:
            atom_records.append(
                f"{atom_id} {comp_idx} {atype} {fmt_float(x, precision)} {fmt_float(y, precision)} {fmt_float(z, precision)}"
            )
            atom_id += 1
    lines.extend(atom_records)
    lines.append("\nBonds\n")

    # Bonds
    bond_id = 1
    atom_base = 1
    bond_records: List[str] = []
    for n in sizes:
        if n == 0:
            continue
        # Connect i -> i+1
        for i in range(n - 1):
            a = atom_base + i
            b = atom_base + i + 1
            bond_records.append(f"{bond_id} 1 {a} {b}")
            bond_id += 1
        if ring and n >= 2:
            # close the ring
            a = atom_base + n - 1
            b = atom_base
            bond_records.append(f"{bond_id} 1 {a} {b}")
            bond_id += 1
        atom_base += n
    lines.extend(bond_records)
    lines.append("\nAngles\n")

    # Angles
    angle_id = 1
    atom_base = 1
    angle_records: List[str] = []
    for n in sizes:
        if n >= 3:
            # i,i+1,i+2 along chain
            last = n if ring else n - 2
            for i in range(last):
                a = atom_base + (i % n)
                b = atom_base + ((i + 1) % n)
                c = atom_base + ((i + 2) % n)
                angle_records.append(f"{angle_id} 1 {a} {b} {c}")
                angle_id += 1
        elif n == 2:
            # no well-defined angles in a 2-atom component
            pass
        # advance
        atom_base += n
    lines.extend(angle_records)

    return "\n".join(lines) + "\n"

def parse_box_arg(vals: List[str]) -> Tuple[float,float,float,float,float,float]:
    if len(vals) != 6:
        raise argparse.ArgumentTypeError("--box expects 6 numbers: xmin xmax ymin ymax zmin zmax")
    x = list(map(float, vals))
    return x[0], x[1], x[2], x[3], x[4], x[5]

def main():
    p = argparse.ArgumentParser(description="Convert KnotPlot CSV/link to LAMMPS data file.")
    p.add_argument("input", type=Path, help="Input CSV/XYZ file from KnotPlot (can have multiple components).")
    p.add_argument("-o", "--output", type=Path,
               help="Output .dat file path. If not given, defaults to input name with .dat")
    p.add_argument("--out-dir", type=Path, default=None,
               help="Directory to place the output file in (used only when --output is not given).")
    p.add_argument("--no-ring", dest="ring", action="store_false", help="Do NOT close bonds/angles as a ring.")
    p.add_argument("--ring", dest="ring", action="store_true", help="Close bonds/angles as a ring (default).")
    p.set_defaults(ring=True)
    p.add_argument("--mass", type=float, default=1.0, help="Mass value for all atom types in 'Masses' section.")
    p.add_argument("--atom-type", type=int, default=1, help="Default atom type to use for atoms (if --component-types not provided).")
    p.add_argument("--component-types", type=str, default=None,
                   help="Comma-separated list of atom types per component, e.g. '1,2,1'. Overrides --atom-type.")
    p.add_argument("--atom-types", type=int, default=1, help="Number of atom types (for 'Masses' section).")
    p.add_argument("--box", nargs=6, metavar=("XMIN","XMAX","YMIN","YMAX","ZMIN","ZMAX"),
                   help="Explicit simulation box bounds. If omitted, computed from data + padding.")
    p.add_argument("--padding", type=float, default=5.0, help="Padding added to auto-computed bounds (each side).")
    p.add_argument("--precision", type=int, default=6, help="Decimal places for floats in output.")
    p.add_argument("--title", type=str, default="LAMMPS data file: KnotPlot link",
                   help="Title line at top of the file.")
    args = p.parse_args()

    components = parse_knotplot_csv(args.input)
    # If explicit component types supplied
    comp_types = None
    if args.component_types:
        comp_types = [int(s.strip()) for s in args.component_types.split(",")]

    # Determine box
    box = None
    if args.box is not None:
        xmin,xmax,ymin,ymax,zmin,zmax = map(float, args.box)
        box = (xmin,xmax,ymin,ymax,zmin,zmax)
    else:
        box = compute_box(components, args.padding)

    data = build_lammps_data(
        components=components,
        ring=args.ring,
        atom_types=args.atom_types,
        mass=args.mass,
        default_atom_type=args.atom_type,
        component_types=comp_types,
        box=box,
        precision=args.precision,
        title=args.title,
    )

    #Decide where to write the file
    if args.output:
        out_path = args.output
    elif args.out_dir:
        out_path = args.out_dir / (args.input.stem + ".dat")
    else:
        out_path = args.input.with_suffix(".dat")

    # Ensure the parent directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(data, encoding="utf-8")
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()

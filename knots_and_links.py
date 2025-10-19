import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import scipy as sp
import copy
from pathlib import Path
import re

class Knot:
    """
    Represents a single knot (component) of a link.
    Stores an array of coordinates and an optional name.
    """

    def __init__(self, name=None, coords=None):
        self.name = name or "Unnamed Knot"

        if coords is None:
            self.coords = np.empty((0, 3))
        else:
            arr = np.array(coords, dtype=float)
            if arr.ndim != 2:
                raise ValueError("coords must be a 2D array")
            self.coords = arr

    def __repr__(self):
        return f"<Knot {self.name!r}: {len(self.coords)} points>"

    def to_string(self):
        """Return this knot’s data as a formatted string block for writing to CSV."""
        lines = [f"{' '.join(map(str, row))}" for row in self.coords]
        return f"{self.name}:\n" + "\n".join(lines)
    
    def orientate(self, return_matrix = False):
        """Moves the knot such that the centre of mass is at the origin,
        and the point furthest from the centre of mass lies on the positive z-axis"""

        # center coordinates at the origin
        self.coords = self.coords - self.center_of_mass
        
        # Compute the rotation matrix that takes the extremum of the knot to the z axis
        rotation_matrix = rotation_matrix_to_z_axis(self.extremum)

        # Apply the rotation matrix to each coordinate
        self.coords = self.coords @ rotation_matrix.T

        assert self.is_orientated, f"Error! We tried to orientate the knot, but it somehow isn't orientated? Center of mass is at {np.linalg.norm(self.center_of_mass)} and extremum at {(self.extremum/np.linalg.norm(self.extremum))}."

        if return_matrix:
            return rotation_matrix
    # def add(self, knot2):
    #     """Adds knot2 to self using the connected sum. Orientates both knots, 
    #     moves/rotates them both such that extremum point lies at the origin and they point in
    #     opposite directions. Concatenate the lists to make a new knot."""

    #     for knot in [self, knot2]:
    #         if not knot.is_orientated:
    #             knot.orientate()
            

    #     # Move each knot below the z axis, pointing upwards      
    #     for knot in [self, knot2]:
    #         knot.coords = knot.coords - knot.extremum
            
    #     # Rotate knot2 by 180 degrees so that extremum points downwards
    #     knot2.coords = knot2.coords @ np.array([[1,0,0],[0,-1,0],[0,0,-1]]).T

    #     new_name = self.name + " + " +  knot2.name

    #     return Knot(name = new_name, coords = splice_knots(self, knot2))

    @property
    def is_orientated(self):
        if np.isclose(np.linalg.norm(self.center_of_mass),0) and np.isclose(np.linalg.norm(self.extremum/np.linalg.norm(self.extremum)-np.array([0,0,1])),0):
            return True
        else:
            return False

    @property
    def num_points(self):
        """Returns the number of points defining the knot"""
        return len(self.coords)
    
    @property
    def segment_length(self):
        """Returns the average separation between points in the knot"""
        return self.arclength/(self.num_points)

    @property
    def arclength(self):
        """Return the total arclength of the knot."""
        c = self.coords
        diffs = np.diff(np.vstack([c, c[0]]), axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        return np.sum(segment_lengths)
    
    @property
    def center_of_mass(self):
        """Renturn centre of mass of the knot."""
        return np.mean(self.coords, axis=0)

    @property
    def extremum_index(self):
        """Returns the index of the coordinate furthest from the center of mass"""
        centred_coordinates = self.coords - self.center_of_mass
        radius = np.linalg.norm(centred_coordinates, axis = 1)
        extremum_index = np.argmax(radius)
        return extremum_index
    
    @property
    def extremum(self):
        """Returns the coordinates of the point furthest from the center of mass"""
        return self.coords[self.extremum_index]

# def rotation_matrix_to_z_axis3(v):
#     """Returns a rotation matrix R that when acting on v gives the z hat direction. I.e. R @ v = z"""

#     v = np.array(v, dtype=float)
#     v /= np.linalg.norm(v)
#     z = np.array([0.0, 0.0, 1.0])
#     rot = Rotation.align_vectors([z], [v])[0]
#     return rot.as_matrix()

# def rotation_matrix_to_z_axis2(v):
#     """Returns a rotation matrix R that when acting on v gives the z hat direction. I.e. R @ v = z"""

#     v_hat = v/np.linalg.norm(v)
#     print(v_hat)
#     z_hat = np.array([0.0, 0.0, 1.0])
#     v_hat_cross_z_hat = np.cross(v_hat, z_hat)
#     print(v_hat_cross_z_hat)
#     norm = np.linalg.norm(v_hat_cross_z_hat)
#     print(norm)
#     if np.isclose(norm,0.):
#         return np.array([[1,0,0],[0,1,0],[0,0,1]])
    
#     else:
#         n_hat = v_hat_cross_z_hat/norm
#         theta = np.arcsin(norm)
#         print(theta)
#         print(np.arccos(np.dot(v_hat,z_hat)))
#         theta = np.arccos(np.dot(v_hat,z_hat))
#         rotation_matrix = Rotation.from_rotvec(theta * n_hat).as_matrix()

#     print(rotation_matrix @ rotation_matrix.T)
#     print(rotation_matrix @ v_hat)

#     return rotation_matrix

def rotation_matrix_to_z_axis(v):
    """
    Returns R such that R @ v = [0, 0, 1].
    Uses an orthonormal-basis construction (no angles), numerically stable as opposed to computing arcsin or arcos.
    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero vector has no direction.")
    a = v / n  # a = v_hat

    # Choose a helper vector not parallel to a
    # Use the component with smallest magnitude to reduce risk of near-parallel choice.
    if abs(a[2]) < 0.7:
        k = np.array([0.0, 0.0, 1.0])  # ẑ is safely not parallel
    else:
        k = np.array([1.0, 0.0, 0.0])  # fall back to x̂

    # b ⟂ a
    b = np.cross(k, a)
    b_norm = np.linalg.norm(b)
    b /= b_norm

    # c completes right-handed frame
    c = np.cross(a, b)
    assert np.isclose(np.linalg.norm(c)-1,0), "Third basis vector is not properly normalized! (This is very surprising). This seems to imply a and b are not indeed perpendicular and normalized."

    # Rows [b; c; a] map a -> ẑ and form a proper rotation (det = +1)
    R = np.vstack([b, c, a])

    assert np.isclose(np.linalg.norm(R @ R.T - np.array([[1,0,0],[0,1,0],[0,0,1]])),0), "Somehow R isn't actually a rotation matrix! (It's not orthogonal)"

    return R


class Link:
    """
    Represents a link composed of multiple Knot components.
    """

    def __init__(self, name=None):
        self.name = name or "Unnamed Link"
        self.subknots = {}  # dict[str, Knot]

    def add_knot(self, name, coords):
        """Add a knot by name and coordinate array."""
        self.subknots[name] = Knot(name, coords)

    @ property
    def total_num_points(self):
        num = 0
        for knot in list(self.subknots.values()):
            num += knot.num_points
        return num
    
    @ property
    def num_subknots(self):
        return len(list(self.subknots.keys()))

    @ property
    def minimum_distance(self):

        all_points = np.vstack(tuple([x.coords for x in list(self.subknots.values())]))
        distance_matrix = sp.spatial.distance.cdist(all_points, all_points)
        np.fill_diagonal(distance_matrix,100)
        return np.min(distance_matrix)
    
    @property
    def maximum_number_of_points_per_subknot(self):
        num_points_list = [subknot.num_points for subknot in self.subknots.values()]
        return np.max(num_points_list)

    @classmethod
    def from_csv(cls, csv_path):
        """
        Load a link from a CSV file with labeled components like:
            Component 1 of 2:
            1.0 2.0 3.0
            ...
        """
        link = cls(name=csv_path)
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        current_name = None
        current_data = []

        for line in lines:
            if line.lower().startswith("component"):
                if current_name and current_data:
                    coords = np.array([list(map(float, r.split())) for r in current_data])
                    link.add_knot(current_name, coords)
                    current_data = []
                current_name = line.rstrip(":")
            else:
                current_data.append(line)

        # Save last one
        if current_name and current_data:
            coords = np.array([list(map(float, r.split())) for r in current_data])
            link.add_knot(current_name, coords)

        return link

    def to_csv(self, path):
        """
        Write the link to a CSV/text file in the same format used for reading.
        """
        n = len(self.subknots)
        with open(path, 'w', encoding='utf-8') as f:
            for i, (name, knot) in enumerate(self.subknots.items(), start=1):
                header = f"Component {i} of {n}:" if name.lower().startswith("component") else name
                f.write(header + "\n")
                for row in knot.coords:
                    f.write(" ".join(map(str, row)) + "\n")
                if i < n:
                    f.write("\n")

    def __repr__(self):
        return f"<Link {self.name!r} with {len(self.subknots)} knots: {list(self.subknots.keys())}>"
    
    def pass_to_knots(self, fn, *args, **kwargs):
        """
        Apply a function to every Knot and return {knot_name: result}.
        `fn` can be:
          - a callable taking a Knot (fn(knot, *args, **kwargs))
          - a string attribute/method name (e.g., "arclength" or "to_string")
        """
        out = {}
        for name, knot in self.subknots.items():
            if callable(fn):
                out[name] = fn(knot, *args, **kwargs)
            else:
                attr = getattr(knot, fn)
                out[name] = attr(*args, **kwargs) if callable(attr) else attr
        return out
    
    def __getattr__(self, name):
        """
        If `name` is an attribute/method on Knot, return a dict mapping each
        component to that attribute (calling it if it's callable).
        This makes `mylink.arclength` -> {"Component 1": ..., "Component 2": ...}.
        """
        # Only runs if normal lookup failed
        if not self.subknots:
            raise AttributeError(f"'Link' has no subknots and no attribute {name!r}")

        # Check against a sample Knot
        sample = next(iter(self.subknots.values()))
        if hasattr(sample, name):
            return self.pass_to_knots(name)

        raise AttributeError(f"'Link' object has no attribute {name!r}")
    

class SetofLinks:

    def __init__(self, links):
        self.links = links

    def normalize(self, desired_num_points_per_subknot = None):

        if desired_num_points_per_subknot is None:
            desired_num_points_per_subknot = np.max([link.maximum_number_of_points_per_subknot for link in self.links])

        for link in self.links:
            normalize_link(link, desired_num_points_per_subknot)

        for link in self.links:
            assert link.total_num_points == self.links[0].total_num_points, f"Some links have different numbers of points! In particular, link {link.name} has {link.total_num_points} whereas {self.links[0].name} has {self.links[0].total_num_points} in total across all components."

        print(f"All links now have {self.links[0].total_num_points} points.")

    @classmethod
    def from_folder_name(cls, folder_name: str):
        """
        Build a SetofLinks from all .csv files in the given folder.
        Uses Link.from_csv on each file found (non-recursive).
        """
        folder = Path(folder_name)
        if not folder.is_dir():
            raise ValueError(f"Not a directory: {folder_name}")

        csv_files = sorted(
            [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
        )
        if not csv_files:
            raise ValueError(f"No .csv files found in: {folder_name}")

        links = [Link.from_csv(str(p)) for p in csv_files]
        return cls(links)
    
    def to_folder_name(self, folder_name: str, overwrite: bool = False):
        """
        Write all links in this SetofLinks to the specified folder.
        Each Link is written to '<folder>/<link_name>.csv'.

        Parameters
        ----------
        folder_name : str
            Path to the output folder (created if missing).
        overwrite : bool, optional
            If False, raises an error if a file already exists.
        """
        folder = Path(folder_name)
        folder.mkdir(parents=True, exist_ok=True)

        for i, link in enumerate(self.links, start=1):
            # Ensure a valid filename
            # Strip invalid characters, default to Link_i if name is unusable
            base_name = Path(str(link.name)).stem or f"Link_{i}"

            # Clean only *illegal* filename characters (keep dots!)
            safe_name = re.sub(r"[^A-Za-z0-9_.\- ]", "", base_name).strip()

            if not safe_name:
                safe_name = f"Link_{i}"

            out_path = folder / f"{safe_name}.csv"

            if out_path.exists() and not overwrite:
                raise FileExistsError(f"File already exists: {out_path}")

            link.to_csv(out_path)

        print(f"✅ Saved {len(self.links)} links to folder: {folder.resolve()}")

###################################################################################################
            

def generate_unknot(num_points, separation, method = "circle"):
    """
    Generate coordinates of an unknot (circle) in the x-y plane.

    Parameters
    ----------
    num_points : int
        Number of points on the circle.
    separation : float
        Arc-length separation between consecutive points.

    Returns
    -------
    points : (num_points, 3) np.ndarray
        Array of xyz coordinates forming a circle in the x-y plane.
    """

    if method == "circle":
        # Compute radius so that adjacent points are 'separation' apart
        radius = (separation * num_points) / (2 * np.pi)

        # Evenly spaced angles around the circle
        theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        # Circle in x-y plane, z = 0
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.zeros(num_points)

        # Add a tiny handle to the unknot so that extremum calculations are well defined
        x[0] += separation/10

        # Stack into Nx3 array
        return Knot(name = "Circular Unknot", coords = np.column_stack((x, y, z)))
    
    if method == "linear":
        if num_points % 2 == 0:
            num_points -= 1
            extra_point = True
        else:
            extra_point = False

        height = (num_points - 1)*separation/2 + separation/np.sqrt(2)
        z_points = np.linspace(separation, height, (num_points - 1)//2)
        x_left = np.full((num_points -1)//2,-separation/np.sqrt(2)) 
        x_right = np.full((num_points -1)//2, separation/np.sqrt(2))
        y_points = np.zeros((num_points -1)//2)
        first_coord = np.zeros(3)
        up_coords = np.column_stack((x_left, y_points, z_points))
        down_coords = np.column_stack((x_right, y_points, z_points[::-1]))
        extra_coord = np.array([0,0, height + separation/np.sqrt(2)/2]) #make sure the extremum point is well defined by making height one smaller

        if extra_point:
            coords = np.vstack((first_coord, up_coords, extra_coord, down_coords))
        
        else:
            coords = np.vstack((first_coord, up_coords, down_coords))

        return Knot(name = "Linear Unknot", coords = coords)


def knot_sum(knot1, knot2, realign = True):

    k1 = copy.deepcopy(knot1)
    k2 = copy.deepcopy(knot2)

    centers_of_mass = []
    rotation_matrices = []
    extrema_at_origin = []
    for knot in [k1, k2]:
        centers_of_mass.append(knot.center_of_mass)
        if not knot.is_orientated:
                rotation_matrices.append(knot.orientate(return_matrix = True))
        
        else:
             rotation_matrices.append(np.array([[1,0,0],[0,1,0],[0,0,1]]))

        extrema_at_origin.append(knot.extremum)
        
        # Move each knot below the z axis, pointing upwards   
        knot.coords = knot.coords - knot.extremum
        
            
    # Rotate knot2 by 180 degrees so that extremum points downwards
    k2.coords = k2.coords @ np.array([[1,0,0],[0,-1,0],[0,0,-1]]).T

    new_name = k1.name + " + " +  k2.name

    # Generate coordiantes of new knot
    i1 = k1.extremum_index
    i2 = k2.extremum_index

    part1 = k1.coords[:i1]             
    part2 = k2.coords[i2+1:]           
    part3 = k2.coords[:i2]             
    part4 = k1.coords[i1+1:]           

    combined = np.vstack((part1, part2, part3, part4))

    if realign:
        combined = combined + extrema_at_origin[0] # Move center of first knot back to origin
        combined = combined @ rotation_matrices[0] # Rotate first knot # we should really be acting with Transpose[Inverse[rotation_matrices[0]]] but this is the same as the matrix
        combined = combined + centers_of_mass[0] # Move first knot back to its original position

    return Knot(name = new_name, coords = combined)


def extend_knot(knot, desired_num_points, method = "circle"):
    assert desired_num_points >= knot.num_points, "The desired number of points is less than the number of points already in the knot"

    if desired_num_points == knot.num_points:
        return copy.deepcopy(knot)
    
    else:
        # Generate new knot as a knot sum of the previous knot with an unknot of a particular number of points
        # the unknow has 2 additional points to compensate for the loss of two points under knot sum
        new_knot = knot_sum(knot, generate_unknot(desired_num_points - knot.num_points+2, knot.segment_length, method = method))
        assert new_knot.num_points == desired_num_points, f"Error! The new knot has the wrong number of points. It's supposed to have {desired_num_points}, but instead has {new_knot.num_points} points"
        return new_knot


def extend_link(link, component_name, desired_num_points, method = "linear"):
    initial_minimum_separation = link.minimum_distance
    link.subknots[component_name] = extend_knot(link.subknots[component_name], desired_num_points, method = method)
    final_minimum_separation = link.minimum_distance

    if (final_minimum_separation + 1e7 < initial_minimum_separation):
        print(f"The minimum distance between points appears to have gone down, this the knot addition may have changed the knot topology.\nInitial minimum distance = {initial_minimum_separation}, final minimum distance = {final_minimum_separation}")
    
    assert link.subknots[component_name].num_points == desired_num_points, f"The normalized link doesn't have the right number of points in it! The subknot {component_name} is supposed to have {desired_num_points} in it, but insetead has {link.subknots[component_name].num_points} points."

def normalize_link(link, desired_num_points_per_subknot):

    assert desired_num_points_per_subknot >= np.max(np.array([knot.num_points for knot in link.subknots.values()])), "The number of points to extend to must exceed the number of points currently in each subknot of the links"

    for subknot_name in list(link.subknots.keys()):
        extend_link(link, subknot_name, desired_num_points_per_subknot, method = "linear")

    assert link.total_num_points == link.num_subknots * desired_num_points_per_subknot, f"The normalized link doesn't have the right number of points in it! It is supposed to have {link.num_subknots * desired_num_points_per_subknot} points in total, but instead has {link.total_num_points} points in total"

###############################################################################

class Visualizer:
    """
    Minimal 3D visualizer for Knot and Link objects.

    Usage:
        vis = Visualizer()
        vis.show_knot(myknot, mode="line")
        vis.show_link(mylink, mode="scatter")
    """

    def __init__(self):
        pass

    # ---------- Public API ----------

    def show_knot(self, knot, *, mode="line", closed=True, ax=None, title=None):
        coords = np.asarray(knot.coords, dtype=float)
        self._validate_coords(coords)

        fig, ax = self._ensure_axes(ax)

        if mode == "line":
            self._plot_line(ax, coords, closed=closed)
        elif mode == "scatter":
            self._plot_scatter(ax, coords)
        else:
            raise ValueError("mode must be 'line' or 'scatter'")

        self._set_equal_aspect(ax, coords)
        self._format_axes(ax, title or getattr(knot, "name", "Knot"))
        plt.show()

    def show_link(self, link, *, mode="line", closed=True, ax=None, title=None, legend=True):
        if not link.subknots:
            raise ValueError("Link has no subknots to display.")

        fig, ax = self._ensure_axes(ax)
        all_coords = []

        for name, knot in link.subknots.items():
            coords = np.asarray(knot.coords, dtype=float)
            self._validate_coords(coords)
            label = name
            if mode == "line":
                self._plot_line(ax, coords, closed=closed, label=label)
            else:
                self._plot_scatter(ax, coords, label=label)
            all_coords.append(coords)

        # ensure all points are visible
        all_coords = np.vstack(all_coords)
        self._set_equal_aspect(ax, all_coords)

        self._format_axes(ax, title or getattr(link, "name", "Link"))
        if legend:
            ax.legend(loc="best")
        plt.show()

    # ---------- Internals ----------

    def _ensure_axes(self, ax):
        if ax is not None:
            return ax.figure, ax
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.mouse_init()  # enable drag rotation
        return fig, ax

    def _validate_coords(self, coords):
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must be a 2D array with 3 columns (x,y,z).")
        if coords.size == 0:
            raise ValueError("coords is empty; nothing to plot.")

    def _plot_line(self, ax, coords, *, closed=False, label=None):
        xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
        if closed:
            xs, ys, zs = np.r_[xs, xs[0]], np.r_[ys, ys[0]], np.r_[zs, zs[0]]
        ax.plot(xs, ys, zs, label=label)

    def _plot_scatter(self, ax, coords, label=None):
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], label=label)

    def _format_axes(self, ax, title):
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    def _set_equal_aspect(self, ax, coords, pad=0.05):
        """Set equal aspect and limits so all points are visible."""
        xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
        x_range = xs.max() - xs.min()
        y_range = ys.max() - ys.min()
        z_range = zs.max() - zs.min()
        max_range = max(x_range, y_range, z_range) or 1.0

        x_mid = (xs.max() + xs.min()) / 2
        y_mid = (ys.max() + ys.min()) / 2
        z_mid = (zs.max() + zs.min()) / 2
        r = max_range * (0.5 + pad)

        ax.set_xlim(x_mid - r, x_mid + r)
        ax.set_ylim(y_mid - r, y_mid + r)
        ax.set_zlim(z_mid - r, z_mid + r)


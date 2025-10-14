# DEAD
import copy
import numpy as np
from knots_and_links import Knot

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
        return knot_sum(knot, generate_unknot(desired_num_points - knot.num_points, knot.segment_length, method = method))

def extend_link(link, component_name, desired_num_points, method = "linear"):
    initial_minimum_separation = link.minimum_distance
    link.subknots[component_name] = extend_knot(link.subknots[component_name], desired_num_points, method = method)
    final_minimum_separation = link.minimum_distance

    if (final_minimum_separation + 1e7 < initial_minimum_separation):
        print(f"The minimum distance between points appears to have gone down, this the knot addition may have changed the knot topology.\nInitial minimum distance = {initial_minimum_separation}, final minimum distance = {final_minimum_separation}")
    

def normalize_link(link, desired_num_points_per_subknot):

    assert desired_num_points_per_subknot > np.max(np.array([knot.num_points for knot in link.subknots.values()])), "The number of points to extend to must exceed the number of points currently in each subknot of the links"

    for subknot_name in list(link.subknots.keys()):
        extend_link(link, subknot_name, desired_num_points_per_subknot, method = "linear")
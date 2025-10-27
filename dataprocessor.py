import numpy as np
from writhe import *

def load_links(filename, num_subknots, points_per_subknot, num_dimensions):
    total_num_points = num_subknots * points_per_subknot
    data = np.loadtxt(filename)

    return data.reshape(-1, total_num_points, num_dimensions)


def load_writhe(filename, num_subknots, points_per_subknot):
    total_num_points = num_subknots * points_per_subknot
    data = np.loadtxt(filename)

    return data.reshape(-1, total_num_points, total_num_points)
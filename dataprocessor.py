import numpy as np
from writhe import *
from plotting import *

def load_links(filename, num_subknots, points_per_subknot, num_dimensions):
    total_num_points = num_subknots * points_per_subknot
    data = np.loadtxt(filename)

    return data.reshape(-1, total_num_points, num_dimensions)


def load_writhe(filename, num_subknots, points_per_subknot):
    total_num_points = num_subknots * points_per_subknot
    data = np.loadtxt(filename)

    return data.reshape(-1, total_num_points, total_num_points)

def compute_total_writhe_from_chunk(chunk): # chunk is numpy array of shape (10000, 102, 102) i.e. (num_iterations, Nbeads, Nbeads)
    return chunk.sum(axis = (1,2))/(np.pi * 4)  # Sum of the writhe axes 

def plot_histogram_from_writhe_dict(writhe_dict, bins = 20, density = False, labels = None, title = "Histograms (line style)"):
    
    keys = list(writhe_dict.keys())
    values = [writhe_dict[key] for key in keys]
    data = [compute_total_writhe_from_chunk(chunk) for chunk in values]

    labels = keys if labels is None else labels 
    
    plot_multi_hist(data, bins=bins, density=density, labels=labels, title=title)
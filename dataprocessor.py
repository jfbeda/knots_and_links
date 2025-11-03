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

def compute_total_writhe_from_chunk(chunk, just_a_to_b_writhe = False): # chunk is numpy array of shape (10000, 102, 102) i.e. (num_iterations, Nbeads, Nbeads)
    """
    just_a_to_b_writhe tells the computer to return the writhe only of the component 1 with component 2, not of all components with each other
    """
    
    # THIS SECTION OF CODE IS BAD
    if just_a_to_b_writhe:        
        # Turn (num_iterations, Nbeads, Nbeads) array into an array of shape
        # (num_iterations, 2, 2, Nbeads/2, Nbeads/2) the (2x2) matrix is a block matrix corresponding to the chunks where
        # sublink i and sublink j are writhed together 
        Nbeads= chunk.shape[1]
        subchunkifiedchunk = chunk.reshape(1000, 2, Nbeads//2, 2, Nbeads//2).transpose(0, 1, 3, 2, 4)

        return subchunkifiedchunk.sum(axis = (3, 4))[0,0]
    
    else:
        return chunk.sum(axis = (1,2))/(np.pi * 4)  # Sum of the writhe axes 

def plot_histogram_from_writhe_dict(writhe_dict, bins = 20, density = False, labels = None, title = "Histograms (line style)", just_a_to_b_writhe = False):
    
    keys = list(writhe_dict.keys())
    values = [writhe_dict[key] for key in keys]
    data = [compute_total_writhe_from_chunk(chunk, just_a_to_b_writhe= just_a_to_b_writhe) for chunk in values]

    labels = keys if labels is None else labels 
    
    
    plot_multi_hist(data, bins=bins, density=density, labels=labels, title=title)
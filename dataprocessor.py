import numpy as np
from writhe import *
from plotting import *
from scipy.stats import iqr as interquartile_range
import os

"""
### writhe_dict:
a single layered dictionary that looks like
writhe_dict["WRITHE_0.2.1.dat"] = np.array() of shape (1000, 120, 120)

### total_writhe_dict:
a single layered dictionary that looks like
total_writhe_dict["WRITHE_0.2.1.dat"] = np.array() of shape (1000)

### data_dict:
a double layered dictionary that looks like
data_dict[<some parameter>] = writhe_dict
data_dict[<some parameter>]["WRITHE_0.2.1.dat"] = np.array() of shape (1000, 120, 120)

so for example, we might have have a temperature, we feed into data_dict to get a writhe dict
"""

def load_links(filename, num_subknots, points_per_subknot, num_dimensions):
    total_num_points = num_subknots * points_per_subknot
    data = np.loadtxt(filename)

    return data.reshape(-1, total_num_points, num_dimensions)

    # Returns an array of shape (num_links, total_num_points, num_dimensions)

def xyz_dict_from_path(input_path, num_subknots, points_per_subknot, num_dimensions):
    xyz_dict = {}
    for filename in os.listdir(input_path):
        if filename.endswith(".dat.nos"):
            filepath = os.path.join(input_path, filename)
            loaded_links = load_links(filepath, num_subknots, points_per_subknot, num_dimensions)
            xyz_dict[filename] = loaded_links
    return xyz_dict

def writhe_dict_from_path(input_path, num_subknots, points_per_subknot):
    writhe_dict = {}
    for filename in os.listdir(input_path):
        if filename.endswith(".dat"):
            filepath = os.path.join(input_path, filename)
            loaded_writhes = load_writhe(filepath, num_subknots, points_per_subknot)
            writhe_dict[filename] = loaded_writhes
    return writhe_dict

def add_noise_to_xyz(datachunk, standard_deviation):
    noise = np.random.normal(0, standard_deviation, datachunk.shape)
    return datachunk + noise

def add_noise_to_xyz_dict(xyz_dict, standard_deviation):
    noisy_xyz_dict = {}
    for key, datachunk in xyz_dict.items():
        noisy_xyz_dict[key] = add_noise_to_xyz(datachunk, standard_deviation)
    return noisy_xyz_dict

def save_xyz_dict_to_folder(folder_name, xyz_dict):
    os.makedirs(folder_name, exist_ok = True)

    for link_name, datachunk in xyz_dict.items():
        full_path = os.path.join(folder_name, link_name)
        data = datachunk.reshape(-1, datachunk.shape[-1])
        np.savetxt(full_path, data)

def produce_noisy_xyz_folder(xyz_output_path, xyz_dict, noises):
    for noise_stdev in noises:
        print(f"Processing noise with standard deviation {noise_stdev}")
        full_folder_path = os.path.join(xyz_output_path, f"4_xyz_files_{noise_stdev}")
        noisy_xyz_dict = add_noise_to_xyz_dict(xyz_dict, noise_stdev)
        save_xyz_dict_to_folder(full_folder_path, noisy_xyz_dict)

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
    
def total_writhe_dict_from_writhe_dict(writhe_dict, just_a_to_b_writhe = False):
    keys = writhe_dict.keys()

    total_writhe_dict = {key : compute_total_writhe_from_chunk(writhe_dict[key], just_a_to_b_writhe= just_a_to_b_writhe) for key in keys}

    return total_writhe_dict

def reduce_writhe_dict(writhe_dict):
    # Reduces a full 1-1, 1-2, 2-1, 2-2 writhe dict to just the 1-2 writhe dict
    reduced_dict = dict()
    num_points = writhe_dict[list(writhe_dict.keys())[0]][0].shape[0]//2
    

    for link_name in writhe_dict.keys():
        old_writhe_matrices = writhe_dict[link_name]
        new_writhe_matrices = np.zeros(shape = (old_writhe_matrices.shape[0], num_points, num_points))
        for i,writhe_matrix in enumerate(old_writhe_matrices):
            new_writhe_matrix = writhe_matrix[:num_points,num_points:]
            new_writhe_matrices[i] = new_writhe_matrix
        reduced_dict[link_name] = new_writhe_matrices

    return reduced_dict

def reduce_writhe_dict_to_11(writhe_dict):
    # Reduces a full 1-1, 1-2, 2-1, 2-2 writhe dict to just the 11 writhe dict
    reduced_dict = dict()
    num_points = writhe_dict[list(writhe_dict.keys())[0]][0].shape[0]//2
    

    for link_name in writhe_dict.keys():
        old_writhe_matrices = writhe_dict[link_name]
        new_writhe_matrices = np.zeros(shape = (old_writhe_matrices.shape[0], num_points, num_points))
        for i,writhe_matrix in enumerate(old_writhe_matrices):
            new_writhe_matrix = writhe_matrix[:num_points,:num_points]
            new_writhe_matrices[i] = new_writhe_matrix
        reduced_dict[link_name] = new_writhe_matrices

    return reduced_dict

def reduce_writhe_dict_to_22(writhe_dict):
    # Reduces a full 1-1, 1-2, 2-1, 2-2 writhe dict to just the 22 writhe dict
    reduced_dict = dict()
    num_points = writhe_dict[list(writhe_dict.keys())[0]][0].shape[0]//2
    

    for link_name in writhe_dict.keys():
        old_writhe_matrices = writhe_dict[link_name]
        new_writhe_matrices = np.zeros(shape = (old_writhe_matrices.shape[0], num_points, num_points))
        for i,writhe_matrix in enumerate(old_writhe_matrices):
            new_writhe_matrix = writhe_matrix[num_points:,num_points:]
            new_writhe_matrices[i] = new_writhe_matrix
        reduced_dict[link_name] = new_writhe_matrices

    return reduced_dict

def save_writhe_dict_to_folder(folder_name, writhe_dict):
    
    os.makedirs(folder_name, exist_ok = False)

    for link_name in writhe_dict.keys():
        full_path = os.path.join(folder_name, link_name)
        writhe_matrices = writhe_dict[link_name]
        np.savetxt(full_path,writhe_matrices.reshape(-1, writhe_matrices.shape[-1]))

    print(f"saved to {folder_name}")

# def totaled_data_dict_from_data_dict(data_dict, just_a_to_b_writhe = False):

#     totaled_data_dict = dict()
#     for parameter in data_dict.keys():
#         totaled_data_dict[parameter] = total_writhe_dict_from_writhe_dict(data_dict[parameter], just_a_to_b_writhe = False)

#     return totaled_data_dict

# def interquartile_range_dict_from_data_dict(data_dict):
#     totaled_data_dict = total_writhe_dict_from_writhe_dict(data_dict)

#     interquartile_range_dict = dict()

#     for parameter in totaled_data_dict.keys():
#         interquartile_range_dict[parameter] = dict()
#         for link_name in totaled_data_dict[parameter].keys():
#             total_writhes = totaled_data_dict[parameter][link_name]
#             interquartile_range_dict[parameter][link_name] = interquartile_range(total_writhes)

#     return interquartile_range_dict
    
def interquartile_range_dict_from_data_dict3(data_dict):
    totaled_data_dict = dict()
    for parameter in data_dict.keys():
        totaled_data_dict[parameter] = total_writhe_dict_from_writhe_dict(data_dict[parameter], just_a_to_b_writhe = False)

    interquartile_range_dict = dict()

    for parameter in totaled_data_dict.keys():
        interquartile_range_dict[parameter] = dict()
        for link_name in totaled_data_dict[parameter].keys():
            total_writhes = totaled_data_dict[parameter][link_name]
            interquartile_range_dict[parameter][link_name] = interquartile_range(total_writhes)

    return interquartile_range_dict



def plot_histogram_from_writhe_dict(writhe_dict, bins = 20, density = False, labels = None, title = "Histograms (line style)", just_a_to_b_writhe = False, xlabel = "value", ylabel = "Density"):
    
    keys = list(writhe_dict.keys())
    values = [writhe_dict[key] for key in keys]
    data = [compute_total_writhe_from_chunk(chunk, just_a_to_b_writhe= just_a_to_b_writhe) for chunk in values]

    labels = keys if labels is None else labels 
    
    
    plot_multi_hist(data, bins=bins, density=density, labels=labels, title=title, xlabel = xlabel, ylabel = ylabel)

from pathlib import Path
import numpy as np

def save_histogram_data_from_writhe_dict(
    writhe_dict,
    out_file="../figure_data/histograms_writhe.txt",
    bins=20,
    density=False,
    labels=None,
    title="Histograms (saved)",
    just_a_to_b_writhe=False,
    hist_range=None,
):
    """
    Save histogram data (shared bin edges + counts/density per series) to a text file.

    hist_range : tuple(float, float) or None
        Optional (min,max) range for binning when bins is an int. If None, uses global min/max.
    """

    keys = list(writhe_dict.keys())

    chunks = [writhe_dict[k] for k in keys]
    data_series = [
        np.asarray(
            compute_total_writhe_from_chunk(chunk, just_a_to_b_writhe=just_a_to_b_writhe),
            dtype=float,
        )
        for chunk in chunks
    ]

    series_names = keys if labels is None else labels
    if len(series_names) != len(data_series):
        raise ValueError("labels must have the same length as writhe_dict keys")

    # Build shared bin edges
    if np.isscalar(bins):
        all_data = np.concatenate([d for d in data_series if d.size > 0], axis=0)
        if all_data.size == 0:
            raise ValueError("No data points found in writhe_dict.")

        if hist_range is None:
            dmin = float(np.min(all_data))
            dmax = float(np.max(all_data))
            if dmin == dmax:
                eps = 0.5 if dmin == 0 else abs(dmin) * 0.01
                dmin -= eps
                dmax += eps
            hist_range_used = (dmin, dmax)
        else:
            hist_range_used = (float(hist_range[0]), float(hist_range[1]))

        bin_edges = np.histogram_bin_edges(all_data, bins=int(bins), range=hist_range_used)
    else:
        bin_edges = np.asarray(bins, dtype=float)
        if bin_edges.ndim != 1 or bin_edges.size < 2:
            raise ValueError("If bins is array-like, it must be 1D bin edges with length >= 2.")
        if not np.all(np.diff(bin_edges) > 0):
            raise ValueError("Bin edges must be strictly increasing.")

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# histogram_data v1")
    lines.append(f"# title: {title}")
    lines.append(f"# density: {1 if density else 0}")
    lines.append(f"# nbins: {len(bin_edges) - 1}")
    lines.append("# BINS block lists bin edges (N+1 lines), then series blocks give per-bin values.")
    lines.append("")

    lines.append("BINS")
    for e in bin_edges:
        lines.append(f"{float(e):.12g}")
    lines.append("END_BINS")
    lines.append("")

    # Per-series histogram
    for name, d in zip(series_names, data_series):
        vals, _ = np.histogram(d, bins=bin_edges, density=density)
        lines.append(f"SERIES {name}")

        # enumerate avoids calling built-in range()
        for i, val in enumerate(vals):
            left = float(bin_edges[i])
            right = float(bin_edges[i + 1])
            lines.append(f"{left:.12g} {right:.12g} {float(val):.12g}")

        lines.append("END_SERIES")
        lines.append("")

    out_path.write_text("\n".join(lines))
    return str(out_path)

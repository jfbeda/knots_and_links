import numpy as np
import matplotlib.pyplot as plt

def plot_multi_hist(data, bins=20, density=False, labels=None, title="Histograms (line style)"):
    """
    Plot multiple histograms as lines, one for each dataset in a list of lists.

    Parameters
    ----------
    data : list of lists or 2D array
        Each sublist is a separate dataset.
    bins : int or sequence, optional
        Number of histogram bins (default: 20).
    density : bool, optional
        If True, normalize to form a probability density.
    labels : list of str, optional
        Labels for each dataset.
    title : str, optional
        Plot title.
    """

    if labels is None:
        labels = [f"Data {i}" for i in range(len(data))]

    for i, d in enumerate(data):
        counts, bin_edges = np.histogram(d, bins=bins, density=density)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        plt.plot(bin_centers, counts, label=labels[i])
    
    plt.legend()
    plt.xlabel("Value")
    plt.ylabel("Density" if density else "Frequency")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

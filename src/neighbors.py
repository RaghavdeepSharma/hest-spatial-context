"""
For each spot in a sample, find its k nearest neighboring spots based on
pixel coordinates. Distance is relative (nearest by rank) rather than a
fixed pixel threshold, since pixel-to-micrometer resolution can differ
between samples - a fixed pixel cutoff would mean a different physical
neighborhood size depending on the sample.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def find_neighbors(coords, k=6):
    """coords: (n_spots, 2) pixel positions.
    Returns (n_spots, k) array of neighbor indices, excluding the spot
    itself, ordered nearest-first."""
    # k+1 because the nearest "neighbor" to any spot is always itself
    # at distance 0, so we ask for one extra and drop that first column
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nn.kneighbors(coords)
    return indices[:, 1:]

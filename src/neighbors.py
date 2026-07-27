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

def add_neighborhood_context(embeddings, neighbor_idx):
    """Combine each spot's own embedding with the average embedding of its
    spatial neighbors. The center spot's own embedding is kept untouched
    and the neighbor average is appended alongside it, rather than blended
    together - this way the center spot's own signal isn't diluted, and
    the regression can learn how much weight to give the surrounding
    context versus the spot itself.

    embeddings: (n_spots, embed_dim)
    neighbor_idx: (n_spots, k) from find_neighbors

    Returns (n_spots, embed_dim * 2) - own embedding concatenated with
    neighbor-average embedding.
    """
    neighbor_avg = embeddings[neighbor_idx].mean(axis=1)
    return np.concatenate([embeddings, neighbor_avg], axis=1)

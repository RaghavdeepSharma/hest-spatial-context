"""
Look at a single hest-bench sample and check that the patch images and the
expression matrix actually refer to the same spots.

The patches file (.h5) stores one image per ST spot plus that spot's barcode.
The expression file (.h5ad) stores the raw gene counts per spot, indexed by
the same barcode. Before doing anything else with this data we need to know
the two files line up correctly and in what order, since nothing downstream
means anything if a patch gets paired with the wrong spot's expression.

Usage:
    python inspect_sample.py --patches path/to/SAMPLE.h5 --expr path/to/SAMPLE.h5ad
"""
import argparse

import anndata as ad
import h5py
import numpy as np


def load_patches(h5_path):
    with h5py.File(h5_path, "r") as f:
        print(f"patches file keys: {list(f.keys())}")
        imgs = f["img"][:]
        barcodes_raw = f["barcode"][:]
    # barcodes are stored as byte strings inside a 2D array, need to flatten
    # and decode them to plain python strings before we can compare anything
    barcodes = np.array([b[0].decode() for b in barcodes_raw])
    return imgs, barcodes


def load_expression(h5ad_path):
    adata = ad.read_h5ad(h5ad_path)
    counts = adata.X
    if hasattr(counts, "todense"):
        counts = np.asarray(counts.todense())
    else:
        counts = np.asarray(counts)
    barcodes = np.asarray(adata.obs_names)
    genes = np.asarray(adata.var_names)
    return counts, barcodes, genes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patches", required=True)
    parser.add_argument("--expr", required=True)
    args = parser.parse_args()

    imgs, bc_img = load_patches(args.patches)
    counts, bc_expr, genes = load_expression(args.expr)

    print(f"\npatch images : {imgs.shape}, dtype={imgs.dtype}")
    print(f"patch barcodes: {bc_img.shape}, e.g. {bc_img[:3]}")
    print(f"expression counts: {counts.shape}")
    print(f"expression barcodes: {bc_expr.shape}, e.g. {bc_expr[:3]}")
    print(f"genes: {genes.shape}, e.g. {genes[:5]}")

    # the two files don't have to be in the same order, or even have the
    # exact same spots (dropout / QC can drop a spot from one but not the
    # other) - so check the overlap directly rather than assuming row i
    # in one file matches row i in the other
    set_img = set(bc_img)
    set_expr = set(bc_expr)
    overlap = set_img & set_expr

    print(f"\nspots in patches file : {len(set_img)}")
    print(f"spots in expression file: {len(set_expr)}")
    print(f"spots present in both  : {len(overlap)}")

    if len(overlap) == 0:
        print("\nNO OVERLAP - something is wrong, these files don't describe the same sample")
    elif len(overlap) < min(len(set_img), len(set_expr)):
        dropped = min(len(set_img), len(set_expr)) - len(overlap)
        print(f"\n{dropped} spot(s) present in only one file - normal amount of QC dropout, not an error")
    else:
        print("\nfull match - every spot in the smaller file is present in the larger one")


if __name__ == "__main__":
    main()

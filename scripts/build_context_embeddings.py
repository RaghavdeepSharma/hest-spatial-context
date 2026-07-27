"""
Build neighborhood-context embeddings for every sample in a task: for each
spot, concatenate its own ResNet50 embedding with the average embedding of
its k nearest spatial neighbors. Saves one output .npz per sample, separate
from the original embeddings, so both versions can be compared directly.

Requires extract_embeddings_batch.py to have already been run for this
task, since this reads its output rather than re-embedding patches.

Usage:
    python build_context_embeddings.py --task_dir data/PRAD --k 6
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from neighbors import find_neighbors, add_neighborhood_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", required=True)
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()

    embeddings_dir = os.path.join(args.task_dir, "embeddings")
    patches_dir = os.path.join(args.task_dir, "patches")
    out_dir = os.path.join(args.task_dir, "embeddings_context")
    os.makedirs(out_dir, exist_ok=True)

    emb_files = sorted(glob.glob(os.path.join(embeddings_dir, "*.npz")))
    print(f"found {len(emb_files)} embedded samples")

    for emb_path in emb_files:
        sample_id = os.path.splitext(os.path.basename(emb_path))[0]
        out_path = os.path.join(out_dir, f"{sample_id}.npz")

        if os.path.exists(out_path):
            print(f"[skip] {sample_id} already done")
            continue

        d = np.load(emb_path)
        embeddings = d["embeddings"]
        barcodes = d["barcodes"]

        # coords live in the original patches file, keyed by the same
        # barcode order as the embeddings were extracted in, so no
        # re-alignment is needed here - both came from the same .h5 read
        with h5py.File(os.path.join(patches_dir, f"{sample_id}.h5"), "r") as f:
            coords = f["coords"][:]

        print(f"[run ] {sample_id}: {len(embeddings)} spots")
        neighbor_idx = find_neighbors(coords, k=args.k)
        combined = add_neighborhood_context(embeddings, neighbor_idx)

        np.savez(out_path, embeddings=combined, barcodes=barcodes)
        print(f"       -> {combined.shape} saved to {out_path}")


if __name__ == "__main__":
    main()

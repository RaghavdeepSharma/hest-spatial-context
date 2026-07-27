"""
Run the ResNet50 baseline benchmark on a task, using the real patient-
stratified folds and the previously extracted embeddings.

Usage:
    python run_baseline.py --task_dir data/PRAD
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from baseline_regression import run_cross_validation
from gene_selection import select_benchmark_genes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", required=True)
    parser.add_argument("--n_pca", type=int, default=256)
    args = parser.parse_args()

    adata_dir = os.path.join(args.task_dir, "adata")
    embeddings_dir = os.path.join(args.task_dir, "embeddings")
    splits_dir = os.path.join(args.task_dir, "splits")

    print("selecting genes...")
    all_h5ad = sorted(glob.glob(os.path.join(adata_dir, "*.h5ad")))
    top_genes = select_benchmark_genes(all_h5ad)
    print(f"selected genes: {list(top_genes)}\n")

    train_files = sorted(glob.glob(os.path.join(splits_dir, "train_*.csv")))
    test_files = sorted(glob.glob(os.path.join(splits_dir, "test_*.csv")))
    fold_pairs = list(zip(train_files, test_files))
    print(f"found {len(fold_pairs)} folds\n")

    run_cross_validation(fold_pairs, embeddings_dir, adata_dir, top_genes, n_pca=args.n_pca)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

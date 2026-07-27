"""
Compare the benchmark's original gene selection (ribosomal/mitochondrial
genes left in, as the paper describes it) against the filtered version
(those genes dropped before ranking by variance), on the same task and
the same folds, so the two numbers are directly comparable.

Usage:
    python compare_gene_filters.py --task_dir data/PRAD
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from baseline_regression import run_cross_validation
from gene_selection import select_benchmark_genes


def run_variant(label, drop_ribo_mito, all_h5ad, fold_pairs, embeddings_dir, adata_dir, n_pca):
    print(f"\n=== {label} ===")
    top_genes, gene_idx = select_benchmark_genes(all_h5ad, drop_ribo_mito=drop_ribo_mito)
    print(f"genes: {list(top_genes)}\n")
    scores = run_cross_validation(fold_pairs, embeddings_dir, adata_dir, gene_idx, n_pca=n_pca)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", required=True)
    parser.add_argument("--n_pca", type=int, default=256)
    args = parser.parse_args()

    adata_dir = os.path.join(args.task_dir, "adata")
    embeddings_dir = os.path.join(args.task_dir, "embeddings")
    splits_dir = os.path.join(args.task_dir, "splits")

    all_h5ad = sorted(glob.glob(os.path.join(adata_dir, "*.h5ad")))
    train_files = sorted(glob.glob(os.path.join(splits_dir, "train_*.csv")))
    test_files = sorted(glob.glob(os.path.join(splits_dir, "test_*.csv")))
    fold_pairs = list(zip(train_files, test_files))

    original_scores = run_variant("original (ribo/mito kept)", False, all_h5ad,
                                  fold_pairs, embeddings_dir, adata_dir, args.n_pca)
    filtered_scores = run_variant("filtered (ribo/mito dropped)", True, all_h5ad,
                                  fold_pairs, embeddings_dir, adata_dir, args.n_pca)

    print("\n=== summary ===")
    print(f"{'variant':<30} {'mean':>8} {'std':>8}")
    print(f"{'original (ribo/mito kept)':<30} {original_scores.mean():>8.4f} {original_scores.std():>8.4f}")
    print(f"{'filtered (ribo/mito dropped)':<30} {filtered_scores.mean():>8.4f} {filtered_scores.std():>8.4f}")


if __name__ == "__main__":
    main()


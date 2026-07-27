"""
The main experiment: does adding spatial neighborhood context improve
gene-expression prediction versus the single-patch baseline?

Runs the same PCA+Ridge regression, same folds, same selected genes, once
on the plain ResNet50 embeddings and once on the neighborhood-context
embeddings (own patch + average of k nearest neighbors), so the only
thing that differs between the two runs is whether neighbor information
was included.

Requires build_context_embeddings.py to have already been run for this task.

Usage:
    python compare_context_vs_baseline.py --task_dir data/PRAD
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
    baseline_emb_dir = os.path.join(args.task_dir, "embeddings")
    context_emb_dir = os.path.join(args.task_dir, "embeddings_context")
    splits_dir = os.path.join(args.task_dir, "splits")

    print("selecting genes (ribo/mito filtered)...")
    all_h5ad = sorted(glob.glob(os.path.join(adata_dir, "*.h5ad")))
    top_genes = select_benchmark_genes(all_h5ad, drop_ribo_mito=True)
    print(f"genes: {list(top_genes)}\n")

    train_files = sorted(glob.glob(os.path.join(splits_dir, "train_*.csv")))
    test_files = sorted(glob.glob(os.path.join(splits_dir, "test_*.csv")))
    fold_pairs = list(zip(train_files, test_files))
    print(f"found {len(fold_pairs)} folds\n")

    print("=== baseline: single patch embedding ===")
    baseline_scores = run_cross_validation(
        fold_pairs, baseline_emb_dir, adata_dir, top_genes, n_pca=args.n_pca
    )

    print("\n=== neighborhood context: patch + neighbor average ===")
    context_scores = run_cross_validation(
        fold_pairs, context_emb_dir, adata_dir, top_genes, n_pca=args.n_pca
    )

    print("\n=== summary ===")
    print(f"{'variant':<30} {'mean':>8} {'std':>8}")
    print(f"{'baseline (single patch)':<30} {baseline_scores.mean():>8.4f} {baseline_scores.std():>8.4f}")
    print(f"{'neighborhood context':<30} {context_scores.mean():>8.4f} {context_scores.std():>8.4f}")

    diff = context_scores.mean() - baseline_scores.mean()
    direction = "improved" if diff > 0 else "did not improve"
    print(f"\nneighborhood context {direction} on average Pearson by {diff:+.4f}")


if __name__ == "__main__":
    main()

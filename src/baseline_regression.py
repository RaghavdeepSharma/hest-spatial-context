"""
The core benchmark: predict the 50 selected genes' log1p expression from
frozen ResNet50 embeddings, using PCA to a fixed dimension then Ridge
regression, evaluated with patient-stratified cross validation.

The train/test splits are the authors' own patient-stratified folds, so a
patient's spots never appear on both sides of a fold - a model can't cheat
by memorizing tissue quirks specific to one patient.
"""
import os

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(n_pca, n_genes):
    # lambda scales with both the PCA dimension and the number of gene
    # targets, so the regularization strength stays sensible regardless
    # of which encoder or task we plug in
    alpha = 100.0 / (n_pca * n_genes)
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=0)),
        ("ridge", Ridge(alpha=alpha)),
    ])


def load_sample_xy(sample_id, embeddings_dir, adata_dir, gene_idx):
    """Load one sample's embeddings and its expression for the selected
    genes, aligned by barcode. gene_idx are column positions into the
    FULL (unfiltered) gene list from gene_selection.py."""
    emb_file = np.load(os.path.join(embeddings_dir, f"{sample_id}.npz"))
    emb_barcodes = emb_file["barcodes"]
    embeddings = emb_file["embeddings"]

    adata = ad.read_h5ad(os.path.join(adata_dir, f"{sample_id}.h5ad"))
    counts = adata.X
    if hasattr(counts, "todense"):
        counts = np.asarray(counts.todense())
    expr_barcodes = np.asarray(adata.obs_names)

    # the embedding file and the h5ad file aren't guaranteed to list spots
    # in the same order, so match them up by barcode rather than assuming
    # row i in one file lines up with row i in the other
    lookup = {b: i for i, b in enumerate(expr_barcodes)}
    keep_emb, keep_expr = [], []
    for i, b in enumerate(emb_barcodes):
        j = lookup.get(b)
        if j is not None:
            keep_emb.append(i)
            keep_expr.append(j)

    X = embeddings[keep_emb]
    y = np.log1p(counts[keep_expr][:, gene_idx])
    return X, y


def load_fold_xy(sample_ids, embeddings_dir, adata_dir, gene_idx):
    """Stack every sample in a fold's sample list into one X, y pair."""
    X_parts, y_parts = [], []
    for sid in sample_ids:
        X, y = load_sample_xy(sid, embeddings_dir, adata_dir, gene_idx)
        X_parts.append(X)
        y_parts.append(y)
    return np.concatenate(X_parts), np.concatenate(y_parts)


def pearson_per_gene(y_true, y_pred):
    """Correlate predicted vs true expression separately for each gene,
    so a highly-expressed gene can't dominate the score just by having
    bigger numbers than a rarer gene."""
    n_genes = y_true.shape[1]
    scores = np.full(n_genes, np.nan)
    for g in range(n_genes):
        if y_true[:, g].std() > 0 and y_pred[:, g].std() > 0:
            scores[g] = pearsonr(y_true[:, g], y_pred[:, g])[0]
    return scores


def run_cross_validation(fold_pairs, embeddings_dir, adata_dir, gene_idx, n_pca=256):
    """fold_pairs: list of (train_csv_path, test_csv_path), the authors'
    own patient-stratified splits. Returns the mean Pearson per fold."""
    n_genes = len(gene_idx)
    fold_scores = []

    for i, (train_csv, test_csv) in enumerate(fold_pairs):
        train_ids = pd.read_csv(train_csv)["sample_id"].tolist()
        test_ids = pd.read_csv(test_csv)["sample_id"].tolist()

        X_train, y_train = load_fold_xy(train_ids, embeddings_dir, adata_dir, gene_idx)
        X_test, y_test = load_fold_xy(test_ids, embeddings_dir, adata_dir, gene_idx)

        n_pca_actual = min(n_pca, X_train.shape[0], X_train.shape[1])
        model = build_model(n_pca=n_pca_actual, n_genes=n_genes)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        gene_scores = pearson_per_gene(y_test, y_pred)
        fold_mean = np.nanmean(gene_scores)
        fold_scores.append(fold_mean)
        print(f"fold {i}: {len(train_ids)} train samples, {len(test_ids)} test samples, "
              f"mean Pearson = {fold_mean:.4f}")

    fold_scores = np.array(fold_scores)
    print(f"\noverall: {np.nanmean(fold_scores):.4f} +/- {np.nanstd(fold_scores):.4f}")
    return fold_scores

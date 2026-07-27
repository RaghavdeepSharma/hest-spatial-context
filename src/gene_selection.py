"""
Pick the 50 most variable genes across all samples in a task.
"""
import numpy as np
import anndata as ad


def load_counts(h5ad_path):
    adata = ad.read_h5ad(h5ad_path)
    counts = adata.X
    if hasattr(counts, "todense"):
        counts = np.asarray(counts.todense())
    else:
        counts = np.asarray(counts)
    genes = np.asarray(adata.var_names)
    return counts, genes

def common_genes(gene_lists):
    """Find the genes present in every sample. Some cohorts mix samples
    processed against different reference panels (e.g. a full
    whole-transcriptome reference vs. a smaller filtered/probe-based one),
    so requiring an exact identical gene list across all samples is too
    strict - taking the intersection is the safe way to combine them
    without ever mixing up which column means which gene."""
    common = set(gene_lists[0])
    for genes in gene_lists[1:]:
        common &= set(genes)
    if len(common) == 0:
        raise ValueError("no genes are shared across every sample - cannot pool counts")
    return np.array(sorted(common))


def pool_counts(h5ad_paths):
    """Load every sample's counts and stack them into one big matrix,
    restricted to genes common to every sample so a mismatched gene panel
    on any one sample can't silently corrupt the column alignment."""
    all_counts = []
    all_genes = []
    for path in h5ad_paths:
        counts, genes = load_counts(path)
        all_counts.append(counts)
        all_genes.append(genes)

    reference_genes = common_genes(all_genes)

    aligned_counts = []
    for counts, genes in zip(all_counts, all_genes):
        lookup = {g: i for i, g in enumerate(genes)}
        col_idx = [lookup[g] for g in reference_genes]
        aligned_counts.append(counts[:, col_idx])

    pooled = np.concatenate(aligned_counts, axis=0)
    return pooled, reference_genes


def filter_low_expressed(counts, genes, min_frac=0.10):
    """Drop genes that are non-zero in fewer than min_frac of all spots.
    A gene expressed in only a handful of spots is too sparse to be a
    useful regression target - mostly zeros with occasional noise spikes."""
    n_spots = counts.shape[0]
    expressed_frac = (counts > 0).sum(axis=0) / n_spots
    keep = expressed_frac >= min_frac
    return counts[:, keep], genes[keep]

def filter_ribo_mito(counts, genes):
    """Drop ribosomal and mitochondrial genes before ranking by variance.
    These are highly expressed in every cell regardless of biology, so a
    raw-variance ranking gets dominated by them, crowding out genes that
    actually distinguish tissue states or disease biology."""
    genes = np.asarray(genes)
    is_ribo = np.char.startswith(genes.astype(str), "RPS") | np.char.startswith(genes.astype(str), "RPL")
    is_mito = np.char.startswith(genes.astype(str), "MT-")
    keep = ~(is_ribo | is_mito)
    return counts[:, keep], genes[keep]

def select_top_variable_genes(counts, genes, n_top=50):
    """Rank genes by variance of their log1p-transformed counts, keep the
    n_top most variable. log1p is used because raw count variance is
    dominated by a handful of extremely highly-expressed genes; log1p
    compresses that scale so variance reflects meaningful biological
    spread rather than just raw magnitude."""
    log_counts = np.log1p(counts)
    variances = log_counts.var(axis=0)
    top_idx = np.argsort(variances)[::-1][:n_top]
    top_idx = np.sort(top_idx)  # keep original gene order, easier to read later
    return genes[top_idx], top_idx


def select_benchmark_genes(h5ad_paths, n_top=50, min_frac=0.10, drop_ribo_mito=True):
    """Full pipeline: pool all samples, drop sparse genes, pick the
    n_top most variable. Returns the gene names and their column
    indices into the ORIGINAL (unfiltered) gene list, so callers can
    pull the right columns out of any individual sample's matrix."""
    pooled, genes = pool_counts(h5ad_paths)
    filtered_counts, filtered_genes = filter_low_expressed(pooled, genes, min_frac)
    if drop_ribo_mito:
        filtered_counts, filtered_genes = filter_ribo_mito(filtered_counts, filtered_genes)
    top_genes, _ = select_top_variable_genes(filtered_counts, filtered_genes, n_top)

    return top_genes


if __name__ == "__main__":
    import argparse
    import glob
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--adata_dir", required=True, help="folder of .h5ad files")
    parser.add_argument("--n_top", type=int, default=50)
    parser.add_argument("--keep_ribo_mito", action="store_true",
                        help="skip the ribosomal/mitochondrial filter (paper's original method)")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.adata_dir, "*.h5ad")))
    print(f"pooling {len(paths)} samples from {args.adata_dir}")

    top_genes, idx = select_benchmark_genes(paths, n_top=args.n_top,
                                            drop_ribo_mito=not args.keep_ribo_mito)


    print(f"\nselected {len(top_genes)} genes:")
    for g in top_genes:
        print(" ", g)

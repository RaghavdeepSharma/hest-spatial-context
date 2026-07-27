"""
One-off fix: select_benchmark_genes() now returns just the gene names
(no more positional index), since positions aren't safely comparable
across samples with different gene panels. This updates every script
that calls it, so they unpack a single value instead of a tuple, and
renames the now-unused 'gene_idx' variable to 'top_genes' everywhere
it's used afterward.

Run once from the project root:
    python fix_gene_idx_usage.py
"""
import re

FILES = [
    "scripts/run_baseline.py",
    "scripts/compare_gene_filters.py",
    "scripts/compare_context_vs_baseline.py",
]

# step 1: turn "X, gene_idx = select_benchmark_genes(" into "X = select_benchmark_genes("
TUPLE_UNPACK = re.compile(r"(\w+),\s*gene_idx\s*=\s*select_benchmark_genes\(")

# step 2: any gene_idx identifier left over (e.g. passed into
# run_cross_validation) becomes top_genes - word boundaries so this can
# never match inside a longer name like "some_gene_idx_other"
STANDALONE_GENE_IDX = re.compile(r"\bgene_idx\b")

for path in FILES:
    with open(path) as f:
        original = f.read()

    updated = TUPLE_UNPACK.sub(r"\1 = select_benchmark_genes(", original)
    updated = STANDALONE_GENE_IDX.sub("top_genes", updated)

    if updated == original:
        print(f"[no change] {path} - pattern not found, check manually")
        continue

    with open(path, "w") as f:
        f.write(updated)
    print(f"[updated]   {path}")

# Does spatial context help predict gene expression from histology?

This project builds on the HEST-Benchmark from the HEST-1k paper (Jaume et al.,
NeurIPS 2024, "HEST-1k: A Dataset for Spatial Transcriptomics and Histology
Image Analysis" - https://arxiv.org/abs/2406.16192, data/code at
https://github.com/mahmoodlab/hest).

The original benchmark predicts a spot's gene expression using only that one
spot's own H&E patch. I wanted to test something the paper doesn't do: does
knowing what the *surrounding* tissue looks like help predict a spot's
expression, on top of the patch itself?

I ran this on the PRAD (prostate cancer) task from hest-bench - 23 Visium
samples from 2 patients.

## Setup

- Data: `MahmoodLab/hest-bench` on HuggingFace, PRAD task only (~11GB)
- Patch encoder: ResNet50, ImageNet pretrained, frozen (this is the paper's
  "ResNet50 (IN)" baseline - no gated model access needed to run this)
- Regression: PCA (256 components) -> Ridge, same as the paper's main result
- Evaluation: patient-stratified cross validation using the splits shipped
  with hest-bench (2 folds for PRAD, since there are 2 patients) - a
  patient's spots never appear on both sides of a fold
- Metric: Pearson correlation per gene, averaged across genes, then averaged
  and reported as mean +/- std across folds

Everything ran on O2 (Harvard Medical School's HPC cluster), on CPU. This
task is small enough that I didn't end up needing a GPU job for it.

## What I changed vs. the paper

**1. Gene selection.** The paper ranks genes by variance of log1p counts and
keeps the top 50. When I ran that as written, the list was dominated by
ribosomal protein genes (RPS*/RPL*) and mitochondrial genes (MT-*) - these
are expressed at high levels in every cell regardless of tissue biology, so
they have large absolute variance without being biologically specific. I
added a filter that drops these before ranking, so the top 50 is actually
things like KLK3 (PSA), TMPRSS2, NKX3-1, MSMB - genes that mean something
specific for prostate tissue. This is a standard step in single-cell/spatial
analysis, I just added it here since the raw benchmark recipe doesn't do it.

Effect of this alone, same folds, same everything else:

| gene selection            | mean Pearson | std    |
|----------------------------|-------------:|-------:|
| original (ribo/mito kept)  | 0.3321       | 0.0227 |
| filtered (ribo/mito dropped)| 0.3666      | 0.0089 |

Filtering out housekeeping genes improved the average and noticeably
tightened the spread across folds. The 0.3321 number is also close to what
the paper itself reports for PRAD with ResNet50 (~0.31), which is a decent
sign the reimplementation is faithful before I started changing anything.

**2. Neighborhood context (the actual experiment).** For each spot, I found
its 6 nearest neighboring spots by pixel coordinate (6, because Visium spots
sit on a hexagonal grid, so 6 is the natural number of immediate neighbors),
averaged their embeddings, and concatenated that average onto the spot's own
embedding - so the regression sees the spot's own patch plus a summary of
what's happening around it, rather than replacing or diluting the spot's own
signal.

| variant                    | mean Pearson | std    |
|-----------------------------|-------------:|-------:|
| baseline (single patch)     | 0.3666       | 0.0089 |
| + neighborhood context      | 0.3716       | 0.0059 |

## Honest read on the result

Neighborhood context gave a small improvement (+0.005 mean Pearson) and it
went in the same direction on both folds, plus the spread across folds got
tighter. That's a decent sign it's a real, if modest, effect and not just
one fold getting lucky.

That said, I only have 2 folds here (2 patients), so I wouldn't call this a
strong result on its own - it's not enough data to rule out this being noise
that happened to land favorably. If I extend this to a task with more
patients (ccRCC has 24), that would be a much better test of whether this
holds up.

## Repo layout

src/
gene_selection.py - pool counts across samples, pick top 50 HVGs, ribo/mito filter
baseline_regression.py - PCA+Ridge regression, patient-stratified CV, Pearson scoring
neighbors.py - k-NN lookup on spot coordinates, neighborhood aggregation
scripts/
inspect_sample.py - sanity check one sample's patch/expression alignment
extract_embeddings.py - embed one sample's patches with ResNet50
extract_embeddings_batch.py - same, looped over every sample in a task
build_context_embeddings.py - build neighbor-aggregated embeddings for a task
run_baseline.py - run the baseline benchmark on a task
compare_gene_filters.py - baseline vs. ribo/mito-filtered gene selection
compare_context_vs_baseline.py - the main experiment: baseline vs. neighborhood context

## Reproducing this

```bash
conda create -n gitproj python=3.11
conda activate gitproj
pip install numpy pandas scipy scikit-learn h5py anndata scanpy
pip install torch torchvision timm huggingface_hub

hf auth login   # needs a free HuggingFace account + access request to
                # MahmoodLab/hest-bench (auto-approved)

hf download MahmoodLab/hest-bench --repo-type dataset --include "PRAD/*" --local-dir data

python scripts/extract_embeddings_batch.py --patches_dir data/PRAD/patches --out_dir data/PRAD/embeddings
python scripts/build_context_embeddings.py --task_dir data/PRAD --k 6
python scripts/compare_context_vs_baseline.py --task_dir data/PRAD
```

## Credit

All data, the original benchmark design, and the HEST-Library belong to the
HEST-1k authors (Jaume et al., Mahmood Lab, NeurIPS 2024), released under
CC BY-NC-SA 4.0. This repo is my own reimplementation of one piece of their
benchmark plus my own modifications on top of it, for a class/portfolio
project - not affiliated with the original authors.

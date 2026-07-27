#!/bin/bash
#SBATCH -J ccrcc_pipeline
#SBATCH -p short
#SBATCH -t 06:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -o logs/ccrcc_%j.out
#SBATCH -e logs/ccrcc_%j.err

# stop the whole job immediately if any single step fails, instead of
# silently continuing on to the next step with broken/missing data
set -euo pipefail

export CONDARC=/n/scratch/users/r/ras7303/rproj1/.condarc
source /n/app/conda/miniforge3/24.11.3-0/etc/profile.d/conda.sh
conda activate gitproj

cd /n/scratch/users/r/ras7303/rproj1/hest-spatial-context

echo "=== extracting embeddings ==="
python scripts/extract_embeddings_batch.py \
    --patches_dir data/CCRCC/patches \
    --out_dir data/CCRCC/embeddings \
    --batch_size 64

echo "=== building neighborhood-context embeddings ==="
python scripts/build_context_embeddings.py \
    --task_dir data/CCRCC \
    --k 6

echo "=== running baseline vs context comparison ==="
python scripts/compare_context_vs_baseline.py \
    --task_dir data/CCRCC \
    > results_context_vs_baseline_CCRCC.txt 2>&1

echo "=== done ==="

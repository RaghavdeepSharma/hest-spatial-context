"""
Run ResNet50 embedding extraction over every sample in a task's patches
folder, saving one output .npz per sample. Skips samples that already have
an output file, so re-running after a partial failure doesn't waste time
redoing finished samples.

Usage:
    python extract_embeddings_batch.py --patches_dir data/PRAD/patches --out_dir data/PRAD/embeddings
"""
import argparse
import glob
import os

import torch

from extract_embeddings import build_encoder, embed_patches, load_patch_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patches_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    model, transform = build_encoder(device)

    h5_paths = sorted(glob.glob(os.path.join(args.patches_dir, "*.h5")))
    print(f"found {len(h5_paths)} samples in {args.patches_dir}")

    for path in h5_paths:
        sample_id = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.out_dir, f"{sample_id}.npz")

        if os.path.exists(out_path):
            print(f"[skip] {sample_id} already done")
            continue

        print(f"[run ] {sample_id}")
        imgs, barcodes = load_patch_images(path)
        embeddings = embed_patches(imgs, model, transform, device, args.batch_size)

        import numpy as np
        np.savez(out_path, embeddings=embeddings, barcodes=barcodes)
        print(f"       -> {embeddings.shape} saved to {out_path}")


if __name__ == "__main__":
    main()

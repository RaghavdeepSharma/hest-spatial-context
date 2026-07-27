"""
Turn every H&E patch in a sample into a fixed-length embedding using a
frozen, ImageNet-pretrained ResNet50 - this is the paper's "ResNet50 (IN)"
baseline encoder. Nothing here is trained; the network is just used as a
fixed feature extractor, and its final classification layer is dropped so
we get the 2048-dim pooled feature vector instead of class scores.

Usage:
    python extract_embeddings.py --patches path/to/SAMPLE.h5 --out path/to/SAMPLE_emb.npz
"""
import argparse

import h5py
import numpy as np
import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def build_encoder(device):
    # num_classes=0 strips the final classification head, so calling the
    # model returns the pooled feature vector instead of 1000-way logits
    model = timm.create_model("resnet50.tv_in1k", pretrained=True, num_classes=0)
    model = model.eval().to(device)

    cfg = resolve_data_config({}, model=model)
    transform = create_transform(**cfg)
    return model, transform


def load_patch_images(h5_path):
    with h5py.File(h5_path, "r") as f:
        imgs = f["img"][:]
        barcodes_raw = f["barcode"][:]
    barcodes = np.array([b[0].decode() for b in barcodes_raw])
    return imgs, barcodes


def embed_patches(imgs, model, transform, device, batch_size=64):
    """imgs: (N, H, W, 3) uint8 array. Returns (N, embed_dim) float32."""
    from PIL import Image

    all_feats = []
    with torch.inference_mode():
        for start in range(0, len(imgs), batch_size):
            batch = imgs[start:start + batch_size]
            # timm's transform expects a PIL image, not a raw numpy array,
            # so each patch gets converted before stacking into a tensor
            tensors = [transform(Image.fromarray(im)) for im in batch]
            x = torch.stack(tensors).to(device)
            feats = model(x)
            all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patches", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    model, transform = build_encoder(device)
    imgs, barcodes = load_patch_images(args.patches)
    print(f"loaded {len(imgs)} patches, shape {imgs.shape}")

    embeddings = embed_patches(imgs, model, transform, device, args.batch_size)
    print(f"produced embeddings: {embeddings.shape}")

    # save both the embeddings and the barcodes together so downstream code
    # can always confirm which row belongs to which spot, same reasoning
    # as the alignment check in inspect_sample.py
    np.savez(args.out, embeddings=embeddings, barcodes=barcodes)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()

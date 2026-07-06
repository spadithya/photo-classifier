"""
Phase 2 — Build and visualize the PyTorch input pipeline.

In Phase 1 we looked at raw files. Phase 2 turns those files into batches of
tensors the model can train on. Three new ideas:

  1. transforms  — how one image file becomes a normalized 224x224 tensor
                   (defined in src/preprocess.py)
  2. Dataset     — an indexable collection of (image_tensor, label) pairs
                   (PhotoDataset in src/data.py)
  3. DataLoader  — batches, shuffles, and loads a Dataset
                   (assembled by make_dataloaders in src/data.py)

The reusable pipeline lives in src/ (so Phase 3+ share it). THIS script's job is
to build it, sanity-check the shapes, and visualize one real (augmented,
un-normalized) batch so you can confirm images and labels line up.

This script does not train anything. Run from the project root:
    python notebooks/02_dataloader.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")                 # headless-safe backend (works over SSH)
import matplotlib.pyplot as plt        # noqa: E402
import torch                           # noqa: E402

from src.categories import CATEGORIES, INDEX_TO_NAME  # noqa: E402
from src.data import make_dataloaders                 # noqa: E402
from src.preprocess import denormalize                # noqa: E402

PLOTS_DIR = PROJECT_ROOT / "notebooks" / "plots"
BATCH_SIZE = 32


def print_split_distribution(train_samples, val_samples):
    """Per-class counts in each split — confirms stratification kept balance."""
    def counts(samples):
        c = {name: 0 for name in CATEGORIES}
        for _, label in samples:
            c[INDEX_TO_NAME[label]] += 1
        return c

    tr, va = counts(train_samples), counts(val_samples)
    print("\n" + "=" * 40)
    print(f"{'CATEGORY':<14}{'TRAIN':>7}{'VAL':>6}")
    print("-" * 40)
    for name in CATEGORIES:
        print(f"{name:<14}{tr[name]:>7}{va[name]:>6}")
    print("-" * 40)
    print(f"{'TOTAL':<14}{len(train_samples):>7}{len(val_samples):>6}")
    print("=" * 40)


def visualize_batch(images, targets, n=16):
    """Save a grid of the first n images in a batch, un-normalized, with labels."""
    n = min(n, images.shape[0])
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            # denormalize -> CHW to HWC (what imshow wants) -> numpy
            img = denormalize(images[i]).permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_title(INDEX_TO_NAME[int(targets[i])], fontsize=9)

    fig.suptitle("One training batch (augmented + un-normalized)", fontsize=13)
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "02_batch.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved batch preview -> {out}")


def main():
    train_loader, val_loader, train_samples, val_samples = make_dataloaders(batch_size=BATCH_SIZE)
    print(f"Total usable photos: {len(train_samples) + len(val_samples)}")

    # --- Sanity check: pull ONE batch and inspect it ---
    images, targets = next(iter(train_loader))
    print(f"\nOne batch of images: {tuple(images.shape)}   (batch, channels, H, W)")
    print(f"One batch of labels: {tuple(targets.shape)}   (one label per image)")
    print(f"Pixel range after normalize: min={images.min():.2f}  max={images.max():.2f}")
    print("  (Note: NOT 0..1 — normalization shifts values negative/positive.)")

    print_split_distribution(train_samples, val_samples)
    visualize_batch(images, targets)

    print(f"\nBatches per epoch:  train={len(train_loader)}  val={len(val_loader)}")


if __name__ == "__main__":
    main()

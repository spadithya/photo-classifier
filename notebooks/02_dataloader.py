"""
Phase 2 — Build the PyTorch input pipeline (Dataset + DataLoader).

In Phase 1 we looked at raw files. Now we turn those files into batches of
tensors the model can train on. Three new ideas:

  1. transforms  — how one image file becomes a normalized 224x224 tensor
                   (defined once in src/preprocess.py, reused everywhere)
  2. Dataset     — an indexable collection of (image_tensor, label) pairs
  3. DataLoader  — batches, shuffles, and parallel-loads a Dataset

We also do a stratified train/validation split, then visualize one real
(augmented, un-normalized) batch to confirm images and labels line up.

This script still doesn't train anything — it only builds and sanity-checks
the pipeline. Run from the project root:

    python notebooks/02_dataloader.py
"""

from pathlib import Path
import sys

# Make src/ importable (same trick as Phase 1).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")                 # headless-safe backend (works over SSH)
import matplotlib.pyplot as plt        # noqa: E402
import torch                           # noqa: E402
from torch.utils.data import Dataset, DataLoader  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.categories import CATEGORIES, NAME_TO_INDEX, INDEX_TO_NAME  # noqa: E402
from src.preprocess import (  # noqa: E402
    build_train_transform,
    build_eval_transform,
    denormalize,
    load_image,
)

DATA_DIR = PROJECT_ROOT / "data" / "raw"
PLOTS_DIR = PROJECT_ROOT / "notebooks" / "plots"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".heic"}

BATCH_SIZE = 32
VAL_FRACTION = 0.2          # hold out 20% of each class for validation
SEED = 42                   # fixed so the split is identical every run


# ---------------------------------------------------------------------------
# 1. Index the files: build a list of (path, label) pairs
# ---------------------------------------------------------------------------
def build_sample_list():
    """Walk data/raw/<category>/ and return [(path, label_index), ...].

    The label index comes from NAME_TO_INDEX (src/categories.py), so the model
    learns OUR defined class order — NOT, say, alphabetical folder order, which
    is what you'd silently get from torchvision's ImageFolder. We also verify
    each file opens *now*, once, and skip broken ones — so training never
    crashes halfway through on a corrupt file.
    """
    samples = []
    skipped = 0
    for category in CATEGORIES:
        label = NAME_TO_INDEX[category]
        folder = DATA_DIR / category
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if not (path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
                continue
            if load_image(path) is None:
                skipped += 1
                continue
            samples.append((path, label))
    if skipped:
        print(f"Skipped {skipped} unreadable file(s).")
    return samples


# ---------------------------------------------------------------------------
# 2. A custom Dataset over those (path, label) pairs
# ---------------------------------------------------------------------------
class PhotoDataset(Dataset):
    """A list of (image_path, label) pairs, decoded and transformed on demand.

    We store PATHS, not decoded images, so we don't hold 450 images in RAM at
    once. __getitem__ opens ONE image, applies the transform, and returns
    (tensor, label). The DataLoader calls __getitem__ repeatedly to assemble a
    batch. A Dataset only needs two methods: __len__ and __getitem__.
    """

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = load_image(path)          # PIL RGB image
        tensor = self.transform(img)    # -> normalized 224x224 float tensor
        return tensor, label


# ---------------------------------------------------------------------------
# 3. Reporting / visualization helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Indexing photos in {DATA_DIR} ...")
    samples = build_sample_list()
    print(f"Total usable photos: {len(samples)}")

    # Stratified split: keep each class's proportion identical in train and val
    # (so 'food', with only 50, isn't accidentally starved in one split).
    labels = [label for _, label in samples]
    train_samples, val_samples = train_test_split(
        samples,
        test_size=VAL_FRACTION,
        stratify=labels,
        random_state=SEED,
    )

    # Train gets augmentation; validation gets the deterministic pipeline.
    train_ds = PhotoDataset(train_samples, build_train_transform())
    val_ds = PhotoDataset(val_samples, build_eval_transform())

    # DataLoaders wrap a Dataset to yield batches. shuffle=True on train so the
    # model doesn't see classes in a fixed order each epoch. num_workers=0 keeps
    # things simple and robust on Windows (worker subprocesses there need extra
    # care); we can raise it later if data loading becomes the bottleneck.
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

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

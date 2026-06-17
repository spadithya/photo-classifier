"""
Phase 1 — Inspect the photos.

Before we train anything, we look at the data. This is the single most
skipped and most important step in ML. A model can only ever be as good as
the data you feed it, so here we answer three questions:

  1. How many photos do I have per category?  (counts + class balance)
  2. Are any files broken / unreadable?         (data hygiene)
  3. What do the photos actually look like?      (a visual sample grid)

This script does NOT use the GPU or PyTorch. It only reads files and draws a
plot, so you can run it the moment your photos are sorted, even before the
torch install finishes.

Run it from the project root:
    python notebooks/01_inspect_photos.py
"""

from pathlib import Path
import sys

# --- Make `src/` importable -------------------------------------------------
# This file lives in notebooks/, but CATEGORIES lives in src/. We add the
# project root (one folder up from this file) to Python's import path so
# `from src.categories import ...` works no matter where you run the script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.categories import CATEGORIES  # noqa: E402  (import after sys.path edit)

import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

# --- Optional HEIC support --------------------------------------------------
# iPhones save photos as .heic, which Pillow can't read on its own. If the
# pillow-heif package is installed, this registers the decoder so .heic files
# load like any other image. If it's not installed, we just skip HEIC files
# and warn you at the end (you can convert them or `pip install pillow-heif`).
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False

# File extensions we treat as images.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".heic"}

DATA_DIR = PROJECT_ROOT / "data" / "raw"
PLOTS_DIR = PROJECT_ROOT / "notebooks" / "plots"

# How many sample images to show per category in the grid.
SAMPLES_PER_CLASS = 5
# Below this many photos, a class is too small to train well.
MIN_RECOMMENDED = 50


def list_images(folder: Path) -> list[Path]:
    """Return all image files directly inside `folder` (sorted, case-insensitive)."""
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_photos() -> dict[str, list[Path]]:
    """Walk each category folder and collect its image paths."""
    per_class = {}
    for category in CATEGORIES:
        per_class[category] = list_images(DATA_DIR / category)
    return per_class


def print_counts_table(per_class: dict[str, list[Path]]) -> None:
    """Print a tidy counts table and flag problems (too few / imbalance)."""
    counts = {c: len(paths) for c, paths in per_class.items()}
    total = sum(counts.values())

    print("\n" + "=" * 40)
    print(f"{'CATEGORY':<14}{'PHOTOS':>8}   STATUS")
    print("-" * 40)
    for category in CATEGORIES:
        n = counts[category]
        if n == 0:
            status = "EMPTY  <-- add photos"
        elif n < MIN_RECOMMENDED:
            status = f"thin (<{MIN_RECOMMENDED})"
        else:
            status = "ok"
        print(f"{category:<14}{n:>8}   {status}")
    print("-" * 40)
    print(f"{'TOTAL':<14}{total:>8}")
    print("=" * 40)

    # Class balance: the ratio between the biggest and smallest non-empty class.
    nonzero = [n for n in counts.values() if n > 0]
    if len(nonzero) >= 2:
        imbalance = max(nonzero) / min(nonzero)
        print(f"\nClass balance ratio (largest / smallest): {imbalance:.1f}x")
        if imbalance > 3:
            print("  ! Fairly imbalanced. The model will be biased toward big classes.")
            print("    Aim for roughly similar counts, or we handle it later with weighting.")


def try_open(path: Path):
    """Open an image and convert to RGB. Returns None if the file is broken."""
    try:
        img = Image.open(path)
        img.load()                 # force a full read so truncated files error here
        return img.convert("RGB")  # collapse RGBA / grayscale / palette to 3 channels
    except (UnidentifiedImageError, OSError):
        return None


def show_sample_grid(per_class: dict[str, list[Path]]) -> None:
    """Draw one row per category, with a few sample photos in each row."""
    rows = len(CATEGORIES)
    cols = SAMPLES_PER_CLASS
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))

    broken_files = []
    for r, category in enumerate(CATEGORIES):
        paths = per_class[category]
        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")
            if c == 0:
                # Label the row with the category name on the far left.
                ax.set_ylabel(category, rotation=0, ha="right", va="center",
                              fontsize=11, labelpad=40)
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])
            if c < len(paths):
                img = try_open(paths[c])
                if img is None:
                    broken_files.append(paths[c])
                    ax.set_title("BROKEN", fontsize=7, color="red")
                else:
                    ax.imshow(img)

    fig.suptitle("Sample photos per category", fontsize=14)
    fig.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / "01_sample_grid.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"\nSaved sample grid -> {out_path}")

    if broken_files:
        print(f"\n! {len(broken_files)} unreadable file(s) found:")
        for p in broken_files:
            print(f"    {p}")

    plt.show()  # opens a window; close it to end the script


def main() -> None:
    print(f"Looking for photos in: {DATA_DIR}")
    if not HEIC_SUPPORTED:
        print("Note: HEIC support is OFF (pillow-heif not installed). "
              "iPhone .heic files will be skipped/counted but not displayed.")

    per_class = count_photos()
    print_counts_table(per_class)

    total = sum(len(p) for p in per_class.values())
    if total == 0:
        print("\nNo photos found yet. Sort some into data/raw/<category>/ and re-run.")
        return

    show_sample_grid(per_class)


if __name__ == "__main__":
    main()

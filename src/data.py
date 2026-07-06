"""Data pipeline — build train/val DataLoaders from data/raw/.

Phase 2 prototyped this logic inline; it lives here now so every training and
evaluation script shares ONE pipeline (the same single-source-of-truth idea as
categories.py and preprocess.py). Import make_dataloaders() and go.
"""

from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from src.categories import CATEGORIES, NAME_TO_INDEX
from src.preprocess import build_train_transform, build_eval_transform, load_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".heic"}


def build_sample_list(data_dir: Path = DATA_DIR):
    """Walk data/raw/<category>/ and return [(path, label_index), ...].

    Labels come from NAME_TO_INDEX so the model learns OUR class order (not
    alphabetical). Broken files are verified and skipped up front, so training
    never crashes mid-epoch on a corrupt image.
    """
    samples = []
    skipped = 0
    for category in CATEGORIES:
        label = NAME_TO_INDEX[category]
        folder = data_dir / category
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


class PhotoDataset(Dataset):
    """A list of (image_path, label) pairs, decoded and transformed on demand.

    Stores PATHS, not decoded images, so RAM stays low. __getitem__ opens one
    image, applies the transform, and returns (tensor, label). The DataLoader
    calls it repeatedly to assemble a batch.
    """

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = load_image(path)
        tensor = self.transform(img)
        return tensor, label


def make_dataloaders(batch_size: int = 32, val_fraction: float = 0.2, seed: int = 42):
    """Build the train and validation DataLoaders.

    Returns (train_loader, val_loader, train_samples, val_samples). The raw
    sample lists come back too so callers can report the split distribution.

    Train gets augmentation; val gets the deterministic pipeline. The split is
    stratified (each class's proportion preserved) and seeded (reproducible).
    """
    samples = build_sample_list()
    labels = [label for _, label in samples]

    train_samples, val_samples = train_test_split(
        samples,
        test_size=val_fraction,
        stratify=labels,
        random_state=seed,
    )

    train_ds = PhotoDataset(train_samples, build_train_transform())
    val_ds = PhotoDataset(val_samples, build_eval_transform())

    # num_workers=0: single-process loading, robust on Windows. Raise later if
    # data loading becomes the training bottleneck.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, train_samples, val_samples

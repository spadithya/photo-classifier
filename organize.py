"""
CLI tool — organize a local folder of photos into per-category subfolders.

The Streamlit app is the "try it in 30 seconds" demo; THIS is the workhorse for
your real library. It streams photos from disk one at a time (constant memory),
so it scales to thousands of files.

Example:
    python organize.py --input "C:\\Users\\Shady\\Pictures\\unsorted" \\
                       --output "C:\\Users\\Shady\\Pictures\\sorted" \\
                       --mode copy --threshold 0.7

  --mode copy|move   copy preserves originals; move relocates them (decisive)
  --threshold        photos below this confidence land in unsure/ for review
"""

from pathlib import Path
import argparse
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from src.inference import load_model, predict, DEFAULT_MODEL_PATH
from src.preprocess import build_eval_transform, load_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".heic"}


def parse_args():
    p = argparse.ArgumentParser(description="Sort photos into category folders with a trained model.")
    p.add_argument("--input", required=True, type=Path, help="Folder of photos to organize.")
    p.add_argument("--output", required=True, type=Path, help="Where sorted folders are created.")
    p.add_argument("--mode", choices=["copy", "move"], default="copy",
                   help="copy keeps originals (default); move relocates them.")
    p.add_argument("--threshold", type=float, default=0.7,
                   help="Confidence below this -> unsure/ (default 0.7).")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH,
                   help="Path to the model weights.")
    return p.parse_args()


def gather_images(input_dir: Path):
    """Recursively find every image file under input_dir."""
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def unique_dest(folder: Path, name: str) -> Path:
    """Return a path in `folder` for `name`, adding _1, _2, ... if it exists.

    Prevents two source files with the same name from overwriting each other.
    """
    dest = folder / name
    if not dest.exists():
        return dest
    stem, suffix = Path(name).stem, Path(name).suffix
    i = 1
    while (folder / f"{stem}_{i}{suffix}").exists():
        i += 1
    return folder / f"{stem}_{i}{suffix}"


def place(src: Path, folder: Path, mode: str):
    """Copy or move `src` into `folder`, creating it and avoiding collisions."""
    folder.mkdir(parents=True, exist_ok=True)
    dest = unique_dest(folder, src.name)
    if mode == "copy":
        shutil.copy2(src, dest)     # copy2 preserves timestamps/metadata
    else:
        shutil.move(str(src), str(dest))


def main():
    args = parse_args()

    if not args.input.is_dir():
        print(f"Input folder not found: {args.input}")
        return
    if not args.model.exists():
        print(f"Model not found: {args.model}\n"
              f"Copy your weights there first, e.g.:\n"
              f"  Copy-Item models\\baseline_frozen.pt models\\photo_classifier.pt")
        return

    paths = gather_images(args.input)
    if not paths:
        print(f"No images found under {args.input}")
        return

    model, device = load_model(args.model)
    transform = build_eval_transform()      # build once, reuse for every image
    print(f"Model on {device}. Organizing {len(paths)} photo(s) "
          f"({args.mode}, threshold {args.threshold}) -> {args.output}")

    counts = {}
    for path in tqdm(paths, desc="Sorting", unit="img"):
        image = load_image(path)
        if image is None:
            # Unreadable file — don't silently lose it; set it aside.
            place(path, args.output / "unreadable", args.mode)
            counts["unreadable"] = counts.get("unreadable", 0) + 1
            continue

        top_name, top_prob = predict(model, image, device, transform=transform)[0]
        folder = top_name if top_prob >= args.threshold else "unsure"
        place(path, args.output / folder, args.mode)
        counts[folder] = counts.get(folder, 0) + 1

    print("\nDone. Summary:")
    for folder in sorted(counts):
        print(f"  {folder:<12} {counts[folder]:>5}")


if __name__ == "__main__":
    main()

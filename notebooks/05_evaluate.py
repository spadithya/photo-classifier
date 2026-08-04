"""
Phase 5 — Evaluate the final model + error analysis.

Phase 3/4 printed a single accuracy number per epoch. That tells you *how much*
the model gets right, but not *where* it's weak. Phase 5 digs in:

  1. Per-class precision / recall / F1   (which classes are strong or weak)
  2. Confusion matrix                     (which classes get mistaken for which)
  3. The actual misclassified images      (look at what it gets wrong, and why)

That last step — looking at the specific wrong images — is the single most
useful habit in applied ML. Aggregate metrics hide the story; the mistakes tell
it. We evaluate the FINAL model we chose to ship: the frozen baseline.

Run from the project root:
    python notebooks/05_evaluate.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")                 # headless-safe (works over SSH)
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
import seaborn as sns                  # noqa: E402
import torch                           # noqa: E402
import torch.nn.functional as F        # noqa: E402
from sklearn.metrics import confusion_matrix, classification_report  # noqa: E402

from src.categories import CATEGORIES, INDEX_TO_NAME  # noqa: E402
from src.data import make_dataloaders                 # noqa: E402
from src.preprocess import load_image                 # noqa: E402
from src.model import build_model, get_device         # noqa: E402

PLOTS_DIR = PROJECT_ROOT / "notebooks" / "plots"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "baseline_frozen.pt"        # the model we chose to ship

BATCH_SIZE = 32


@torch.no_grad()   # decorator: disable gradient tracking for this whole function
def collect_predictions(model, loader, device):
    """Run the model over `loader` and gather predictions for every image.

    Returns three aligned lists (same order as the loader, which is unshuffled):
      preds  — predicted class index per image
      probs  — the model's confidence (softmax prob) in that prediction
      trues  — the true class index per image
    """
    model.eval()
    preds, probs, trues = [], [], []
    for images, targets in loader:
        images = images.to(device)
        logits = model(images)                     # raw scores [B, num_classes]
        # softmax turns raw scores into probabilities that sum to 1 per image.
        p = F.softmax(logits, dim=1)
        conf, pred = p.max(dim=1)                  # highest prob + its class index
        preds.extend(pred.cpu().tolist())
        probs.extend(conf.cpu().tolist())
        trues.extend(targets.tolist())
    return preds, probs, trues


def plot_confusion_matrix(trues, preds):
    """Heatmap: rows = true class, columns = predicted class.

    The diagonal is correct predictions; any off-diagonal cell is a confusion
    (true class on that row was predicted as the class on that column).
    """
    cm = confusion_matrix(trues, preds, labels=range(len(CATEGORIES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=CATEGORIES, yticklabels=CATEGORIES, ax=ax,
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (validation set)")
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "05_confusion_matrix.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix -> {out}")


def show_errors(val_samples, preds, probs, trues):
    """Save a grid of every misclassified image with true/pred/confidence.

    val_samples[i] is aligned with preds[i]/trues[i] because the val loader is
    unshuffled. We show the ORIGINAL photo (not the normalized tensor) so it's
    readable.
    """
    wrong = [i for i in range(len(trues)) if preds[i] != trues[i]]
    if not wrong:
        print("\nNo misclassified images — the model got the whole val set right.")
        return

    print(f"\n{len(wrong)} misclassified image(s):")
    for i in wrong:
        path, _ = val_samples[i]
        print(f"  {path.name:35s}  true={INDEX_TO_NAME[trues[i]]:10s} "
              f"pred={INDEX_TO_NAME[preds[i]]:10s} conf={probs[i]:.2f}")

    cols = min(4, len(wrong))
    rows = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for slot, i in enumerate(wrong):
        path, _ = val_samples[i]
        ax = axes[slot // cols][slot % cols]
        img = load_image(path)
        if img is not None:
            ax.imshow(img)
        ax.set_title(
            f"true: {INDEX_TO_NAME[trues[i]]}\npred: {INDEX_TO_NAME[preds[i]]} ({probs[i]:.0%})",
            fontsize=9, color="darkred",
        )

    fig.suptitle("Misclassified validation images", fontsize=13)
    fig.tight_layout()
    out = PLOTS_DIR / "05_errors.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved error grid -> {out}")


def main():
    device = get_device()
    print(f"Device: {device}")

    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}. Run Phase 3 first.")
        return

    # Rebuild the SAME split (same seed) so the val set matches training's.
    _, val_loader, _, val_samples = make_dataloaders(batch_size=BATCH_SIZE)

    # Build the architecture and load our shipped weights.
    model = build_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Loaded {MODEL_PATH.name}\n")

    preds, probs, trues = collect_predictions(model, val_loader, device)

    acc = np.mean(np.array(preds) == np.array(trues))
    print(f"Validation accuracy: {acc:.3f}  ({int(acc * len(trues))}/{len(trues)})\n")

    # Per-class precision / recall / F1. digits=3 for readable decimals.
    print(classification_report(trues, preds, target_names=CATEGORIES, digits=3))

    plot_confusion_matrix(trues, preds)
    show_errors(val_samples, preds, probs, trues)


if __name__ == "__main__":
    main()

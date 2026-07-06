"""
Phase 3 — Feature-extraction baseline (frozen ResNet).

Transfer learning, step one. We take a ResNet18 already trained on ImageNet,
FREEZE all its layers, and train ONLY a fresh classification head on our 6
categories. The frozen backbone acts as a fixed feature extractor; the tiny new
head learns to map those features to our classes.

This is fast (only the head's ~3k weights train) and gives us a BASELINE
accuracy. In Phase 4 we'll unfreeze the network and fine-tune it to (hopefully)
beat this number.

Run from the project root:
    python notebooks/03_baseline_frozen.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")                 # headless-safe (works over SSH)
import matplotlib.pyplot as plt        # noqa: E402
import torch                           # noqa: E402
import torch.nn as nn                  # noqa: E402

from src.data import make_dataloaders  # noqa: E402
from src.model import build_model, get_device  # noqa: E402

PLOTS_DIR = PROJECT_ROOT / "notebooks" / "plots"
MODELS_DIR = PROJECT_ROOT / "models"

EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3   # the head trains from scratch, so a normal LR is fine


def run_one_epoch(model, loader, criterion, optimizer, device, train):
    """One full pass over `loader`. Updates weights only when train=True.

    Returns (average_loss, accuracy) for the pass.
    """
    # train() vs eval() switches layers like BatchNorm/Dropout into the right
    # mode. It does NOT by itself turn gradients on/off — that's the with-block.
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    # Build the gradient graph only during training. At validation we don't need
    # gradients, so no_grad saves memory and time.
    with torch.set_grad_enabled(train):
        for images, targets in loader:
            # Move this batch onto the GPU (or CPU). The model is already there.
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)              # [B, num_classes] raw scores (logits)
            loss = criterion(outputs, targets)   # CrossEntropyLoss compares to true labels

            if train:
                optimizer.zero_grad()            # clear gradients from the previous step
                loss.backward()                  # backprop: compute new gradients
                optimizer.step()                 # nudge the trainable weights

            # Accumulate stats. loss.item() is the batch's mean loss; multiply by
            # batch size so batches of different sizes are weighted correctly.
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)        # index of highest score = predicted class
            correct += (preds == targets).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def plot_history(history):
    """Save side-by-side loss and accuracy curves over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, history["train_loss"], "-o", label="train")
    ax_loss.plot(epochs, history["val_loss"], "-o", label="val")
    ax_loss.set_title("Loss"); ax_loss.set_xlabel("epoch"); ax_loss.legend()

    ax_acc.plot(epochs, history["train_acc"], "-o", label="train")
    ax_acc.plot(epochs, history["val_acc"], "-o", label="val")
    ax_acc.set_title("Accuracy"); ax_acc.set_xlabel("epoch"); ax_acc.legend()

    fig.suptitle("Phase 3 — frozen ResNet18 baseline", fontsize=13)
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "03_baseline_curves.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved training curves -> {out}")


def main():
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, _, _ = make_dataloaders(batch_size=BATCH_SIZE)

    # Frozen backbone + fresh head, moved onto the GPU.
    model = build_model(freeze_backbone=True).to(device)

    # Confirm only the head trains.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    criterion = nn.CrossEntropyLoss()
    # Optimize ONLY the parameters that require gradients (the head). Handing the
    # optimizer frozen params would be pointless.
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=LEARNING_RATE
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"Epoch {epoch:2d}/{EPOCHS}   "
              f"train: loss {tr_loss:.3f}  acc {tr_acc:.3f}    "
              f"val: loss {va_loss:.3f}  acc {va_acc:.3f}")

        # Keep the best model by validation accuracy (not the last epoch's).
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), MODELS_DIR / "baseline_frozen.pt")

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Saved best model -> {MODELS_DIR / 'baseline_frozen.pt'}")
    plot_history(history)


if __name__ == "__main__":
    main()

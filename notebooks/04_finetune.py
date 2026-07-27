"""
Phase 4 — Full fine-tuning (unfreeze the whole network).

In Phase 3 the backbone was frozen and only the head learned. Now we UNFREEZE
everything and train the entire ResNet18 on your photos, so its edge/texture/
part detectors can adapt slightly to *your* categories instead of ImageNet's.

Four things change from Phase 3 (diff this file against 03 to see them):

  1. freeze_backbone=False   -> every layer is trainable (~11.2M params, not 3k)
  2. WARM START              -> we load the Phase 3 head weights first, so we
                                don't start fine-tuning with a random head
                                (random-head gradients could wreck the backbone)
  3. A MUCH smaller LR       -> 1e-4 instead of 1e-3. We're nudging weights that
                                are already good, not learning from scratch.
  4. MIXED PRECISION (AMP)   -> run most math in float16 to save GPU memory and
                                go faster on your 6 GB card (the float32->float16
                                lever mentioned back in Phase 3).

Run from the project root:
    python notebooks/04_finetune.py
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

EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-4        # 10x smaller than Phase 3 — fine-tuning, not scratch


def run_one_epoch(model, loader, criterion, optimizer, scaler, device, train, use_amp):
    """One pass over `loader`. Updates weights only when train=True.

    Same skeleton as Phase 3, plus mixed precision (autocast + GradScaler).
    Returns (average_loss, accuracy).
    """
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)

            # autocast: inside this block PyTorch automatically runs suitable ops
            # in float16 (half precision) instead of float32 — less memory, faster
            # on NVIDIA GPUs. It keeps numerically-sensitive ops in float32 itself.
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            if train:
                optimizer.zero_grad()
                # float16 gradients can "underflow" to zero. GradScaler multiplies
                # the loss up before backward(), then unscales before the step, so
                # tiny gradients survive. On CPU (use_amp=False) it's a no-op.
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
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

    fig.suptitle("Phase 4 — fine-tuned ResNet18", fontsize=13)
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "04_finetune_curves.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved training curves -> {out}")


def main():
    device = get_device()
    use_amp = device.type == "cuda"       # mixed precision only helps on GPU
    print(f"Device: {device}   mixed precision: {use_amp}")

    train_loader, val_loader, _, _ = make_dataloaders(batch_size=BATCH_SIZE)

    # Unfrozen network: every layer is trainable now.
    model = build_model(freeze_backbone=False).to(device)

    # WARM START: load the Phase 3 head (and unchanged backbone) as our starting
    # point. baseline_frozen.pt is a full state_dict, so this fills in the trained
    # head instead of leaving it random. map_location=device puts tensors on the
    # right device as they load.
    ckpt = MODELS_DIR / "baseline_frozen.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print("Warm-started from baseline_frozen.pt")
    else:
        print("No baseline checkpoint found — starting head from random.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    criterion = nn.CrossEntropyLoss()
    # Now the optimizer gets the WHOLE model's parameters, not just the head.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # The GradScaler partners with autocast for mixed-precision training.
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_one_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True, use_amp=use_amp)
        va_loss, va_acc = run_one_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False, use_amp=use_amp)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"Epoch {epoch:2d}/{EPOCHS}   "
              f"train: loss {tr_loss:.3f}  acc {tr_acc:.3f}    "
              f"val: loss {va_loss:.3f}  acc {va_acc:.3f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), MODELS_DIR / "finetuned.pt")

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Saved best model -> {MODELS_DIR / 'finetuned.pt'}")
    plot_history(history)


if __name__ == "__main__":
    main()

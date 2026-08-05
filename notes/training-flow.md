# Training flow (the algorithm)

The end-to-end pipeline that turns a folder of photos into a shipped model, then
the inner loop that actually does the learning.

---

## Big picture: Phases 1 → 5

```
data/raw/<category>/*.jpg
        │
        ▼
[Phase 1] inspect            count per class, check balance, find broken files,
 01_inspect_photos.py        save a sample grid.  (no model, no GPU)
        │
        ▼
[Phase 2] build pipeline     files → (path, label) list → stratified 80/20 split
 02_dataloader.py            → PhotoDataset (+transforms) → DataLoader (batches).
        │                     Sanity-check batch shape; visualize one batch.
        ▼
[Phase 3] frozen baseline    ResNet18(ImageNet) → freeze backbone → new head
 03_baseline_frozen.py       Linear(512→6) → train ONLY the head (3,078 params).
        │                     Save best-by-val → models/baseline_frozen.pt.
        │                     Result: 97.8% val.
        ▼
[Phase 4] fine-tune          unfreeze all (11.2M) → warm-start from baseline →
 04_finetune.py              tiny LR (1e-4) + AMP → train whole net.
        │                     Result: OVERFIT (95.5%). Not shipped.
        ▼
[Phase 5] evaluate           load shipped model → predict val set →
 05_evaluate.py              accuracy + precision/recall/F1 + confusion matrix +
        │                     look at the actual misclassified images.
        ▼
   models/photo_classifier.pt   (copy of baseline_frozen.pt — the chosen model)
```

Key decision: **the frozen baseline (Phase 3) is the shipped model**, because
Phase 4's full fine-tune overfit on only 356 training images.

---

## Data pipeline (Phase 2), step by step

```
1. build_sample_list()
     for each category in CATEGORIES:
         label = NAME_TO_INDEX[category]        # my order, not alphabetical
         for each image file in data/raw/<category>/:
             if it opens ok: append (path, label)
             else: skip (count as broken)
2. train_test_split(samples, test_size=0.2, stratify=labels, seed=42)
     → train_samples, val_samples   (each class's ratio preserved)
3. PhotoDataset(train_samples, train_transform)   # augmentation
   PhotoDataset(val_samples,   eval_transform)     # deterministic
4. DataLoader(train_ds, batch_size=32, shuffle=True)
   DataLoader(val_ds,   batch_size=32, shuffle=False)
```
- Store paths, decode on demand (low RAM).
- `__getitem__(i)`: load image → apply transform → return `(tensor, label)`.

---

## Model build (Phase 3 / 4)

```
build_model(freeze_backbone):
    model = resnet18(weights=IMAGENET1K_V1)     # architecture + pretrained weights
    if freeze_backbone:                          # Phase 3
        every param.requires_grad = False        # lock the backbone
    in_features = model.fc.in_features           # = 512
    model.fc = nn.Linear(512, 6)                 # fresh head (requires_grad=True)
```
- Freeze BEFORE swapping the head → only the head trains (Phase 3).
- `freeze_backbone=False` (Phase 4) → everything trainable (~11.2M params).

---

## The training loop (inner algorithm)

```
device = cuda if available else cpu
model  = build_model(...).to(device)
criterion = CrossEntropyLoss
optimizer = Adam(trainable params, lr)
scaler    = GradScaler          # Phase 4 only (AMP)
best_val_acc = 0

for epoch in 1..EPOCHS:

    # ---- TRAIN pass (updates weights) ----
    model.train()
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        with autocast:                     # Phase 4 (float16 where safe)
            outputs = model(images)        # forward → logits [B,6]
            loss    = criterion(outputs, targets)
        optimizer.zero_grad()              # clear old gradients
        loss.backward()                    # backprop → gradients
        optimizer.step()                   # nudge weights
        # (Phase 4: scaler.scale(loss).backward(); scaler.step(opt); scaler.update())
        accumulate loss (× batch size) and correct-count

    # ---- VALIDATION pass (measures only) ----
    model.eval()
    with no_grad():
        for images, targets in val_loader:
            move to device
            outputs = model(images)
            accumulate val loss and val accuracy

    log(train_loss, train_acc, val_loss, val_acc)

    if val_acc > best_val_acc:              # keep the BEST epoch, not the last
        best_val_acc = val_acc
        torch.save(model.state_dict(), checkpoint)

plot loss & accuracy curves
```

### Why each piece
- **`model.train()` / `model.eval()`** — switch BatchNorm/Dropout modes.
- **`no_grad()` on validation** — no learning, just scoring (faster, less memory).
- **`zero_grad → backward → step`** — gradients accumulate by default, so clear
  them each step; backward computes them; step applies them.
- **loss × batch size, then ÷ total** — honest per-image average across unequal
  batch sizes.
- **best-by-val saving** — the shipped file is the best epoch; later (possibly
  overfit) epochs can't corrupt it.

### One epoch in numbers (this project)
```
356 train images ÷ 32 = 12 batches   → 12 forward+backward+step iterations
 89 val   images ÷ 32 =  3 batches   → 3 forward-only iterations
```

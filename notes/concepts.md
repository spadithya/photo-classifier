# Concepts

Everything I learned, grouped by topic. Examples use this project's numbers:
6 classes, ResNet18, 512-feature vector, ~445 photos (356 train / 89 val).

---

## 1. Transfer learning

### ImageNet (a dataset)
- ~1.2 million labeled photos across 1,000 categories.
- Networks trained on it learn **generic visual features** (edges → textures →
  object parts) that transfer to almost any image task.
- "Pretrained on ImageNet" = I inherit that visual machinery instead of learning
  it from my 445 photos.

### ResNet (an architecture)
- A CNN design (2015). The number = layer count (ResNet**18** = 18 weight layers).
- Key idea: the **residual / skip connection**, `output = F(x) + x`. The `+ x`
  lets gradients flow back through very deep networks without vanishing, so deep
  nets actually train.
- I use ResNet18 (smallest, ~11.2M params) because my dataset is tiny and my GPU
  has 6 GB.

### Feature extraction vs fine-tuning
- **Feature extraction (Phase 3):** freeze the backbone, train only a new head.
  Backbone acts as a fixed "fingerprint generator." Few params, fast, safe.
- **Fine-tuning (Phase 4):** unfreeze the backbone too, train everything with a
  small learning rate so features adapt to my data. More capacity → more
  overfitting risk on small data.

### The classification head
- Every classifier = **backbone** (image → feature vector) + **head** (features →
  class scores).
- ResNet18's head is `model.fc`, originally `Linear(512 → 1000)`.
- I replace it with `Linear(512 → 6)`. The backbone does the hard work of
  *understanding* the image; the head just *names* it.

### Features (in a CNN)
- The numbers the network computes to represent an image, learned automatically.
- Hierarchy by depth: edges/colors → textures/patterns → object parts → a final
  **512-number fingerprint** per image.
- "Freezing the features" = locking the weights that compute them.

---

## 2. Weights, tensors, parameters, saved models

### Weight
- A single learnable number. A neuron does
  `output = in₁·w₁ + in₂·w₂ + … + bias`. The `wᵢ` are weights; `bias` is added at
  the end. "Learning" = nudging these numbers to reduce loss.

### Weight tensor
- Millions of weights aren't stored loose — they're grouped into **tensors**
  (multi-dimensional arrays). One layer's weights = one tensor.
- Head example: weight tensor shape `(6, 512)` = 3,072 numbers, bias `(6,)` = 6.
  Total **3,078 parameters**.

### Neuron / fully-connected wiring (512 → 6)
- The head has **6 neurons** (one per class). Each has its **own** 512 weights +
  1 bias.
- **Fully connected** = every one of the 512 inputs wires to every one of the 6
  neurons → 6×512 = 3,072 "wires" (each wire = one weight).
- A big positive weight = "this feature pushes toward my class"; near zero =
  "ignore it"; negative = "pushes away." Training tunes the *strength* of each
  wire; the wiring pattern (every-to-every) is fixed.

### requires_grad — frozen vs trainable
- Every parameter has a flag `requires_grad`:
  - `True` → trainable (optimizer updates it during `optimizer.step()`).
  - `False` → **frozen** (no gradient computed, never changes).
- Trick used in `build_model`: load ResNet → **freeze all** (`requires_grad=False`)
  → **then replace the head**. New layers default to `requires_grad=True`, and
  because the freeze ran *before* the swap, only the fresh head is trainable.
  (Reverse the order and you'd freeze the head too → nothing learns.)

### A saved model = a `state_dict`
- `torch.save(model.state_dict(), ...)` saves an ordered dict `{name: tensor}` of:
  - **weights + biases** (learned parameters), and
  - **buffers** — non-learned tensors the layers need, mainly BatchNorm's
    `running_mean` / `running_var`.
- It contains **only numbers, no code**. To use it I must (1) rebuild the
  architecture in code, then (2) `load_state_dict` to pour the numbers in.
- Model size ≈ 11.2M params × 4 bytes ≈ **45 MB** — small enough to commit to
  GitHub and host on a free tier.

---

## 3. The data pipeline

### Tensor shapes and dtypes
- One batch of images: `(32, 3, 224, 224)` = (batch, channels, H, W), `float32`.
- One batch of labels (`targets`): `(32,)`, `int64` (`torch.long`) — class indices.
- `images` = what the model looks at; `targets` = the correct answers, only used
  at the loss step (the model never sees them).

### Eval transform (deterministic — validation, app, CLI)
```
Resize(256)        # short side → 256 px, keeps aspect ratio (no stretching)
CenterCrop(224)    # central 224×224 square (ResNet's required input size)
ToTensor()         # 0–255 uint8 HWC → float 0–1 CHW (also transposes axes)
Normalize(mean,std)# per channel: (pixel − mean) / std, ImageNet stats
```
- `ToTensor` does two jobs: rescale 0–255 → 0–1, and reorder **HWC → CHW**
  (PyTorch wants channels first).
- `Normalize` uses ImageNet's mean `[0.485,0.456,0.406]` / std `[0.229,0.224,0.225]`
  because ResNet was trained on data normalized this way. After it, pixels are no
  longer 0–1 — they span roughly −2.1 … +2.6 (expected, not a bug).

### Train transform (augmentation — training only)
```
RandomResizedCrop(224, scale=(0.7,1.0))  # random zoom + crop + position
RandomHorizontalFlip()                    # 50% mirror left↔right
ColorJitter(0.2, 0.2, 0.2)                # ±20% brightness/contrast/saturation
ToTensor(); Normalize(...)                # same as eval
```
- `scale=(0.7,1.0)` = fraction of the image **area** the crop keeps. `1.0` ≈ whole
  image; `0.7` = keep 70% → looks zoomed in (~1.2× linear). Lower = more zoom.
- New random values every fetch, so the model rarely sees identical pixels twice —
  fakes a bigger dataset and fights overfitting.
- Horizontal (not vertical) flip: a mirrored photo is still realistic; upside-down
  isn't. Augmentation is **training only** — eval must be deterministic.

### Dataset and DataLoader
- **Dataset** (`PhotoDataset`) = indexable list of `(image_tensor, label)`. Needs
  only `__len__` and `__getitem__`. Stores **paths**, decodes one image on demand
  (low RAM).
- **DataLoader** wraps a Dataset → yields **batches**, **shuffles** train each
  epoch, can parallel-load.
- Labels come from `NAME_TO_INDEX` (my defined order), NOT `ImageFolder`'s
  alphabetical order — so the model learns the order `categories.py` declares.
- **Stratified split**: `train_test_split(..., stratify=labels)` keeps each
  class's proportion equal in train and val (so `food` with only 50 isn't starved).

---

## 4. ResNet internals

### Named parts (image flows top → bottom)
```
conv1 → bn1 → relu → maxpool → layer1 → layer2 → layer3 → layer4 → avgpool → fc
```
- `conv1` = first conv layer (one small piece, not the whole net).
- `fc` = **fully connected** — the final head.
- `layer1`–`layer4` = 4 **stages**, each a group of residual blocks (not single
  layers).

### The "18"
- Counts weight layers: `conv1` (1) + 4 stages × (2 blocks × 2 convs) = 16 + `fc`
  (1) = **18**. BatchNorm/ReLU/pool aren't counted.
- So `layer4` is the **last stage**, not "layer 4 of 18."

### Resolution vs channels vs neurons (I mixed these up)
| Stage  | Resolution (H×W) | Channels |
|--------|------------------|----------|
| layer1 | 56×56            | 64       |
| layer2 | 28×28            | 128      |
| layer3 | 14×14            | 256      |
| layer4 | 7×7              | **512**  |
- Resolution **shrinks**, channels **grow** — the core CNN pattern (trade spatial
  detail for more high-level concepts).
- A **channel** (feature map) = one filter's response across the image. NOT a
  neuron. "Neurons" only applies to the 6-unit head.
- `avgpool` averages each of the 512 channels' 7×7 grid → the 512-number vector
  fed to the head.

### state_dict key naming
- Keys are dotted paths: `layer2.0.conv1.weight` = stage `layer2`, block `0`, conv
  `conv1`, its weight tensor.
- One tensor per conv; BatchNorm contributes `weight` (γ), `bias` (β),
  `running_mean`, `running_var`; head is `fc.weight` + `fc.bias`.
- Saved **per weight-bearing layer**, organized by name — that's why loading
  requires the matching architecture (names must line up).

### BatchNorm
- Sits after (almost) every conv: `conv → batchnorm → relu`. ~17 of them in
  ResNet18 (not just `bn1`).
- Per channel, per batch: `(x − mean) / sqrt(var)`, then `× γ + β` (two learnable
  knobs). Keeps each layer's outputs in a tame range → faster, stabler training.
- `running_mean` / `running_var` = buffers averaged over training, used at
  inference. This is the main reason `model.train()` vs `model.eval()` matters:
  train mode uses the batch's stats (and updates the running ones); eval mode uses
  the stored ones, so a single image classifies consistently.

---

## 5. Training mechanics

### Epoch
- One complete pass through the entire training set. Here: 356 images ÷ 32 =
  12 batches = 1 epoch. I ran 10 (Phase 3) / 15 (Phase 4).

### The core loop (five lines every framework wraps)
```python
outputs = model(images)              # forward → 6 logits per image
loss    = criterion(outputs, targets)# CrossEntropyLoss: how wrong (lower better)
optimizer.zero_grad()                # clear last step's gradients
loss.backward()                      # backprop → gradient for each weight
optimizer.step()                     # nudge trainable weights
```
- **Logits** = raw scores. `CrossEntropyLoss` applies softmax internally, so the
  model outputs raw scores, not probabilities.
- **`argmax(dim=1)`** = predicted class = highest score. Compare to `targets` for
  accuracy.
- **Learning rate (LR)** = step size. Phase 3 head-from-scratch: `1e-3`. Phase 4
  fine-tune: `1e-4` (10× smaller — nudging already-good weights, not learning from
  scratch; rule of thumb: 10–100× smaller than from-scratch).

### `.to(device)`
- PyTorch does NOT auto-use the GPU. Move the **model once**
  (`build_model().to(device)`) and **each batch** (`images.to(device)`,
  `targets.to(device)`) because the DataLoader produces CPU tensors.
- All tensors in an op must be on the same device, or it errors. On CPU-only,
  `.to(device)` is a harmless no-op (portable).

### train vs eval pass
- `model.train()` / `model.eval()` switches BatchNorm/Dropout modes.
- `with torch.set_grad_enabled(train):` — validation builds no gradient graph
  (faster, less memory) and never updates weights. Validation only **measures**.

### Loss averaging: `total_loss += loss.item() * images.size(0)`
- `CrossEntropyLoss` returns the **mean** loss per image in the batch.
- Batches aren't equal size (last one is partial: 356 = 11×32 + 4). Averaging the
  per-batch means would over-weight the small batch.
- So multiply mean × batch size → **sum** of losses; accumulate; divide by total
  images at the end → an honest per-image average. Same reason accuracy counts
  correct predictions ÷ total images.

### Datatype: float32
- Images `float32`, weights `float32`, labels `int64`. GPU math runs in FP32 by
  default (4 bytes/number).

### Mixed precision (AMP) — Phase 4
- **AMP = Automatic Mixed Precision.** Uses float16 (2 bytes, fast on GPU Tensor
  Cores) for heavy ops + float32 for delicate ops. Saves memory, runs faster.
- **`torch.autocast`** — auto-picks float16 for suitable ops (convs, matmuls),
  keeps float32 for losses/softmax/BN. PyTorch decides *what*; CUDA + GPU hardware
  make it *fast*. GPU only (no Tensor Cores on CPU).
- **`GradScaler`** — float16 gradients can **underflow to 0**. Scaler multiplies
  loss up before `backward()`, un-scales before `step()`, auto-tunes the factor.
  Pattern: `scaler.scale(loss).backward(); scaler.step(opt); scaler.update()`.
- Needed **only for float16 training**. Not for validation (no backward), not for
  bfloat16 (wide range), no-op on CPU.

### Warm start (Phase 4)
- Load the Phase 3 checkpoint before fine-tuning so the head isn't random.
- A random head → huge first-batch loss → huge gradients → would wreck the
  pretrained backbone before the head catches up. Warm start = "head already good,
  backbone pristine," then gentle adjustments.

---

## 6. Overfitting (what Phase 4 showed me)

- **Definition:** model memorizes the training set instead of learning to
  generalize.
- **Signatures (read the LOSS curve, not just accuracy):**
  - train loss → ~0, train acc → 1.000 (memorized),
  - **val loss turns around and rises** while train keeps falling → widening "X".
- **Cause here:** 11.2M trainable weights on only 356 images. The frozen baseline
  (3,078 params) *couldn't* overfit — too few knobs — so it generalized better
  (97.8% vs 95.5%).
- **Lesson:** more capacity isn't always better on small data. The simpler model
  won, and I have the curves to justify shipping it.
- **Best-model saving** (`if val_acc > best: torch.save(...)`) keeps the best epoch,
  so extra epochs can't worsen the saved file — but the curves still reveal
  overfitting. Fixes if I needed them: partial unfreeze (layer4 only), weight
  decay, early stopping, more data.

---

## 7. Evaluation & error analysis

### softmax → confidence
- `F.softmax(logits, dim=1)` turns the 6 raw scores into probabilities that sum to
  1. `.max(dim=1)` gives the winning class **and** its probability (confidence).
- A wrong prediction at 51% (unsure) is very different from wrong at 99%
  (confidently wrong — often a bad label or genuinely ambiguous image).

### `@torch.no_grad()`
- Decorator disabling gradient tracking for a whole function — evaluation never
  calls `backward()`, so skip the graph (faster, less memory).

### precision / recall / F1
- **Precision:** of images predicted class X, how many really were X? (punishes
  false alarms)
- **Recall:** of images that really were X, how many did it catch? (punishes
  misses)
- **F1:** harmonic mean of the two. **Support:** how many val images that class had.
- A class can be high on one, low on the other — the pair tells you *how* it fails.

### confusion matrix
- Rows = true class, columns = predicted. Diagonal = correct; off-diagonal = a
  specific confusion (true row → predicted column).

### error analysis (the real payoff)
- Look at the *actual* misclassified images. On my val set the 2 misses were:
  1. a screenshot I **mislabeled** as `landscape` — the model correctly said
     `documents` (a data bug the accuracy number hid);
  2. an `adi`→`people` miss at 0.61 conf — inherent overlap (both are humans).
- Reading mistakes > staring at the aggregate metric.

### honest caveat
- The val set also picked the best epoch, so it slightly influenced model
  selection → 97.8% mildly optimistic. A rigorous setup adds a separate untouched
  test set (skipped here given only 445 photos — worth stating, not hiding).

---

## 8. Deployment

### Streamlit mental model
- Streamlit **re-runs the whole script top-to-bottom on every interaction**
  (click, upload, slider). No callbacks — widgets return their current value each
  run, and I branch on them.
- **`@st.cache_resource`** on the model loader → model loads from disk **once**,
  survives across re-runs. Without it, every click reloads 45 MB.

### App flow (Quick test)
```
upload → rerun → get_model() (cached) → open_rgb() → predict()
       → eval transform → unsqueeze(0) [1,3,224,224] → model → softmax
       → sort → st.image + st.progress bars (top-3)
```
- `unsqueeze(0)` adds a batch dimension — the model always expects a batch, even
  of 1 image.

### Bulk / ZIP flow
- Input files (incl. a ZIP, unpacked with `zipfile`) → flatten to
  `[(name, bytes), …]` → per image: predict top-1, route to `category/` or
  `unsure/` by threshold → write **original bytes** into an in-memory output ZIP →
  `st.download_button`.
- **Max upload = 200 MB per file** by default (`server.maxUploadSize`), applies to
  the ZIP itself. Everything is in RAM, so the real limit on a free tier is memory
  (~1 GB) — keep to tens of photos; the CLI handles big libraries.

### Hosting
- Model ~45 MB → commit to GitHub, deploys with the repo. Runs on **CPU** on
  Streamlit Cloud (no GPU) — sub-second per image; `get_device()` falls back
  automatically. Install-time (CPU torch wheel) is the only heavy part.

---

## 9. Python / PyTorch syntax nuggets

- **Multiple assignment:** `a, b = x, y` packs `(x, y)` then unpacks → `a=x, b=y`.
  Used for `images, targets = images.to(device), targets.to(device)`.
- **Generator expression + sum:**
  `sum(p.numel() for p in model.parameters() if p.requires_grad)` — loop with a
  filter, streamed into `sum()`. `p.numel()` = number of elements in a tensor.
- **Reading vs writing an attribute:** `in_features = model.fc.in_features` READS
  the old layer's input size (512); `model.fc = nn.Linear(512, 6)` OVERWRITES the
  `fc` slot with a new layer. Same slot, two operations.
- **f-string formatting:** `{x:,}` = thousands separators; `{x:.2f}` = 2 decimals;
  `{x:.1%}` = percent with 1 decimal.

# Photo Classifier

**🔗 Live demo: [spadida-photo-classifier.streamlit.app](https://spadida-photo-classifier.streamlit.app/)**

A personal-photo categorizer: a CNN trained with **transfer learning** to sort a
photo library into content categories — `people`, `documents`, `landscape`,
`adi` (me), `food`, `lab`. Ships as two tools sharing one model:

1. **Streamlit web app** — drop in a photo or a ZIP, see predictions, download an
   organized ZIP. Live at
   [spadida-photo-classifier.streamlit.app](https://spadida-photo-classifier.streamlit.app/).
2. **Command-line tool** — point at a local folder of thousands of photos and
   sort them into per-category subfolders on disk.

> Third project in my ML curriculum, after a tabular regressor (Maricopa housing)
> and MNIST. This one introduces **transfer learning** — reusing an
> ImageNet-pretrained network instead of training from scratch.

## Results

Trained on a small personal dataset (**445 photos**, 6 classes, 80/20 split).

| Approach | Trainable params | Best val accuracy |
|----------|------------------|-------------------|
| Frozen ResNet18 + new head (feature extraction) | 3,078 | **97.8%** |
| Full fine-tuning (all layers unfrozen) | 11.2M | 95.5% — **overfit** |

**The frozen baseline won, and that's the shipped model.** Full fine-tuning drove
train accuracy to 100% while validation loss *rose* — textbook overfitting when
11.2M parameters meet ~356 training images. The simpler model generalized better.

**Error analysis** on the 2 validation misses was more revealing than the score:
- One was a screenshot I had **mislabeled** as `landscape` — the model correctly
  said `documents`. (Fixing the label → ~98.9% effective accuracy.)
- One was an `adi`→`people` confusion at low confidence — an inherent overlap
  (both are humans; the distinction is identity).

## The two interfaces

### Streamlit app (`app.py`)

Two modes via a sidebar toggle:
- **Quick test** — upload one photo, see top-3 predictions with confidence bars.
- **Bulk organize** — upload photos (or a ZIP), download a ZIP sorted into
  category subfolders, with low-confidence photos routed to `unsure/`.

```bash
streamlit run app.py
```

### CLI tool (`organize.py`)

The workhorse for a real library — streams from disk at constant memory:

```bash
python organize.py \
    --input  "C:\Users\Shady\Pictures\unsorted" \
    --output "C:\Users\Shady\Pictures\sorted" \
    --mode   copy \
    --threshold 0.7
```

- `--mode copy|move` — copy preserves originals; move relocates them.
- `--threshold` — photos below this confidence land in `unsure/` for manual review.
- Progress bar with ETA; unreadable files are set aside in `unreadable/`.

## Folder layout

```
photo-classifier/
├── app.py                      # Streamlit web app (two modes)
├── organize.py                 # CLI for local libraries
├── src/                        # Shared logic (imported by everything)
│   ├── categories.py           # Class list — single source of truth for label order
│   ├── preprocess.py           # Image → tensor transforms (train + eval)
│   ├── data.py                 # Dataset + DataLoader pipeline
│   ├── model.py                # Build ResNet18, freeze/unfreeze, pick device
│   └── inference.py            # Load model + predict (shared by app & CLI)
├── notebooks/                  # Training pipeline, one script per phase
│   ├── 01_inspect_photos.py    # Counts, class balance, sample grid
│   ├── 02_dataloader.py        # Build + visualize the input pipeline
│   ├── 03_baseline_frozen.py   # Frozen-backbone baseline (the shipped model)
│   ├── 04_finetune.py          # Full fine-tuning (overfit; kept for the writeup)
│   └── 05_evaluate.py          # Confusion matrix + error analysis
├── data/raw/<category>/        # Your photos, per class (gitignored — never committed)
├── models/
│   └── photo_classifier.pt     # Final model (frozen-backbone baseline), committed
├── requirements.txt
├── README.md
└── LICENSE
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                # Windows  (source .venv/bin/activate on macOS/Linux)
```

**PyTorch (GPU):** the default PyPI wheel is CPU-only. For an NVIDIA GPU, install
torch first from the CUDA wheel index, then the rest:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

CPU-only is fine too (just `pip install -r requirements.txt`) — inference on a
single image is well under a second, which is how the app runs on Streamlit Cloud.

## Prepare your photos (before Phase 1)

Sort training photos into per-class folders whose names match `src/categories.py`:

```
data/raw/
├── people/      ├── documents/   ├── landscape/
├── adi/         ├── food/        └── lab/
```

Guidelines: 50 photos/class minimum (200+ comfortable), roughly balanced, varied
lighting/angles. `data/` is gitignored — personal photos are never committed.

## Run the pipeline

```bash
python notebooks/01_inspect_photos.py        # Counts + sample grid
python notebooks/02_dataloader.py            # Build + visualize batches
python notebooks/03_baseline_frozen.py       # Frozen baseline  -> models/baseline_frozen.pt
python notebooks/04_finetune.py              # Fine-tune (overfits on this dataset)
python notebooks/05_evaluate.py              # Confusion matrix + error analysis

# Promote the chosen model to the canonical name, then launch the app:
#   Copy-Item models\baseline_frozen.pt models\photo_classifier.pt
streamlit run app.py
```

## What this project taught me

- **Transfer learning** — loading a pretrained `torchvision` ResNet18 and the
  freeze / unfreeze pattern.
- **ImageNet preprocessing** (mean/std normalization, 224×224) and why train and
  inference transforms must match exactly.
- **Data augmentation** (random crop, flip, color jitter) for a tiny dataset.
- **Custom `Dataset`** classes and stratified train/val splits.
- **Recognizing overfitting** from loss curves — and choosing the simpler model
  when it generalizes better.
- **Mixed-precision (AMP)** training on a 6 GB GPU.
- **Error analysis** — reading the actual mistakes, which surfaced a data-labeling
  bug the aggregate accuracy hid.
- **Two-interface deployment** — one model behind a web app and a CLI.

## License

Code is MIT-licensed ([LICENSE](LICENSE)). Personal photos in `data/` are
gitignored and never committed.

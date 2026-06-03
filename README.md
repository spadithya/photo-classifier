# Photo Classifier

A personal-photo categorizer: train a CNN to sort your photo library into
folders by content (people, food, pet, landscape, screenshot, …). Ships as
two tools sharing the same model:

1. **Streamlit web app** — drop in a few photos or a ZIP, see predictions
   and download an organized ZIP. Hosted on Streamlit Cloud.
2. **Command-line script** — point at a local folder of 1,000+ photos and
   organize them into per-category subfolders on disk.

> Third project in my [ML curriculum](../curriculum.txt). Follows:
> - [Maricopa Housing Predictor](../maricopa-housing-predictor) — tabular regression
> - MNIST Digit Recognizer — neural-net foundations
>
> This project introduces **transfer learning**, the technique that powers
> almost all real-world computer vision.

## Status

| Phase | Description                                       | Status      |
|-------|---------------------------------------------------|-------------|
| 1     | Pick categories + organize photos into folders    | in progress |
| 2     | Build PyTorch DataLoader + visualize batches      | pending     |
| 3     | Feature-extraction baseline (frozen ResNet)       | pending     |
| 4     | Full fine-tuning                                  | pending     |
| 5     | Evaluate + error analysis                         | pending     |
| 6     | Streamlit app + CLI organize tool                 | pending     |

## The two interfaces

### Streamlit app (`app.py`)

Live at `<your-app-name>.streamlit.app`. Two modes via a sidebar toggle:

- **Quick test** — upload a single photo, see top-3 predictions with confidence bars
- **Bulk organize** — upload up to ~50 photos (or a ZIP), get back a ZIP with
  photos sorted into category subfolders

Recruiters and friends can try the model in 30 seconds without setup.

### CLI tool (`organize.py`)

For actually organizing your local library:

```bash
python organize.py \
    --input  "C:\Users\desig\Pictures\unsorted" \
    --output "C:\Users\desig\Pictures\sorted" \
    --mode   copy \
    --threshold 0.7
```

- `--mode copy|move` — copy preserves originals; move is decisive
- `--threshold` — photos below the confidence cutoff land in `unsure/` for manual review
- Progress bar shows runtime; ~5-10 minutes for 8,000 photos on CPU

## Folder layout

```
photo-classifier/
├── app.py                            # Streamlit web app (two modes)
├── organize.py                       # CLI for local libraries
├── src/                              # Shared logic
│   ├── model.py                      # Load weights + run inference
│   ├── preprocess.py                 # Image → tensor pipeline
│   └── categories.py                 # Class name list
├── notebooks/                        # Training pipeline (Phases 1-5)
│   ├── 01_inspect_photos.py
│   ├── 02_dataloader.py
│   ├── 03_baseline_frozen.py
│   ├── 04_finetune.py
│   └── 05_evaluate.py
├── data/
│   ├── raw/                          # Your photos, organized by class (gitignored)
│   │   ├── people/
│   │   ├── food/
│   │   ├── pet/
│   │   └── ...
│   └── processed/                    # Cached tensors (gitignored)
├── models/
│   └── photo_classifier.pt           # Final fine-tuned weights (committed)
├── assets/                           # README screenshots, demo GIFs
├── requirements.txt
├── README.md
└── LICENSE
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                # Windows
# source .venv/bin/activate           # macOS / Linux

pip install -r requirements.txt
```

PyTorch installation note: the default `pip install torch` pulls the CPU
build, which is fine for fine-tuning ResNet18 on ~1,000 photos (3-5 min per
epoch). For GPU acceleration see [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally).

## Prepare your photos

Before Phase 2, organize your training photos into per-class folders:

```
data/raw/
├── people/         (50-200 photos of people)
├── food/           (50-200 photos of meals, drinks, snacks)
├── pet/            (50-200 photos of your pet / other pets)
├── landscape/      (50-200 outdoor / nature photos)
├── screenshot/     (50-200 phone screenshots)
└── ...
```

Rough guidelines:
- 50 photos per class minimum, 200+ is comfortable
- More classes = harder (start with 6-8, expand later)
- Roughly equal counts per class (avoid 50/50/50/2000 splits)
- Photos should be varied — different lighting, angles, subjects

## Run the pipeline

```bash
python notebooks/01_inspect_photos.py        # Count + visualize per class
python notebooks/02_dataloader.py            # Build PyTorch dataset
python notebooks/03_baseline_frozen.py       # Frozen ResNet baseline
python notebooks/04_finetune.py              # Fine-tune
python notebooks/05_evaluate.py              # Confusion matrix + error analysis
streamlit run app.py                         # Launch the web demo
```

## What this project teaches

Beyond what MNIST covered:

- **Loading pretrained models** from `torchvision.models`
- The **freeze / unfreeze** pattern for transfer learning
- **ImageNet preprocessing** (mean/std normalization, 224×224 inputs)
- **Color images** (3 channels) and `torchvision.transforms`
- **Data augmentation** for tiny datasets (random crops, flips, color jitter)
- **Custom `Dataset`** classes for your own files
- **Learning rate tuning** for fine-tuning (typically 10-100× smaller than from-scratch)
- **Error analysis** — looking at *which* images the model gets wrong, not just the aggregate metric
- **Two-interface deployment** — same model, web app + CLI

## License

Code is MIT-licensed ([LICENSE](LICENSE)).
Personal photos in `data/` are never committed — gitignored by default.

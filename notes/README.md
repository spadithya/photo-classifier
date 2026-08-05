# Notes — Photo Classifier

Study notes built from the questions I worked through while building this project.

- **[concepts.md](concepts.md)** — every concept I learned, grouped by topic
  (transfer learning, weights/tensors, the data pipeline, ResNet internals,
  training mechanics, overfitting, evaluation, deployment, Python syntax).
- **[training-flow.md](training-flow.md)** — the algorithm / flow of the whole
  training pipeline (Phases 1–5) and the inner epoch loop.
- **[cli-flow.md](cli-flow.md)** — the algorithm / flow of the CLI tool
  (`organize.py`), from a folder of photos to sorted subfolders.

## The 60-second summary

Take a **ResNet18** already trained on **ImageNet**, **freeze** its feature-
extracting body, and train only a small new **head** (`Linear(512 → 6)`) to map
its 512-number image "fingerprint" to my 6 categories. That frozen baseline hit
**97.8%** on 445 photos. Full fine-tuning overfit, so the simpler model shipped —
behind both a Streamlit app and a CLI that share one `predict()` function.

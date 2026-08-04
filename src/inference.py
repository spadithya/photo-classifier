"""Inference helpers — load the trained model and predict on images.

Shared by app.py (Streamlit) and organize.py (CLI) so both interfaces use
IDENTICAL prediction logic. This loads the frozen-baseline model we chose to
ship in Phase 4 (saved here under the canonical name photo_classifier.pt).
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from src.categories import CATEGORIES, INDEX_TO_NAME
from src.model import build_model, get_device
from src.preprocess import build_eval_transform

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "photo_classifier.pt"


def load_model(model_path: Path = DEFAULT_MODEL_PATH, device=None):
    """Build the architecture, load the saved weights, put it in eval mode.

    Returns (model, device). eval() matters: it switches BatchNorm to use its
    stored running stats, so a single image is classified consistently.
    """
    device = device or get_device()
    model = build_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


@torch.no_grad()
def predict(model, image, device, transform=None):
    """Predict on ONE PIL RGB image.

    Returns a list of (category_name, probability) for all classes, sorted
    highest-probability first. The caller can take [0] for the top guess or
    [:3] for a top-3 display.
    """
    transform = transform or build_eval_transform()

    # transform -> [3,224,224]; unsqueeze(0) adds a batch dim -> [1,3,224,224],
    # because the model always expects a batch, even of size 1.
    tensor = transform(image).unsqueeze(0).to(device)

    probs = F.softmax(model(tensor), dim=1)[0].cpu()   # [num_classes] probabilities
    ranked = sorted(
        ((INDEX_TO_NAME[i], float(probs[i])) for i in range(len(CATEGORIES))),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return ranked

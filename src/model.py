"""Model definition — a ResNet18 adapted to our categories.

Used by the training scripts (Phase 3 frozen baseline, Phase 4 fine-tuning) and
later by the Streamlit app / CLI at inference. build_model() is the ONE place
the network architecture is defined, so training and inference can't disagree
about what the model is.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from src.categories import NUM_CLASSES


def get_device():
    """Return the CUDA GPU if available, else CPU.

    PyTorch does NOT use the GPU automatically — you must move both the model
    and each batch of data onto this device with .to(device). This helper is
    the single place we decide where computation happens.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes: int = NUM_CLASSES, freeze_backbone: bool = True):
    """Load a pretrained ResNet18 and swap its head for our classes.

    Steps:
      1. Load ResNet18 with its ImageNet-pretrained weights.
      2. If freeze_backbone: set requires_grad=False on every existing layer so
         they DON'T update during training — the pretrained features stay put.
         This is 'feature extraction' (Phase 3).
      3. Replace the final fully-connected layer. The original maps 512 features
         -> 1000 ImageNet classes; we swap in a fresh Linear(512 -> num_classes).
         This new layer starts with random weights and requires_grad=True, so it
         trains even when everything before it is frozen.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # resnet18's classifier head is `model.fc`: Linear(in_features=512, out=1000).
    in_features = model.fc.in_features          # 512 for resnet18
    model.fc = nn.Linear(in_features, num_classes)

    return model

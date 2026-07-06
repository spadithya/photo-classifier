"""Image preprocessing — shared across training, evaluation, and the app.

Like src/categories.py, this is a single source of truth: it defines HOW a raw
image file becomes the tensor the model sees. The training script, evaluation
script, Streamlit app, and CLI all import from here so their preprocessing can
never drift apart.

Why that matters: a model must be fed images preprocessed the *same way* at
inference as during training. If they disagree (e.g. different resize or
normalization), the model silently gets worse and it's a nightmare to debug.
Centralizing it here makes that impossible.
"""

from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms

# --- HEIC support (iPhone .heic photos) ------------------------------------
# Same optional decoder registration as the Phase 1 script. If pillow-heif is
# installed, .heic files open like any other image; if not, they'll fail to
# open and get skipped by load_image() below.
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

# --- ImageNet statistics ----------------------------------------------------
# ResNet was pretrained on ImageNet, whose images were normalized with these
# per-channel (R, G, B) means and standard deviations. To reuse those
# pretrained weights we MUST normalize our images the same way, so the input
# distribution matches what the network already learned to expect.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ResNet expects 224x224 inputs.
IMG_SIZE = 224


def load_image(path):
    """Open an image file and force it to 3-channel RGB. Returns None if broken.

    Photos arrive as RGBA, grayscale, palette, etc. The CNN always expects
    exactly 3 channels, so we collapse everything to RGB here.
    """
    try:
        img = Image.open(path)
        img.load()                 # force a full read so truncated files error now
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None


def build_eval_transform():
    """Deterministic pipeline for validation and real predictions.

    No randomness: we want evaluation and inference to be repeatable. Resize the
    short side to 256, take the central 224x224, convert to a tensor, normalize.
    """
    return transforms.Compose([
        transforms.Resize(256),             # short side -> 256 px (keeps aspect ratio)
        transforms.CenterCrop(IMG_SIZE),    # central 224x224 square
        transforms.ToTensor(),              # PIL [0..255] HWC uint8 -> float [0..1] CHW
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_train_transform():
    """Augmented pipeline for TRAINING ONLY.

    With ~450 photos, augmentation shows the model a slightly different version
    of each image every epoch (random crop, flip, color shift) so it learns to
    generalize instead of memorizing exact pixels. These random ops must never
    run at eval time, where we want stable, repeatable predictions.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),  # random zoom+crop
        transforms.RandomHorizontalFlip(),                         # 50% mirror
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(tensor):
    """Undo Normalize so a tensor can be displayed as a real image.

    Normalized tensors contain negative values and won't render correctly. This
    reverses the operation per channel (x * std + mean), then clamps to [0,1].
    Used only for visualization, never for the model.
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)

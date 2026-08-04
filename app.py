"""
Streamlit web app for the photo classifier.

Two modes (chosen from the sidebar):
  - Quick test    : upload one photo, see the top-3 predictions with confidence
  - Bulk organize : upload many photos (or a ZIP), download a ZIP sorted into
                    per-category subfolders, with low-confidence photos in unsure/

Run locally:
    streamlit run app.py
"""

from pathlib import Path
import sys
import io
import zipfile

# Make src/ importable (app.py lives at the project root).
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from PIL import Image

from src.categories import CATEGORIES
from src.inference import load_model, predict, DEFAULT_MODEL_PATH

# Extensions we accept for upload. (heic works because importing src.inference ->
# src.preprocess registers the pillow-heif opener.)
UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp", "bmp", "gif", "tiff", "heic"]

st.set_page_config(page_title="Photo Classifier", page_icon="📷", layout="centered")


# --- Load the model once, cached across reruns -----------------------------
# Streamlit re-runs the whole script on every interaction. @st.cache_resource
# makes sure the model is loaded from disk only ONCE, not on every click.
@st.cache_resource
def get_model():
    return load_model()


def open_rgb(file_or_bytes):
    """Open an uploaded file / raw bytes as a 3-channel RGB PIL image."""
    if isinstance(file_or_bytes, (bytes, bytearray)):
        file_or_bytes = io.BytesIO(file_or_bytes)
    return Image.open(file_or_bytes).convert("RGB")


# ---------------------------------------------------------------------------
# Mode 1 — Quick test
# ---------------------------------------------------------------------------
def quick_test(model, device):
    st.subheader("Quick test")
    st.caption("Upload one photo and see the model's top-3 guesses.")

    uploaded = st.file_uploader("Choose a photo", type=UPLOAD_TYPES,
                                accept_multiple_files=False)
    if uploaded is None:
        return

    image = open_rgb(uploaded)
    ranked = predict(model, image, device)

    col_img, col_pred = st.columns([1, 1])
    with col_img:
        st.image(image, caption=uploaded.name, use_container_width=True)
    with col_pred:
        top_name, top_prob = ranked[0]
        st.markdown(f"### {top_name}")
        st.caption(f"top prediction · {top_prob:.1%} confident")
        st.write("")
        for name, prob in ranked[:3]:
            st.write(f"**{name}** — {prob:.1%}")
            st.progress(prob)


# ---------------------------------------------------------------------------
# Mode 2 — Bulk organize
# ---------------------------------------------------------------------------
def collect_images(uploaded_files):
    """Flatten uploads into [(filename, raw_bytes), ...], expanding any ZIPs."""
    items = []
    for f in uploaded_files:
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(f.getvalue())) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    ext = Path(info.filename).suffix.lower().lstrip(".")
                    if ext in UPLOAD_TYPES:
                        items.append((Path(info.filename).name, zf.read(info)))
        else:
            items.append((f.name, f.getvalue()))
    return items


def bulk_organize(model, device):
    st.subheader("Bulk organize")
    st.caption("Upload many photos (or a ZIP). Download them sorted into "
               "per-category folders.")

    threshold = st.slider(
        "Confidence threshold", min_value=0.0, max_value=1.0, value=0.60, step=0.05,
        help="Photos the model is less sure about than this go into an 'unsure/' "
             "folder for you to check by hand.",
    )
    uploaded_files = st.file_uploader(
        "Choose photos or a ZIP", type=UPLOAD_TYPES + ["zip"],
        accept_multiple_files=True,
    )
    if not uploaded_files:
        return

    items = collect_images(uploaded_files)
    if not items:
        st.warning("No images found in the upload.")
        return

    # Classify every image and build an output ZIP in memory.
    counts = {c: 0 for c in CATEGORIES}
    counts["unsure"] = 0
    out_buffer = io.BytesIO()

    progress = st.progress(0.0, text="Classifying…")
    with zipfile.ZipFile(out_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, (fname, raw) in enumerate(items):
            try:
                image = open_rgb(raw)
            except Exception:
                continue  # skip unreadable files
            top_name, top_prob = predict(model, image, device)[0]
            folder = top_name if top_prob >= threshold else "unsure"
            counts[folder] += 1
            # index prefix avoids name collisions between different source folders
            zf.writestr(f"{folder}/{idx:04d}_{fname}", raw)
            progress.progress((idx + 1) / len(items), text=f"Classifying… {idx + 1}/{len(items)}")
    progress.empty()

    st.success(f"Sorted {len(items)} photo(s).")
    # Summary table of how many landed in each folder.
    summary = {k: v for k, v in counts.items() if v > 0}
    st.table({"category": list(summary.keys()), "photos": list(summary.values())})

    st.download_button(
        "Download sorted ZIP",
        data=out_buffer.getvalue(),
        file_name="sorted_photos.zip",
        mime="application/zip",
    )


# ---------------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------------
def main():
    st.title("📷 Photo Classifier")
    st.caption("A ResNet18 fine-tuned to sort personal photos into "
               + ", ".join(CATEGORIES) + ".")

    if not DEFAULT_MODEL_PATH.exists():
        st.error(
            f"Model file not found: `{DEFAULT_MODEL_PATH.name}`.\n\n"
            "Copy your trained weights there first:\n\n"
            "`Copy-Item models\\baseline_frozen.pt models\\photo_classifier.pt`"
        )
        st.stop()

    model, device = get_model()

    mode = st.sidebar.radio("Mode", ["Quick test", "Bulk organize"])
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Running on: **{device}**")
    with st.sidebar.expander("About"):
        st.write(
            "Transfer learning with a frozen ImageNet-pretrained ResNet18 "
            "backbone and a trained classification head. ~98% validation "
            "accuracy on a small personal dataset."
        )

    if mode == "Quick test":
        quick_test(model, device)
    else:
        bulk_organize(model, device)


if __name__ == "__main__":
    main()

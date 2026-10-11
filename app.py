import io
import threading
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

from scripts.models import VGG16Binary
from scripts.explain import binary_gradcam


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "outputs" / "vgg16_frozen" / "best.pt"

st.set_page_config(
    page_title="Household Animals Classifier",
    page_icon="🐾",
    layout="wide",
)


@st.cache_resource
def load_model(checkpoint_path, checkpoint_modified):
    saved = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    config = saved["config"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = VGG16Binary(
        hidden_units=config["hidden_units"],
        dropout=config["dropout"],
        pretrained=False,
    )
    model.load_state_dict(saved["model_state_dict"])
    model.to(device)
    model.eval()

    # Explanations need input gradients, not parameter updates.
    for parameter in model.parameters():
        parameter.requires_grad = False

    preprocess = transforms.Compose([
        transforms.Resize((
            config["image_size"],
            config["image_size"],
        )),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config["normalization_mean"],
            std=config["normalization_std"],
        ),
    ])

    # Protect the shared model while temporary Grad-CAM hooks are active.
    lock = threading.RLock()

    return model, preprocess, device, lock


st.title("🐾 Household Animals Classifier")
st.write("Upload a cat or dog photograph to classify it.")
st.caption(
    "This model has two classes: cat and dog. "
    "Other subjects will also receive one of these labels."
)

if not CHECKPOINT.is_file():
    st.error(f"Checkpoint not found: {CHECKPOINT}")
    st.stop()

with st.spinner("Loading the trained model..."):
    model, preprocess, device, model_lock = load_model(
        str(CHECKPOINT),
        CHECKPOINT.stat().st_mtime_ns,
    )

uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded is None:
    st.info("Upload an image to start.")
    st.stop()

try:
    with Image.open(io.BytesIO(uploaded.getvalue())) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.load()
except (
    UnidentifiedImageError,
    OSError,
    ValueError,
    Image.DecompressionBombError,
):
    st.error("This image could not be read. Try another JPG or PNG.")
    st.stop()

tensor = preprocess(image).unsqueeze(0).to(device)

with model_lock:
    with torch.inference_mode():
        logit = model(tensor)[0]

        if not torch.isfinite(logit).item():
            st.error("The model produced an invalid prediction.")
            st.stop()

        dog_probability = torch.sigmoid(logit).item()
        predicted_class = int(logit.item() >= 0)

cat_probability = 1.0 - dog_probability
predicted_name = "Dog" if predicted_class == 1 else "Cat"

image_column, prediction_column = st.columns(2)

with image_column:
    st.image(
        image,
        caption="Uploaded image",
        width="stretch",
    )

with prediction_column:
    st.subheader(f"Prediction: {predicted_name}")

    cat_column, dog_column = st.columns(2)
    cat_column.metric("Cat score", f"{cat_probability:.2%}")
    dog_column.metric("Dog score", f"{dog_probability:.2%}")

    st.caption(
        "These are model probability scores, "
        "not calibrated confidence guarantees."
    )

    show_heatmap = st.button(
        "Generate Grad-CAM heatmap",
        type="primary",
    )

if show_heatmap:
    with st.spinner("Generating the explanation..."):
        with model_lock:
            result = binary_gradcam(
                model=model,
                image_tensor=tensor,
                target_layer=model.features[29],
                target_class=predicted_class,
            )

    if not result["has_positive_signal"]:
        st.info("No positive Grad-CAM signal for this prediction.")
    else:
        heatmap = result["heatmap"]

        # Match the model input geometry for accurate overlay alignment.
        display_image = image.resize(
            (224, 224),
            resample=Image.Resampling.BILINEAR,
        )
        display_array = (
            np.asarray(display_image).astype(np.float32) / 255.0
        )

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(display_array)
        axes[0].set_title("Model input")

        axes[1].imshow(
            heatmap, cmap="jet", vmin=0, vmax=1
        )
        axes[1].set_title(f"Grad-CAM for {predicted_name}")

        axes[2].imshow(display_array)
        axes[2].imshow(
            heatmap,
            cmap="jet",
            vmin=0,
            vmax=1,
            alpha=0.4,
        )
        axes[2].set_title("Overlay")

        for ax in axes:
            ax.axis("off")

        fig.tight_layout()

        st.pyplot(fig)

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150)
        plt.close(fig)

        st.download_button(
            "Download explanation",
            data=buffer.getvalue(),
            file_name="gradcam_explanation.png",
            mime="image/png",
        )

        st.caption(
            "Warmer regions show stronger relative positive "
            "contributions in this Grad-CAM map. "
            "The heatmap is not an object segmentation."
        )
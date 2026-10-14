from pathlib import Path
from io import BytesIO

import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Dog vs Cat Classification",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "Models" / "dog_cat_cnn_v1.keras"
TEST_DIR = PROJECT_DIR / "Data" / "test"
RESULTS_DIR = PROJECT_DIR / "Results"
CONFUSION_MATRIX_PATHS = [
    RESULTS_DIR / "confusion_matrix.png",
    RESULTS_DIR / "plots" / "confusion_matrix.png",
]

IMAGE_SIZE = (256, 256)
CLASS_NAMES = ["cats", "dogs"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

st.title("Dog vs Cat Classification")
st.caption("Custom CNN with Grad-CAM visualisation")


# ------------------------------------------------------------
# Model loading and Grad-CAM graph
# ------------------------------------------------------------
@st.cache_resource
def load_model_and_gradcam(model_path: str):
    trained_model = tf.keras.models.load_model(model_path, compile=False)

    last_conv_layer = next(
        (
            layer for layer in reversed(trained_model.layers)
            if isinstance(layer, tf.keras.layers.Conv2D)
        ),
        None,
    )
    if last_conv_layer is None:
        raise ValueError("No convolutional layer was found in the model.")

    # Build a functional graph layer by layer for compatibility with Keras 3.
    grad_input = tf.keras.Input(shape=trained_model.input_shape[1:])
    x = grad_input
    conv_features = None

    for layer in trained_model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        x = layer(x)
        if layer.name == last_conv_layer.name:
            conv_features = x

    if conv_features is None:
        raise ValueError("Could not access the last convolutional layer output.")

    grad_model = tf.keras.Model(
        inputs=grad_input,
        outputs=[conv_features, x],
    )
    return trained_model, grad_model


if not MODEL_PATH.is_file():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    model, grad_model = load_model_and_gradcam(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()


# ------------------------------------------------------------
# Image helpers
# ------------------------------------------------------------
def preprocess_image(image: Image.Image):
    resized = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = tf.convert_to_tensor(array[np.newaxis, ...], dtype=tf.float32)
    return tensor, np.asarray(resized, dtype=np.uint8)


def predict_image(image: Image.Image, threshold: float = 0.5):
    tensor, _ = preprocess_image(image)
    output = model(tensor, training=False)
    dog_score = float(output[0, 0].numpy())
    cat_score = 1.0 - dog_score
    class_index = 1 if dog_score >= threshold else 0
    confidence = dog_score if class_index == 1 else cat_score

    return {
        "label": "Dog" if class_index == 1 else "Cat",
        "class_index": class_index,
        "confidence": confidence,
        "dog_score": dog_score,
        "cat_score": cat_score,
    }


def make_gradcam(image: Image.Image, class_index: int):
    tensor, original = preprocess_image(image)

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(tensor, training=False)
        dog_score = predictions[:, 0]
        class_score = dog_score if class_index == 1 else 1.0 - dog_score
        score = tf.reduce_sum(class_score)

    gradients = tape.gradient(score, conv_output)
    if gradients is None:
        raise RuntimeError("Could not calculate Grad-CAM gradients.")

    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(weights * conv_output, axis=-1)[0]
    heatmap = tf.nn.relu(heatmap)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(
        maximum > 0,
        heatmap / (maximum + tf.keras.backend.epsilon()),
        tf.zeros_like(heatmap),
    )
    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis], IMAGE_SIZE, method="bilinear"
    ).numpy().squeeze()
    heatmap = np.clip(heatmap, 0, 1)

    heatmap_rgb = (
        plt.get_cmap("jet")(heatmap)[..., :3] * 255
    ).astype(np.uint8)
    overlay = np.clip(
        0.6 * original + 0.4 * heatmap_rgb, 0, 255
    ).astype(np.uint8)

    return original, heatmap_rgb, overlay


def find_example_images():
    examples = []
    for class_name in CLASS_NAMES:
        class_dir = TEST_DIR / class_name
        if class_dir.is_dir():
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    examples.append((class_name, path))
                    if len([x for x in examples if x[0] == class_name]) >= 10:
                        break
    return examples


# ------------------------------------------------------------
# Simple interface: prediction and confusion matrix only
# ------------------------------------------------------------
prediction_tab, matrix_tab = st.tabs(["Predict an image", "Confusion matrix"])

with prediction_tab:
    threshold = st.slider(
        "Dog classification threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
    )

    examples = find_example_images()
    source = st.radio(
        "Image source",
        ["Upload an image", "Choose a test example"] if examples else ["Upload an image"],
        horizontal=True,
    )

    selected_image = None
    true_label = None

    if source == "Upload an image":
        uploaded = st.file_uploader(
            "Choose a cat or dog image",
            type=["jpg", "jpeg", "png", "webp"],
        )
        if uploaded is not None:
            try:
                selected_image = Image.open(uploaded).convert("RGB")
            except Exception as exc:
                st.error(f"Could not open this image: {exc}")
    else:
        example_labels = [f"{label.title()} — {path.name}" for label, path in examples]
        selected_example = st.selectbox("Select an example", example_labels)
        example_index = example_labels.index(selected_example)
        true_label, selected_path = examples[example_index]
        try:
            selected_image = Image.open(selected_path).convert("RGB")
        except Exception as exc:
            st.error(f"Could not open this example: {exc}")

    if selected_image is None:
        st.info("Upload an image or choose a test example to begin.")
    else:
        result = predict_image(selected_image, threshold)
        image_col, result_col = st.columns([1, 1], gap="large")

        with image_col:
            st.image(selected_image, caption="Input image", use_container_width=True)

        with result_col:
            st.subheader(f"Prediction: {result['label']}")
            st.metric("Confidence score", f"{result['confidence'] * 100:.2f}%")
            st.write(f"Cat score: {result['cat_score'] * 100:.2f}%")
            st.progress(float(np.clip(result["cat_score"], 0, 1)))
            st.write(f"Dog score: {result['dog_score'] * 100:.2f}%")
            st.progress(float(np.clip(result["dog_score"], 0, 1)))

            if true_label is not None:
                expected = "Dog" if true_label == "dogs" else "Cat"
                if result["label"] == expected:
                    st.success(f"Dataset label: {expected}; prediction matches.")
                else:
                    st.warning(f"Dataset label: {expected}; prediction differs.")

        with st.expander("Show Grad-CAM explanation", expanded=True):
            st.caption(
                "Grad-CAM highlights regions that influence the selected class score; "
                "it is an approximate explanation."
            )
            try:
                original, heatmap, overlay = make_gradcam(
                    selected_image, result["class_index"]
                )
                grad_cols = st.columns(3)
                grad_cols[0].image(original, caption="Original", use_container_width=True)
                grad_cols[1].image(heatmap, caption="Grad-CAM heatmap", use_container_width=True)
                grad_cols[2].image(overlay, caption="Overlay", use_container_width=True)

                buffer = BytesIO()
                Image.fromarray(overlay).save(buffer, format="PNG")
                st.download_button(
                    "Download Grad-CAM overlay",
                    data=buffer.getvalue(),
                    file_name="gradcam_overlay.png",
                    mime="image/png",
                )
            except Exception as exc:
                st.warning(f"Could not generate Grad-CAM: {exc}")

        st.caption(
            "Confidence is the model's output score and may not represent a calibrated probability."
        )

with matrix_tab:
    st.subheader("Model confusion matrix")
    confusion_path = next(
        (path for path in CONFUSION_MATRIX_PATHS if path.is_file()),
        None,
    )
    if confusion_path is not None:
        st.image(str(confusion_path), use_container_width=True)
        st.caption(
            "This displays the saved confusion matrix from your evaluation. "
            "For a final unbiased score, evaluate on a held-out test set not used during training."
        )
    else:
        st.info(
            "Confusion matrix image not found. Save it as "
            "Results/confusion_matrix.png."
        )

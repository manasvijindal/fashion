import time
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from tensorflow import keras

st.set_page_config(page_title="Fashion-MNIST Classifier", page_icon="🧥", layout="centered")

CLASS_NAMES = np.array([
    "T-shirt/top","Trouser","Pullover","Dress","Coat",
    "Sandal","Shirt","Sneaker","Bag","Ankle boot"
])

@st.cache_resource
def load_model():
    # Make sure the file is in the same folder as this script
    return keras.models.load_model("fashion_mnist_cnn.keras")

def preprocess_user_image(pil_image: Image.Image, invert_if_bright: bool = True) -> np.ndarray:
    """Center-crop -> 28x28 -> grayscale -> [0,1] -> (28,28,1)."""
    img = pil_image.convert("L")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((28, 28), Image.BILINEAR)
    x = np.array(img).astype("float32") / 255.0
    if invert_if_bright and x.mean() > 0.5:
        x = 1.0 - x
    return x[..., np.newaxis]  # (28,28,1)

st.title("Fashion Classifier")
st.caption("Upload a clothing image. The model predicts one of 10 Fashion-MNIST classes and shows Top-3 with probabilities.")

# UI controls
col_a, col_b = st.columns(2)
with col_a:
    invert = st.checkbox("Auto-invert if background is bright", value=True)
with col_b:
    show_pre = st.checkbox("Show 28×28 preprocessed preview", value=True)

model = load_model()
uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg","webp","bmp"])

if uploaded is not None:
    pil = Image.open(uploaded)
    st.image(pil, caption="Original", use_container_width=True)

    x = preprocess_user_image(pil, invert_if_bright=invert)
    if show_pre:
        st.image(
            Image.fromarray((np.squeeze(x) * 255).astype(np.uint8), mode="L").resize((112, 112)),
            caption="Preprocessed (28×28 grayscale)",
            use_container_width=False,
        )

    x_batch = np.expand_dims(x, axis=0)  # (1,28,28,1)

    t0 = time.time()
    proba = model.predict(x_batch, verbose=0)[0]  # shape (10,)
    latency_ms = (time.time() - t0) * 1000

    # Top-3
    top3_idx = np.argsort(proba)[-3:][::-1]
    top3_labels = CLASS_NAMES[top3_idx]
    top3_scores = proba[top3_idx]

    st.subheader(f"Prediction: **{top3_labels[0]}** ({top3_scores[0]:.2%})")
    st.caption(f"Inference time: {latency_ms:.1f} ms (CPU)")

    # Show top-3 nicely
    top3_df = pd.DataFrame({
        "Class": top3_labels,
        "Confidence": (top3_scores * 100).round(2).astype(str) + " %"
    })
    st.table(top3_df)

    # Full probability bar chart (sorted desc)
    chart_df = pd.DataFrame({
        "class": CLASS_NAMES,
        "probability": proba
    }).sort_values("probability", ascending=False).reset_index(drop=True)
    st.bar_chart(chart_df.set_index("class"))
else:
    st.info("👆 Upload a clothing image to see predictions.")

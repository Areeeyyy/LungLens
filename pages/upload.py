from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from pages.gradcam import generate_gradcam_heatmap, overlay_heatmap_on_image

MODEL_PATH = Path("pneumonia_cnn_model.h5")
IMG_SIZE = (150, 150)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


@st.cache_resource
def load_model():
    """Load the CNN model."""
    if MODEL_PATH.exists():
        return tf.keras.models.load_model(str(MODEL_PATH))
    st.error(
        f"Model not found at `{MODEL_PATH}`. "
        "Please place the trained model file in the project root."
    )
    st.stop()


def preprocess(uploaded_image: Image.Image) -> np.ndarray:
    """Resize, normalize, and add batch dimension."""
    img = uploaded_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def page_upload():
    st.markdown(
        """
        <div class="upload-header">
            <h2>Upload Your X-Ray</h2>
            <p>Drag and drop a chest X-ray image below (.png, .jpg, .jpeg)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload chest X-ray",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        image = Image.open(uploaded)

        # Load model
        model = load_model()

        # Run prediction
        x = preprocess(image)
        prob_pneumonia = float(model.predict(x, verbose=0)[0][0])
        pred_label = CLASS_NAMES[1] if prob_pneumonia >= 0.5 else CLASS_NAMES[0]
        confidence = prob_pneumonia if pred_label == "PNEUMONIA" else 1 - prob_pneumonia

        # Generate Grad-CAM heatmap
        heatmap = generate_gradcam_heatmap(model, x)
        gradcam_image = overlay_heatmap_on_image(image.resize(IMG_SIZE), heatmap, alpha=0.4)

        # ── Layout: 3 columns ────────────────────────────────────────
        col_img, col_gradcam, col_result = st.columns([1, 1, 1], gap="large")

        with col_img:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(image, caption="Uploaded X-Ray", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_gradcam:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(gradcam_image, caption="Grad-CAM Heatmap", use_container_width=True)
            st.markdown(
                """
                <div class="gradcam-legend">
                    <p class="gradcam-legend-title">Area Fokus Model</p>
                    <div class="gradcam-legend-bar"></div>
                    <div class="gradcam-legend-labels">
                        <span>Rendah</span>
                        <span>Tinggi</span>
                    </div>
                    <p class="gradcam-legend-desc">
                        Area berwarna <strong style="color:#C62828;">merah</strong> menunjukkan
                        region yang paling berpengaruh terhadap keputusan klasifikasi model.
                        Pada kasus pneumonia, area ini umumnya menunjukkan
                        <em>infiltrat</em> atau peningkatan opasitas pada lapang paru.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_result:
            # Result badge
            badge_class = "result-pneumonia" if pred_label == "PNEUMONIA" else "result-normal"
            bar_color_pneu = "#C62828" if prob_pneumonia >= 0.5 else "#2E7D32"
            bar_color_norm = "#2E7D32" if prob_pneumonia < 0.5 else "#C62828"

            st.markdown(
                f'<h3 style="font-size:1.1rem; font-weight:700; margin-bottom:1rem; '
                f'color:var(--text-primary);">Prediction Result</h3>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="text-align:center; margin-bottom:1.5rem;">'
                f'<span class="result-badge {badge_class}">{pred_label}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="metric-row">'
                f'<span class="metric-label">Confidence</span>'
                f'<span class="metric-value">{confidence * 100:.1f}%</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="metric-row">'
                f'<span class="metric-label">P(Pneumonia)</span>'
                f'<span class="metric-value">{prob_pneumonia:.4f}</span></div>'
                f'<div class="prob-bar-container">'
                f'<div class="prob-bar-fill" style="width:{prob_pneumonia*100:.1f}%; '
                f'background:{bar_color_pneu};"></div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="metric-row">'
                f'<span class="metric-label">P(Normal)</span>'
                f'<span class="metric-value">{1 - prob_pneumonia:.4f}</span></div>'
                f'<div class="prob-bar-container">'
                f'<div class="prob-bar-fill" style="width:{(1-prob_pneumonia)*100:.1f}%; '
                f'background:{bar_color_norm};"></div></div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="disclaimer">
                <strong>⚠️ Disclaimer:</strong> This prediction is for educational/research
                purposes only and is <strong>not</strong> a substitute for professional medical diagnosis.
            </div>
            """,
            unsafe_allow_html=True,
        )

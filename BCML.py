import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO
from tensorflow.keras.models import load_model
import tempfile
import time

# =========================
# CUSTOM STYLE
# =========================
st.markdown("""
<style>

/* =========================
   MAIN APP BACKGROUND
========================= */

.stApp {
    background:
        radial-gradient(circle at 20% 20%, rgba(255,105,180,0.35), transparent 32%),
        radial-gradient(circle at 80% 30%, rgba(255,182,193,0.30), transparent 34%),
        radial-gradient(circle at 30% 80%, rgba(255,160,200,0.28), transparent 30%),
        radial-gradient(circle at 70% 75%, rgba(255,220,230,0.30), transparent 32%),
        linear-gradient(
            135deg,
            #fff0f3 0%,
            #ffd6e7 18%,
            #f4c2ff 38%,
            #d8b4ff 55%,
            #ffb3d9 72%,
            #ffeef6 88%,
            #fff8f6 100%
        );

    background-size: 250% 250%;
    animation: gradientMove 12s ease infinite;
    overflow-x: hidden;
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Floating blobs */
.stApp::before,
.stApp::after {
    content: "";
    position: fixed;
    width: 550px;
    height: 550px;
    border-radius: 50%;
    z-index: -1;
    filter: blur(70px);
    opacity: 0.60;
    animation: floatBlob 18s infinite alternate ease-in-out;
}

.stApp::before {
    background:
        radial-gradient(circle,
            rgba(255,105,180,0.75) 0%,
            rgba(255,182,193,0.45) 45%,
            transparent 75%);
    top: -140px;
    left: -140px;
}

.stApp::after {
    background:
        radial-gradient(circle,
            rgba(180,120,255,0.70) 0%,
            rgba(255,120,220,0.45) 45%,
            transparent 75%);
    bottom: -150px;
    right: -120px;
    animation-delay: 5s;
}

@keyframes floatBlob {
    from { transform: translateY(0px) translateX(0px) scale(1); }
    to { transform: translateY(40px) translateX(30px) scale(1.12); }
}

/* =========================
   TEXT
========================= */

h1 { color: #7a1f3d; font-weight: 800; }
h2, h3 { color: #8f2d56; }
p, label, div { color: #4d2a36; }

/* =========================
   GLASS EFFECT
========================= */

[data-testid="stVerticalBlock"] > div:has(.element-container),
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.28);
    border: 1px solid rgba(255,255,255,0.35);
    backdrop-filter: blur(10px);
    border-radius: 18px;
    padding: 0.5rem;
}

/* =========================
   BUTTONS
========================= */

.stButton>button {
    width: 100%;
    border-radius: 14px;
    height: 3rem;
    font-size: 18px;
    font-weight: bold;
    border: none;
    color: white;

    background: linear-gradient(
        135deg,
        #c94fcf 0%,
        #d96c8a 30%,
        #b46cff 65%,
        #f5b7c8 100%
    );

    box-shadow: 0 4px 18px rgba(180,108,255,0.35);
}

/* =========================
   RESULT BOXES
========================= */

.result-box {
    padding: 1rem;
    border-radius: 20px;
    text-align: center;
    margin-top: 1rem;
    backdrop-filter: blur(8px);
}

.good {
    background-color: rgba(120,255,180,0.12);
    border: 1px solid rgba(80,220,140,0.35);
}

.bad {
    background-color: rgba(255,80,120,0.12);
    border: 1px solid rgba(255,80,120,0.35);
}

/* =========================
   FILE UPLOADER (FIXED STABLE DROP ZONE)
========================= */

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.28);
    border-radius: 22px;
    border: 2px dashed rgba(201, 79, 207, 0.7);

    padding: 1.5rem;
    min-height: 180px;

    text-align: center;
    transition: all 0.3s ease;
    overflow: hidden;
}

/* hover effect */
[data-testid="stFileUploader"]:hover {
    border: 2px dashed rgba(201, 79, 207, 1);
    background: rgba(255,255,255,0.38);
}

/* prevent layout breaking after upload */
[data-testid="stFileUploader"] section {
    border: none;
}

/* keep file list compact */
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] ul {
    font-size: 12px !important;
    line-height: 1.2;
}

/* prevent overflow expansion */
[data-testid="stFileUploader"] div {
    max-height: 140px;
    overflow-y: auto;
}

/* =========================
   IMAGES
========================= */

img {
    border-radius: 16px;
}

/* =========================
   PROGRESS BAR
========================= */

.stProgress > div > div > div > div {
    background-color: #c94fcf;
}

</style>
""", unsafe_allow_html=True)


# =========================
# LOAD MODEL
# =========================
model = load_model("/src/model/Refined_model_cw_v6.keras")


# =========================
# FUNCTION
# =========================
def load_crop_resize_image(image_path, s, mode="RGB"):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2

    img = img.crop((left, top, left + crop_size, top + crop_size))
    img = img.resize((s, s), Image.Resampling.LANCZOS)

    if mode == "L":
        return np.array(img.convert("L"))
    return np.array(img)


# =========================
# APP
# =========================
#st.title(" 🔬 Breast Tumor Histopathology Classifier 🧬")
st.markdown(
    "<h1 style='text-align: center; font-size: 42px;'>🔬 Breast Tumor Histopathology Classifier 🧬</h1>",
    unsafe_allow_html=True
)
#st.caption(
#    "A deep learning–based approach for the classification of histopathological tissue images. This model uses a pre-trained ImageNet model to predict breast cancer tumor classifications from histopathology images. This application is intended for educational and research purposes only and should not be considered a real clinical or diagnostic tool."
#)
st.markdown(
    """
    <p style='text-align: center; font-size: 18px; color: #4d2a36; max-width: 900px; margin: auto;'>
    This application uses a deep learning model to classify histopathological tissue images for educational and research purposes. 
    It is based on a pre-trained ImageNet backbone and predicts breast cancer tumor categories from image data. 
    This tool is NOT intended for clinical diagnosis or medical decision-making.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("### 📤 Upload Histopathology Image")

uploaded_file = st.file_uploader(
    "Drag and drop your image here, or click to browse",
    type=["png", "jpg", "jpeg", "bmp", "webp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    width, height = image.size

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Width", width)
    with col2:
        st.metric("Height", height)
    with col3:
        st.metric("Crop Size", min(width, height))

    square_size = min(width, height)

    max_x = width - square_size
    max_y = height - square_size

    st.subheader("✂️ Select Crop Area")

    x = st.slider("X Position", 0, max_x, max_x // 2) if max_x > 0 else 0
    y = st.slider("Y Position", 0, max_y, max_y // 2) if max_y > 0 else 0

    cropped = image.crop((x, y, x + square_size, y + square_size))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Cropped")
        st.image(cropped, use_container_width=True)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cropped.save(tmp.name)
        processed = load_crop_resize_image(tmp.name, 224, "L")

    model_input = processed.reshape(1, 224, 224, 1).astype(np.float32)

    if st.button("🔬 Predict"):

        progress = st.progress(0)

        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        prediction = model.predict(model_input)
        probability = float(prediction[0][0])

        if probability >= 0.5:
            st.markdown(f"""
            <div class="result-box bad">
                <h2>⚠️ Malignant</h2>
                <h3>Probability: {probability:.2%}</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box good">
                <h2>✅ Benign</h2>
                <h3>Probability: {(1-probability):.2%}</h3>
            </div>
            """, unsafe_allow_html=True)

    buffer = BytesIO()
    cropped.save(buffer, format="PNG")

    st.download_button(
        "⬇️ Download Cropped Image",
        buffer.getvalue(),
        "cropped.png",
        "image/png"
    )
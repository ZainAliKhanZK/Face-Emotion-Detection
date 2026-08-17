import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download

# ────────────────────────────────
# Page config
# ────────────────────────────────
st.set_page_config(
    page_title="Emotion Detection",
    page_icon="🎭",
    layout="centered",
)

# ────────────────────────────────
# Custom styling
# ────────────────────────────────
st.markdown("""
    <style>
        .main {
            padding-top: 1.5rem;
        }
        .title-text {
            text-align: center;
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0;
        }
        .subtitle-text {
            text-align: center;
            color: #888;
            font-size: 1.05rem;
            margin-top: 0.2rem;
            margin-bottom: 1.5rem;
        }
        .emotion-badge {
            display: inline-block;
            padding: 0.5rem 1.3rem;
            border-radius: 999px;
            font-size: 1.3rem;
            font-weight: 600;
            color: white;
            text-align: center;
            margin-top: 0.8rem;
        }
        div[data-testid="stCameraInput"] {
            border: 2px dashed #ccc;
            border-radius: 12px;
            padding: 0.8rem;
        }
    </style>
""", unsafe_allow_html=True)

EMOTION_COLORS = {
    "Angry": "#e74c3c",
    "Disgust": "#8e44ad",
    "Fear": "#7f8c8d",
    "Happy": "#f1c40f",
    "Neutral": "#3498db",
    "Sad": "#2c3e50",
    "Surprise": "#e67e22",
}

EMOTION_EMOJI = {
    "Angry": "😠",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😄",
    "Neutral": "😐",
    "Sad": "😢",
    "Surprise": "😲",
}

# ────────────────────────────────
# Header
# ────────────────────────────────
st.markdown('<p class="title-text">🎭 Emotion Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Take a photo and let the model read your expression</p>', unsafe_allow_html=True)

# ────────────────────────────────
# Sidebar
# ────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app detects faces in your photo and predicts the emotion using a trained CNN model.")
    st.write("**Detectable emotions:**")
    for emo, emoji in EMOTION_EMOJI.items():
        st.write(f"{emoji} {emo}")
    st.divider()
    st.caption("Tip: good lighting and a front-facing pose improve accuracy.")

# ────────────────────────────────
# Model loading
# ────────────────────────────────
@st.cache_resource
def load_emotion_model():
    model_path = hf_hub_download(
        repo_id="ZainAliKhanZAK/emotion-model",
        filename="emotion_model.keras"
    )
    return load_model(model_path)

@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

with st.spinner("Loading model..."):
    model = load_emotion_model()
    face_cascade = load_face_cascade()

classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def preprocess_face(face_img):
    face = cv2.resize(face_img, (48, 48))
    face = face / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    return face.astype(np.float32)

def predict_emotion(face_img):
    processed = preprocess_face(face_img)
    prediction = model.predict(processed, verbose=0)
    idx = np.argmax(prediction)
    confidence = float(prediction[0][idx]) * 100
    return classes[idx], confidence

# ────────────────────────────────
# Camera input
# ────────────────────────────────
img_file = st.camera_input("📸 Take a photo")

if img_file is not None:
    bytes_data = img_file.getvalue()
    img_array = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    img_array = cv2.flip(img_array, 1)  # mirror to match preview

    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    with st.spinner("Analyzing expression..."):
        faces = face_cascade.detectMultiScale(gray, minNeighbors=10, minSize=(80, 80))

    if len(faces) == 0:
        st.warning("😕 No face detected — try retaking the photo with better lighting.")
    else:
        results = []
        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            label, confidence = predict_emotion(face_gray)
            color = EMOTION_COLORS.get(label, "#2ecc71")
            bgr = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))

            cv2.rectangle(img_array, (x, y), (x + w, y + h), bgr, 3)
            cv2.putText(img_array, label, (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
            results.append((label, confidence))

        st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), caption="Result", use_container_width=True)

        st.markdown("### Detected emotion(s)")
        cols = st.columns(len(results))
        for col, (label, confidence) in zip(cols, results):
            color = EMOTION_COLORS.get(label, "#2ecc71")
            emoji = EMOTION_EMOJI.get(label, "")
            col.markdown(
                f'<div style="text-align:center;">'
                f'<div class="emotion-badge" style="background:{color};">{emoji} {label}</div>'
                f'<div style="color:#888; margin-top:4px;">{confidence:.1f}% confidence</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
else:
    st.info("👆 Click the camera button above to get started.")

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download

st.title("Emotion Detection")


@st.cache_resource
def load_emotion_model():
    model_path = hf_hub_download(
        repo_id="ZainAliKhanZAK/emotion-model",
        filename="emotion_model.keras"
    )
    return load_model(model_path)

model = load_emotion_model()

@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
    return classes[np.argmax(prediction)]

img_file = st.camera_input("Take a photo")

if img_file is not None:
    bytes_data = img_file.getvalue()
    img_array = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # ✅ Flip horizontally to match the mirrored preview the user saw
    img_array = cv2.flip(img_array, 1)

    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, minNeighbors=10, minSize=(80, 80))

    if len(faces) == 0:
        st.warning("No face detected — try retaking the photo with better lighting.")
    else:
        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            label = predict_emotion(face_gray)
            cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_array, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), caption="Result")

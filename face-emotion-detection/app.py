# pip install streamlit-webrtc streamlit opencv-python-headless

import cv2
import av
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from streamlit_webrtc import VideoProcessorBase
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from huggingface_hub import hf_hub_download


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


@st.cache_resource
def load_emotion_model():
    model_path = hf_hub_download(
        repo_id="ZainAliKhanZAK/emotion-model",
        filename="emotion_model.keras"
    )
    return load_model(model_path)

model = load_emotion_model()

# @st.cache_resource
# def load_emotion_model():
#     return load_model("second_model.keras")

# model = load_emotion_model()


# ✅ Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ✅ Load the trained emotion recognition model

# ✅ Define emotion classes
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ✅ Preprocessing function (expects grayscale face image)
def preprocess_face(face_img):
    face = cv2.resize(face_img, (48, 48))          # Resize to match model input
    face = face / 255.0                            # Normalize pixel values
    face = np.expand_dims(face, axis=-1)           # Add channel dimension (48,48,1)
    face = np.expand_dims(face, axis=0)            # Add batch dimension (1,48,48,1)
    return face

# ✅ Emotion prediction
def predict_emotion(face_img):
    processed = preprocess_face(face_img)
    prediction = model.predict(processed, verbose=0)
    class_idx = np.argmax(prediction)
    return classes[class_idx]




class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.frame_count = 0
        self.last_faces = []
        self.last_label = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")   # convert incoming frame to a normal cv2 image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self.frame_count += 1
        if self.frame_count % 3 == 0:
            self.last_faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=10, minSize=(80, 80)
            )
            if len(self.last_faces) > 0:
                (x, y, w, h) = self.last_faces[0]
                face_gray = gray[y:y+h, x:x+w]
                if face_gray.size > 0:
                    self.last_label = predict_emotion(face_gray)

        for (x, y, w, h) in self.last_faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, self.last_label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")




st.title("Live Emotion Detection")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

# import cv2
# print(cv2.__version__)
# print(cv2.data.haarcascades)  # should print a path, not error
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# print("OK")
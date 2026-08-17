# 🎭 SeeFace — Real-Time Emotion Detection

SeeFace is a web app that detects faces in a photo and predicts the person's emotion using a Convolutional Neural Network (CNN) built with Keras. It's built with Streamlit for a fast, interactive UI, right in your browser.

**🔗 Live Demo:** [seeface.streamlit.app](https://seeface.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

- 📸 **In-browser camera capture** — no upload needed, just take a photo
- 🧠 **CNN-based emotion classification** across 7 emotion classes
- 🟩 **Face detection** using OpenCV's Haar Cascade classifier
- 🎨 **Color-coded results** with confidence scores for each detected face
- 👥 **Multi-face support** — detects and classifies more than one face per photo
- ⚡ **Cached model loading** for fast repeat predictions

---

## 🖥️ Try It Live

No installation needed — try the deployed app here:

👉 **https://seeface.streamlit.app/**

Just allow camera access, take a photo, and see your predicted emotion.

---

## 🧠 Emotions Detected

| Emotion  | Emoji |
|----------|-------|
| Angry    | 😠 |
| Disgust  | 🤢 |
| Fear     | 😨 |
| Happy    | 😄 |
| Neutral  | 😐 |
| Sad      | 😢 |
| Surprise | 😲 |

---

## 🛠️ Tech Stack

- **Frontend / App Framework:** [Streamlit](https://streamlit.io/)
- **Face Detection:** OpenCV (Haar Cascade Classifier)
- **Emotion Model:** Keras `Sequential` CNN, trained on 48×48 grayscale face images
- **Model Hosting:** [Hugging Face Hub](https://huggingface.co/)
- **Core Libraries:** `opencv-python-headless`, `numpy`, `tensorflow`, `huggingface_hub`

---

## 📂 Project Structure

```
├── app.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ How It Works

1. The app captures a photo using `st.camera_input`.
2. The image is converted to grayscale, and OpenCV's Haar Cascade detects face regions.
3. Each detected face is cropped, resized to 48×48, normalized, and passed through the CNN model.
4. The model outputs a probability distribution over 7 emotion classes.
5. The predicted emotion and confidence score are displayed with a bounding box and colored badge.

---

## 🚀 Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📦 requirements.txt

```
streamlit
opencv-python-headless
numpy
tensorflow
huggingface_hub
```

---

## 🧩 Model

The emotion recognition model is a CNN trained on grayscale 48×48 facial images (similar to the FER2013 format) and is automatically downloaded from the Hugging Face Hub on first run:

```
repo_id: ZainAliKhanZAK/emotion-model
filename: emotion_model.keras
```

---

## 📌 Notes & Limitations

- Prediction accuracy depends on lighting, face angle, and image quality.
- The model works best with a single, front-facing, well-lit face.
- Facial expression recognition models can carry biases inherited from their training data — results should be treated as approximate, not definitive.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open a pull request or file an issue.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋 Author

Built by **Zain Ali Khan**

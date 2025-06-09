######################################################3

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO
import time
import os

# --- Load models ---
model = tf.keras.models.load_model("sign_language(M2)_model.h5")
yolo_model = YOLO("modelsyolov11n_custom.pt")

# --- Class Labels ---
class_labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "DEL", "SPACE", "BLANK"
]

# --- Session State Setup ---
if "history" not in st.session_state:
    st.session_state.history = []
if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False
if "last_saved_time" not in st.session_state:
    st.session_state.last_saved_time = 0
if "last_saved_class" not in st.session_state:
    st.session_state.last_saved_class = None

# --- Preprocess image for .h5 model ---
def preprocess_image(image):
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# --- Save predictions to CSV ---
def save_to_csv(predicted_class, confidence, source):
    df = pd.DataFrame([{
        "source": source,
        "prediction": predicted_class,
        "confidence": confidence
    }])
    df.to_csv("predictions.csv", mode="a", index=False, header=not os.path.exists("predictions.csv"))

# --- Load history from CSV ---
def load_history():
    try:
        return pd.read_csv("predictions.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["source", "prediction", "confidence"])

# --- Predict and Store ---
def predict_and_store(image, source="Uploaded Image"):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 80
    st.session_state.history.append({
        "source": source,
        "prediction": predicted_class,
        "confidence": confidence
    })
    save_to_csv(predicted_class, confidence, source)
    return predicted_class, confidence

# --- App UI ---
st.title("Sign Language Recognition")
st.write("Upload an image or use webcam to recognize hand signs.")

# --- Input Mode Selection ---
option = st.radio("Choose Input Type:", ("Upload Image", "Use Webcam"))

# --- Upload Image Flow ---
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        predicted_class, confidence = predict_and_store(image)
        st.success(f"Predicted Sign: {predicted_class} ({confidence:.2f}% confidence)")

# --- Webcam Flow ---
elif option == "Use Webcam":
    st.write("Real-time webcam detection with bounding boxes.")
    col1, col2 = st.columns(2)
    if col1.button("Start Webcam"):
        st.session_state.webcam_running = True
    if col2.button("Stop Webcam"):
        st.session_state.webcam_running = False

    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while st.session_state.webcam_running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame.")
                break

            results = yolo_model(frame)
            annotated_frame = results[0].plot()

            stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            # Save prediction only if 5 sec passed or new class
            if results[0].boxes and len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                predicted_class = class_labels[cls_id]

                current_time = time.time()
                if (current_time - st.session_state.last_saved_time >= 5) and (predicted_class != st.session_state.last_saved_class):
                    st.session_state.history.append({
                        "source": "Webcam",
                        "prediction": predicted_class,
                        "confidence": conf * 80
                    })
                    save_to_csv(predicted_class, conf * 100, "Webcam")
                    st.session_state.last_saved_time = current_time
                    st.session_state.last_saved_class = predicted_class

        cap.release()
        cv2.destroyAllWindows()
        st.info("Webcam stopped.")

# --- Sidebar: Prediction History ---
st.sidebar.header("Prediction History")
history_df = load_history()

if not history_df.empty:
    st.sidebar.write("### Previous Predictions")
    st.sidebar.dataframe(history_df)

    if st.sidebar.button("Clear Last Prediction"):
        st.session_state.history.pop()
        history_df = history_df[:-1]
        history_df.to_csv("predictions.csv", index=False)
        st.sidebar.success("Last prediction removed!")

    if st.sidebar.button("Clear All History"):
        open("predictions.csv", "w").close()
        st.session_state.history = []
        st.sidebar.success("All history cleared!")

st.write("Thank you for using this tool!")


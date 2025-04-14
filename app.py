######################################################3

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import time

# Load classification model
model = load_model("D:/SSSS/sign_language(M2)_model.h5")

# Load YOLOv11n model
yolo_model = YOLO("D:/SSSS/modelsyolov11n_custom.pt")

# Class labels
class_labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "DEL", "SPACE", "BLANK"
]

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

# Preprocess image for classification
def preprocess_image(image):
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Predict and save
def predict_and_store(image, source="Uploaded Image"):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 80

    st.session_state.history.append({"source": source, "prediction": predicted_class, "confidence": confidence})
    save_to_csv(predicted_class, confidence, source)

    return predicted_class, confidence

# Save to CSV
def save_to_csv(predicted_class, confidence, source):
    df = pd.DataFrame([{"source": source, "prediction": predicted_class, "confidence": confidence}])
    df.to_csv("predictions.csv", mode="a", index=False, header=False)

# Load CSV history
def load_history():
    try:
        return pd.read_csv("predictions.csv", names=["source", "prediction", "confidence"])
    except FileNotFoundError:
        return pd.DataFrame(columns=["source", "prediction", "confidence"])

# App title
st.title("Sign Language Recognition ✋🤟")
st.write("Upload an image or use webcam to recognize hand signs.")

# Choose input type
option = st.radio("Choose Input Type:", ("Upload Image", "Use Webcam"))

# Upload image section
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image of a hand sign", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        predicted_class, confidence = predict_and_store(image)
        st.success(f"Predicted Sign: {predicted_class} ({confidence:.2f}% confidence)")

# Webcam detection section
elif option == "Use Webcam":
    st.write("Real-time webcam detection with bounding boxes.")

    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False

    if "last_saved_time" not in st.session_state:
        st.session_state.last_saved_time = 0

    if "last_saved_class" not in st.session_state:
        st.session_state.last_saved_class = None

    if st.button("Start Webcam", key="start_webcam"):
        st.session_state.webcam_running = True
    if st.button("Stop Webcam", key="stop_webcam"):
        st.session_state.webcam_running = False

    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while st.session_state.webcam_running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame")
                break

            # Run YOLO detection
            results = yolo_model(frame)

            # Annotate the frame
            annotated_frame = results[0].plot()

            # Display frame
            stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            # Save prediction if new class or 5 seconds passed
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

# Sidebar: Prediction history
st.sidebar.header("Prediction History")
history_df = load_history()
if not history_df.empty:
    st.sidebar.write("### Previous Predictions")
    st.sidebar.dataframe(history_df)

    if st.sidebar.button("Clear Last Prediction"):
        st.session_state.history.pop()
        history_df = history_df[:-1]
        history_df.to_csv("predictions.csv", index=False, header=False)
        st.sidebar.success("Last prediction removed!")

    if st.sidebar.button("Clear All History"):
        open("predictions.csv", "w").close()
        st.session_state.history = []
        st.sidebar.success("All history cleared!")

st.write("Thank you 🤝 for using this tool ❤")

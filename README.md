Real-Time Sign Language Recognition System

Objective:
To build a real-time and image-based Sign Language Recognition system using deep learning models integrated into a Streamlit web application. The system recognizes hand gestures representing sign language alphabets and provides users with prediction results and an intuitive interface for interaction.

1. Project Pipeline:-

Step 1: Environment Setup
        python version (3.1.11)

A Python virtual environment was created to manage dependencies and ensure a clean development environment.

All necessary packages such as TensorFlow, OpenCV, Streamlit, pandas, NumPy, Pillow, and Ultralytics (YOLO) were installed.


Step 2: Model Preparation

Classification Model:

A pre-trained Keras model (.h5) is used for predicting sign language from static images.

Input images are preprocessed to 64x64 resolution and normalized.

Detection Model:

A custom YOLOv11n model (.pt) is employed for real-time object detection through webcam.

It detects hand signs in each frame and classifies them accordingly.

Step 3: Streamlit Application Development

User Interface:

The app is built with Streamlit, providing a user-friendly web interface.

Users can choose between uploading an image or using the webcam for real-time detection.

Image Upload:

Users can upload image files, which are processed and predicted using the classification model.

Webcam Stream:

Real-time webcam video is streamed with detected bounding boxes using the YOLO model.

Predictions are updated continuously per frame.
  

Step 4: Prediction Management

Predictions, including the source, predicted class, and confidence score, are stored in session state and saved to a predictions.csv file.

A sidebar panel displays a table of past predictions.

Users can clear the last prediction or delete the entire prediction history.


Step 5: Additional Features

Display of annotated frames for better visualization.

Confidence filtering and timestamp-based prediction saving.

Integration of user feedback messages for better interactivity.


 Tools and Frameworks Used :-

TensorFlow/Keras: For training and running the classification model.

YOLOv11n (Ultralytics): For real-time hand gesture detection.

OpenCV: For webcam handling and image processing.

Streamlit: For building the web application.

Pandas/Numpy/Pillow: For data handling and image manipulation.

 Conclusion :-

The developed system provides a robust platform for recognizing sign language both from static images and live video feed.
It promotes inclusivity and accessibility using a combination of deep learning and modern web technologies.


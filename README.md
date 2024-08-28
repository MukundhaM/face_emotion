Emotion Recognition Web Application
Overview
This project is a web application for real-time emotion recognition using deep learning. The application uses a Convolutional Neural Network (CNN) to detect faces in images or video feeds and classify their emotions. The application is built with Flask, a lightweight web framework for Python, and uses OpenCV for image processing.

Features
->Real-time Emotion Detection: Capture emotions from live video feed using your webcam.
->Image Upload: Upload an image to detect and classify the emotions of any detected faces.
->Pre-trained Model: The model is pre-trained on a dataset of facial expressions and can recognize seven different emotions.


Installation
Prerequisites
-Python 3.x
-Pip (Python package installer)
-Virtual environment (recommended)


Install Dependencies
pip install -r requirements.txt

Required Files
Trained Model: The model (model_file.h5) that you trained using the Jupyter Notebook should be placed in the project root directory.
Face Detection Model: Ensure you have the haarcascade_frontalface_default.xml file in the project directory for face detection.

Usage
Real-Time Emotion Detection
->Open the web application.
->Allow access to your webcam.
->The application will detect faces in the video feed and classify their emotions in real time.
Upload an Image
->On the home page, upload an image containing faces.
->The application will detect faces in the image and display the predicted emotions.


Directory Structure

├── static/                     # Folder for static files like uploaded images
├── templates/                  # HTML templates for the Flask app
│   ├── index.html              # Main page
│   └── results.html            # Results page for displaying predictions
├── app.py                      # Main Flask application
├── model_file.h5               # Trained Keras model (from Jupyter Notebook)
├── haarcascade_frontalface_default.xml  # Haar Cascade XML for face detection
├── requirements.txt            # List of Python dependencies
└── README.md                   # This README file

Model Training
The emotion recognition model was trained using a Jupyter Notebook with a custom dataset. The model is a Convolutional Neural Network (CNN) that can recognize the following emotions:
Angry
Disgust
Fear
Happy
Neutral
Sad
Surprise

Dataset
The model was trained on the FER-2013 dataset, which contains over 35,000 labeled facial images.

Training Process
The Jupyter Notebook used for training is included in this repository. It outlines the process of data preprocessing, model architecture definition, and model training.

Dependencies
Flask
OpenCV
NumPy
Keras
TensorFlow


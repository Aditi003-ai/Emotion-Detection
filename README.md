# Emotion-Detection
# Emotion Detection System using Computer Vision
## About the Project

This project is a Real-Time Emotion Detection System developed using Computer Vision and Deep Learning (CNN). The system captures live video through a webcam, detects human faces, and predicts emotions in real time.

The model is trained using facial expression images and classifies emotions into seven categories:

Angry

Disgusted

Fearful

Happy

Neutral

Surprised

Sad

This project demonstrates the practical implementation of Artificial Intelligence, Image Processing, and Convolutional Neural Networks for emotion recognition.

## Team Members
Aditi Verma

Jayanshi Ratan Sinha

Anchal Rani

Yash Chauhan

## Features
Real-time emotion detection using webcam

Face detection using Haar Cascade Classifier

CNN-based emotion classification

Training and live prediction modes

Data augmentation for better performance

Accuracy and loss visualization

Automatic model saving after training

## Technology Stack
Programming Language
Python

Libraries and Frameworks

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

Concepts Used

Computer Vision

Deep Learning

Convolutional Neural Networks (CNN)

Image Processing

Real-Time Prediction

## Project Structure
Emotion-Detection/
│
├── emotion.py

├── model.h5

├── training_plot.png

├── haarcascade_frontalface_default.xml

├── fer2013_extracted/

│   ├── train/

│   └── test/
│
├── requirements.txt
└── README.md

## Dataset
Dataset Used: FER-2013 (Facial Expression Recognition Dataset)
Dataset Information:

Training Images: 28,709

Validation Images: 7,178

Image Size: 48 × 48

Grayscale Images

Dataset is used to train the CNN model for facial emotion classification.

## Model Architecture

The system uses a Convolutional Neural Network consisting of:

Input Layer (48×48×1)

Convolution Layer (32 Filters)

Convolution Layer (64 Filters)

Max Pooling
Dropout
Convolution Layer (128 Filters)

Max Pooling
Convolution Layer (128 Filters)

Max Pooling
Flatten Layer
Dense Layer (1024 Neurons)

Dropout
Output Layer (7 Classes)

The output layer uses Softmax Activation for emotion prediction.

## Installation

Clone Repository
git clone <your-repository-link>

Move into Project Directory
cd Emotion-Detection

Install Dependencies

pip install tensorflow

pip install opencv-python

pip install numpy

pip install matplotlib

How to Run the Project

Train the Model

python emotion.py --mode train

Output Generated:

model.h5
training_plot.png

Run Live Emotion Detection

python emotion.py --mode display

Press:
q → Exit Webcam

## Working Methodology
Load training and testing dataset

Apply image preprocessing and augmentation

Train CNN model

Save trained model

Open webcam

Detect faces using Haar Cascade

Predict emotions

Display emotion label on screen

## Future Improvements
Improve model accuracy

Support multiple face detection

Deploy as web application

Add emotion analytics dashboard

Improve UI design

## Learning Outcomes
This project helped in understanding:

Deep Learning workflow

CNN architecture

Image preprocessing

Real-time computer vision

Model evaluation and prediction

## Acknowledgements
FER-2013 Dataset

TensorFlow Documentation

OpenCV Documentation

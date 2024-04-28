
![1703153572931](https://github.com/Cyril-7/Face-Recognition/assets/129573220/005b080e-57cd-429e-867a-c93eaa1dace6)

![1703153610502](https://github.com/Cyril-7/Face-Recognition/assets/129573220/d43bd6b3-e2a9-4129-bac1-3efa54f6d1f9)


# Face Recognition using KNN

This project demonstrates a simple face recognition system using the K-Nearest Neighbors (KNN) algorithm implemented with Python and OpenCV.

## Description

The project consists of two main scripts:

1. `add_new_face.py`: This script captures images from the webcam, detects faces, and allows users to add their faces to the dataset for training the KNN classifier.

2. `face_recognition.py`: This script loads the pre-trained KNN classifier along with the face dataset and performs real-time face recognition using the webcam feed.

## Dependencies

- Python 3.x
- OpenCV
- NumPy
- scikit-learn

## Installation

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/your_username/Face-Recognition-using-KNN.git
    ```

2. Install the required dependencies using pip:

    ```bash
    pip install opencv-python numpy scikit-learn
    ```

## Usage

1. Run `add_new_face.py` to capture new faces and add them to the dataset.

    ```bash
    python add_new_face.py
    ```

    Follow the instructions to enter your name and capture your face samples.

2. Run `face_recognition.py` to perform face recognition using the trained model.

    ```bash
    python face_recognition.py
    ```

    The script will open a webcam feed and display real-time face recognition results.

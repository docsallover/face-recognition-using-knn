# Face Recognition Using KNN

This project demonstrates a simple face recognition system using the K-Nearest Neighbors (KNN) algorithm. KNN is a supervised machine learning algorithm that classifies new data points based on the majority class of their nearest neighbors. In this case, the algorithm will be used to recognize faces by comparing them to a database of known faces.

## Overview

The system consists of the following steps:

1. Data Collection: Collect a dataset of images containing faces of known individuals. Ensure that the images have consistent lighting and pose.

2. Feature Extraction: Extract numerical features from each image. Common features include Haar-like features or deep learning-based features.

3. Training: Train the KNN classifier using the extracted features and corresponding labels (identity of the person in each image).

4. Recognition: To recognize a new face, extract its features and find the K nearest neighbors in the training dataset. The class of the majority of these neighbors is considered the predicted identity of the new face.

## How to Use
To use the system, follow these steps:
1. Clone the repository.
2. Create a virtual environment (venv or virtualenv) in the project directory.
3. Activate the virtual environment.
4. Install the required dependencies.
   - Run `pip install -r requirements.txt`.
5. Adding New Faces
   - Run `add-new-face.py`:
   - Locate the script named `add-new-face.py`. Execute this script to launch the program. It will typically activate your computer's camera.
6. Face Detection and Naming:
   - When prompted, enter a name for the person whose face is being detected. This name will be associated with the captured face image for future recognition.
   - Face the camera so that your face is centered within the frame. The script should detect your face.
   
8. Face Recognition
   - Run `face-recognition.py`:
   - Locate the script responsible for face recognition, likely named `face-recognition.py`. Execute this script to start the program. It will typically activate your computer's camera.

9. Real-Time Recognition:
   - Face the camera again. The script should detect faces in the camera view. If a recognized face is detected, the script should display the corresponding name associated with that face.

## Dependencies
The system requires the following dependencies:
- numpy
- opencv-python
- scikit-learn
- scipy

## License
This project is licensed under the MIT License. See the LICENSE file for more details.


## Visit and Follow
For more details and tutorials, visit the website: [DocsAllOver](https://docsallover.com/).

Follow us on:
- [Facebook](https://www.facebook.com/docsallover)
- [Instagram](https://www.instagram.com/docsallover.tech/)
- [LinkedIn](https://www.linkedin.com/company/docsallover/)
- [YouTube](https://www.youtube.com/@docsallover)
- [Threads.net](https://threads.net/docsallover.tech)

and visit our website to know more about our tutorials and blogs.

# Sign-Language-Detector🖐️

A simple deep learning project for classifying American Sign Language (ASL) hand gestures using TensorFlow and OpenCV.

## 📦 Requirements

- Python 3.8+
- TensorFlow
- OpenCV
- NumPy

Install dependencies:
```bash
pip install tensorflow opencv-python numpy

🧠 Project Overview
This project trains a Convolutional Neural Network (CNN) to recognize ASL hand signs from grayscale images. The model is trained on a dataset of labeled hand gesture images and saved for future use.

📁 Project Structure
sign-language-project/
│
├── train_model.py         # Train and save the model
├── detect_sign.py         # Run live prediction using webcam
├── model/
│   ├── asl_model.h5       # Trained model
│   └── labels.txt         # Class labels
├── dataset/
│   └── asl data/          # Training images (folders per letter)


🚀 How to Use
1. Train the Model
python train_model.py


- Loads images from dataset/asl data
- Converts them to grayscale and resizes to 64x64
- Builds and trains a CNN model
- Saves the model to model/asl_model.h5

2. Run Live Detection (Optional)
python detect_sign.py
- Opens webcam
- Captures hand gesture
- Predicts the corresponding ASL letter
- Displays result on screen

## 📊 Model Training Progress

The model was trained for 20 epochs using a CNN architecture. Below is a snapshot of the training accuracy and loss:

| Epoch | Accuracy | Loss    |
|-------|----------|---------|
| 1     | 0.0765   | 3.2301  |
| 2     | 0.3212   | 2.4541  |
| 3     | 0.6208   | 1.3598  |
| 4     | 0.8019   | 0.6899  |
| 5     | 0.8965   | 0.3650  |
| 6     | 0.9565   | 0.1970  |
| 7     | 0.9854   | 0.0868  |
| 8     | 0.9950   | 0.0449  |
| 9     | 0.9977   | 0.0235  |
| 10    | 0.9996   | 0.0142  |
| 11    | 0.9996   | 0.0077  |
| 12    | 1.0000   | 0.0049  |
| 13    | 1.0000   | 0.0033  |
| 14    | 1.0000   | 0.0026  |
| 15    | 1.0000   | 0.0022  |
| 16    | 1.0000   | 0.0018  |
| 17    | 1.0000   | 0.0016  |
| 18    | 1.0000   | 0.0014  |
| 19    | 1.0000   | 0.0012  |
| 20    | 1.0000   | 0.0011  |

The model converged successfully and achieved perfect accuracy on the training set.


📚 Files Explained
- train_model.py: Loads and preprocesses data, trains the CNN model.
- detect_sign.py: Loads the trained model and performs live prediction using webcam.
- model/asl_model.h5: Saved model after training.
- model/labels.txt: List of class labels used during training.

💡 Notes
- Make sure the dataset is organized with one folder per ASL letter inside dataset/asl data.
- Images should be .png format and clearly show the hand gesture.
- The model expects grayscale images resized to 64x64 pixels.





🧠 Handwritten Digit Recognition with CNN
This project features a Convolutional Neural Network (CNN) trained on the MNIST dataset and a Streamlit web interface. Users can either draw a digit on an interactive canvas or upload an image to see the model's prediction in real-time.

🚀 Features
Interactive Drawing: Real-time digit recognition using streamlit-drawable-canvas.

Image Upload: Support for PNG, JPG, and JPEG files.

CNN Model: High-accuracy model built with TensorFlow/Keras.

Probability Analysis: Visualizes the model's confidence across all 10 digits (0–9).

🛠️ Installation
Clone the Repository:

Bash

git clone https://github.com/your-username/digit-recognition-cnn.git
cd digit-recognition-cnn
Install Dependencies: Make sure you have Python 3.9+ installed.

Bash

pip install tensorflow streamlit streamlit-drawable-canvas numpy pillow pandas
📖 Usage
The project is split into two parts: Training and Deployment.

1. Train the Model
First, generate the trained model file (digit_model.h5).

Bash

python model.py
This script downloads the MNIST dataset.

Trains a CNN with 2 Convolutional layers.

Saves the final weights to digit_model.h5.

2. Run the Web App
Once you have the .h5 file, launch the Streamlit interface:

Bash

streamlit run app.py
🏗️ Model Architecture
The CNN architecture used for this project is optimized for grayscale 28x28 images:

Conv2D (32 filters, 3x3): Extracts basic features.

MaxPooling2D: Reduces spatial dimensions.

Conv2D (64 filters, 3x3): Extracts complex patterns.

Flatten: Converts 2D maps to a 1D vector.

Dense (128 units): Fully connected layer for classification.

Softmax Output: Returns probabilities for digits 0–9.

📂 Project Structure
model.py: The training script for the CNN.

app.py: The Streamlit application code.

digit_model.h5: The saved model (generated after training).

requirements.txt: List of required Python packages.

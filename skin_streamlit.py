import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import requests  # For downloading the model from Google Drive
import os  # To check if file already exists

# Set up Streamlit page
st.set_page_config(page_title="Skin Disease Detection", layout="centered")
st.title("Skin Disease Detection App")
st.write("Upload an image to detect the type of skin disease.")

# Function to download the model from Google Drive
def download_model_from_gdrive(file_id, destination):
    if not os.path.exists(destination):  # Download only if the file doesn't exist
        download_url = f"https://drive.google.com/file/d/1QkkAcXnK3sQW--yI8O-PmJsCQKP6BtRO/view?usp=sharing"
        response = requests.get(download_url)
        if response.status_code == 200:
            with open(destination, "wb") as f:
                f.write(response.content)
            print("Model downloaded successfully.")
        else:
            st.error("Failed to download the model from Google Drive.")
    else:
        print("Model already exists locally.")

# Load ONNX Model
@st.cache_resource
def load_model():
    file_id = "1QkkAcXnK3sQW--yI8O-PmJsCQKP6BtRO"  # Replace this with your actual Google Drive file ID
    model_path = "model.onnx"
    download_model_from_gdrive(file_id, model_path)  # Download at runtime
    session = ort.InferenceSession(model_path)
    return session

session = load_model()

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((227, 227))  # Resize to 227x227
    img_array = np.array(image).astype('float32')
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.transpose(img_array, (0, 3, 1, 2))  # Convert to NCHW format for ONNX
    return img_array

# Predict function
def predict(image):
    img_array = preprocess_image(image)
    inputs = {session.get_inputs()[0].name: img_array}
    outputs = session.run(None, inputs)
    probabilities = outputs[0][0]  # Get the output scores
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]
    return predicted_class, confidence

# Class labels (Update these based on your dataset labels)
class_labels = ['Acne', 'Hives', 'No Skin Disease']

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Predict button
    if st.button('Predict'):
        predicted_class, confidence = predict(image)
        st.success(f"Predicted Class: {class_labels[predicted_class]}\nConfidence: {confidence*100:.2f}%")

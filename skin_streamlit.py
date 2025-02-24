import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import gdown
import os

# Set up Streamlit page
st.set_page_config(page_title="Skin Disease Detection", layout="centered")
st.title("Skin Disease Detection App")
st.write("Upload an image to detect the type of skin disease.")

# Function to download the model from Google Drive
@st.cache_resource
def download_and_load_model():
    model_path = "model.onnx"
    file_id = "1QkkAcXnK3sQW--yI8O-PmJsCQKP6BtRO"
    download_url = f"https://drive.google.com/uc?id={file_id}"

   

    # Download if not already present
    if not os.path.exists(model_path):
        gdown.download(download_url, model_path, quiet=False)

    # Load the ONNX model
    session = ort.InferenceSession(model_path)
    return session

session = download_and_load_model()

# Function to preprocess the image
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

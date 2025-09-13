# streamlit_app.py
import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image

# ---------------------------
# Header image
# ---------------------------
st.image(
    "https://i.ytimg.com/vi/u2bg8OwB4Z0/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&amp;rs=AOn4CLCqhObtNshXtK9_YbOZSWYuJ2lNUQ",
    use_column_width=True
)

# ---------------------------
# Title and description
# ---------------------------
st.title("Potato Leaf Disease Classifier")
st.write("Upload an image of a potato leaf to predict its disease")

# ---------------------------
# Load the model
# ---------------------------
with open("model/potato_model.pkl", "rb") as f:
    model = pickle.load(f)

# Class names (match your training classes)
class_names = ['Early_blight', 'Late_blight', 'healthy']

# ---------------------------
# Image uploader and inference
# ---------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = image.load_img(uploaded_file, target_size=(128,128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)
    class_index = np.argmax(pred, axis=1)[0]
    predicted_class = class_names[class_index]
    confidence = np.max(pred) * 100

    # Display results
    st.success(f"Predicted Disease: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

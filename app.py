import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load the trained model
model = tf.keras.models.load_model("tiger_model.h5")

# Upload folder (optional, Streamlit can handle files in memory)
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

st.title("🐯 Tiger Detection System")
st.write("Upload an image to check if it contains a tiger.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file to static/uploads
    image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(image_path, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array, verbose=0)[0][0]

    if prediction >= 0.5:
        result = "Tiger"
        confidence = round(prediction * 100, 2)
    else:
        result = "Not Tiger"
        confidence = round((1 - prediction) * 100, 2)

    # Show results
    st.success(f"Prediction: {result} ({confidence}% confidence)")

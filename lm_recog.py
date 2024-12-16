import streamlit as st
import PIL
import tensorflow as tf
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim

# Load labels from CSV
labels_path = 'landmarks_classifier_asia_V1_label_map.csv'
df = pd.read_csv(labels_path)
labels = dict(zip(df.id, df.name))

# Load the model from the 'models' directory
MODEL_DIR = './models'
model = tf.saved_model.load(MODEL_DIR)

def image_processing(image):
    img_shape = (321, 321)  # Expected input shape for the model
    img = PIL.Image.open(image).convert("RGB")  # Ensure RGB channels
    img = img.resize(img_shape)  # Resize to required shape
    img1 = img.copy()  # Copy for display purposes
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Use the 'default' signature for predictions
    predictions = model.signatures["default"](tf.constant(img))
    logits = predictions["predictions:logits"].numpy()  # Extract logits

    # Find the class ID with the highest probability
    prediction_id = np.argmax(logits)
    return labels[prediction_id], img1  # Return predicted label and processed image


def get_map(loc):
    geolocator = Nominatim(user_agent="Your_Name")
    location = geolocator.geocode(loc)
    return location.address, location.latitude, location.longitude

def run():
    st.title("Landmark Recognition")
    img = PIL.Image.open('logo.png')
    img = img.resize((256, 256))
    st.image(img)
    img_file = st.file_uploader("Choose your Image", type=['png', 'jpg'])
    if img_file is not None:
        save_image_path = './Uploaded_Images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())
        prediction, image = image_processing(save_image_path)
        st.image(image)
        st.header("📍 **Predicted Landmark is: " + prediction + '**')
        try:
            address, latitude, longitude = get_map(prediction)
            st.success('Address: ' + address)
            loc_dict = {'Latitude': latitude, 'Longitude': longitude}
            st.subheader('✅ **Latitude & Longitude of ' + prediction + '**')
            st.json(loc_dict)
            data = [[latitude, longitude]]
            df = pd.DataFrame(data, columns=['lat', 'lon'])
            st.subheader('✅ **' + prediction + ' on the Map**' + '🗺️')
            st.map(df)
        except Exception as e:
            st.warning("No address found!!")
run()

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

# Print available signature keys
print("Available signatures:", model.signatures.keys())

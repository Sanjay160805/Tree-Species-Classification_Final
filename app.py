import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model(r"C:\Users\hp\Downloads\imp1_cnn_mod_updated.keras")

# All 30 species class names (in the order your model was trained)
class_names = [
    'neem', 'other', 'gunda', 'vad', 'pipal', 'champa', 'garmalo', 'kesudo',
    'bili', 'nilgiri', 'sugarcane', 'coconut', 'gulmohor', 'simlo', 'sitafal',
    'banyan', 'shirish', 'asopalav', 'jamun', 'mango', 'amla', 'kanchan',
    'saptaparni', 'motichanoti', 'bamboo', 'babul', 'sonmahor', 'cactus',
    'pilikaren', 'khajur'
]

# Image preprocessing
def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))  # Adjust if your model input size is different
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Streamlit UI
st.set_page_config(page_title="🌿 Tree Species Classifier")
st.title("🌳 Tree Species Identification")
st.write("Upload a leaf or tree image to detect its species.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    with st.spinner("Predicting..."):
        img_array = preprocess_image(uploaded_file)
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        species_name = class_names[class_index]
        confidence = float(np.max(prediction)) * 100

    st.success(f"✅ Predicted Species: **{species_name.capitalize()}**")
    st.info(f"🔍 Confidence: **{confidence:.2f}%**")

import numpy as np
import cv2
from PIL import Image
import streamlit as st
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource  # Cache model so it's not reloaded every time
def load_model():
    base_model = VGG19(include_top=False, input_shape=(128,128,3))
    x = base_model.output
    flat = Flatten()(x)
    class_1 = Dense(4608, activation='relu')(flat)
    drop_out = Dropout(0.2)(class_1)
    class_2 = Dense(1152, activation='relu')(drop_out)
    output = Dense(2, activation='softmax')(class_2)
    model = Model(base_model.inputs, output)
    model.load_weights("vgg_unfrozen.h5")  # ensure this file is in same folder
    return model

model_03 = load_model()

# ----------------------------
# Helper functions
# ----------------------------
def get_className(classNo):
    if classNo == 0:
        return "Normal"
    elif classNo == 1:
        return "Pneumonia"

def getResult(img):
    image = Image.open(img).convert("RGB")
    image = image.resize((128, 128))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model_03.predict(input_img)
    result01 = np.argmax(result, axis=1)
    return result01[0]

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🩻 Pneumonia Detection using VGG19")
st.write("Upload a chest X-ray image, and the model will predict whether it's *Normal* or *Pneumonia*.")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Chest X-ray", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing Image..."):
            value = getResult(uploaded_file)
            result = get_className(value)
        st.success(f"### Prediction: {result}")
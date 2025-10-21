# ==============================================================================
# CliniScan: Lung-Abnormality Classification & Detection 
# ==============================================================================

import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from ultralytics import YOLO
import gdown
import plotly.graph_objects as go
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
CLASS_MODEL_PATH = "resnet50_multilabel_vinbigdata.pt"
CLASS_DRIVE_ID = "1CG3_0OJe5hc2JyXsyerHc0DZcSTfKt1v"
CLASS_MODEL_URL = f"https://drive.google.com/uc?id={CLASS_DRIVE_ID}"

DETECT_MODEL_PATH = "best.pt"
DETECT_DRIVE_ID = "1Tt7-qfGC8509TGZTMIT_cWIovgVyRiyc"
DETECT_MODEL_URL = f"https://drive.google.com/uc?id={DETECT_DRIVE_ID}"

VALIDATOR_MODEL_PATH = "validator.h5"
VALIDATOR_DRIVE_ID = "19bfxZ6qirYPHA3wb09pzJh_4ePm92zgX"
VALIDATOR_MODEL_URL = f"https://drive.google.com/uc?id={VALIDATOR_DRIVE_ID}"

LOGO_PATH = "assets/logo.jpg"

classes = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis"
]

# ------------------------------------------------------------------------------
# Page Setup
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="CliniScan: Lung-Abnormality Analysis",
    layout="centered",
    initial_sidebar_state="auto"
)

# ------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image(LOGO_PATH, use_container_width=True)
    st.markdown("### Select Mode")
    mode = st.radio("Choose analysis type", ["Classification", "Detection"])
    st.markdown("---")
    st.markdown("### Confidence Threshold")
    confidence_threshold = st.slider(
        "Select minimum confidence", 0.0, 1.0, 0.5 if mode == "Classification" else 0.25, 0.05
    )
    st.markdown("---")
    st.markdown("#### About the Model")
    if mode == "Classification":
        st.info("""
        ResNet50-based multi-label classifier trained to identify multiple lung abnormalities from chest X-ray images.
        """)
    else:
        st.info("""
        YOLOv8 object detection model trained to identify 14 types of lung abnormalities from chest X-ray images.
        """)
    st.markdown("---")
    st.markdown("#### Disclaimer")
    st.warning("""
    For educational/research purposes only. Not a substitute for professional medical advice.
    """)

# ------------------------------------------------------------------------------
# Download model weights if not present
# ------------------------------------------------------------------------------
if not os.path.exists(CLASS_MODEL_PATH):
    with st.spinner("Downloading classification model... Please wait"):
        gdown.download(CLASS_MODEL_URL, CLASS_MODEL_PATH, quiet=False)

if not os.path.exists(DETECT_MODEL_PATH):
    with st.spinner("Downloading detection model... Please wait"):
        gdown.download(DETECT_MODEL_URL, DETECT_MODEL_PATH, quiet=False)

if not os.path.exists(VALIDATOR_MODEL_PATH):
    with st.spinner("Downloading validator model... Please wait"):
        gdown.download(VALIDATOR_MODEL_URL, VALIDATOR_MODEL_PATH, quiet=False)

# ------------------------------------------------------------------------------
# Load Models
# ------------------------------------------------------------------------------
@st.cache_resource
def load_classification_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(CLASS_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device

@st.cache_resource
def load_detection_model():
    return YOLO(DETECT_MODEL_PATH)

@st.cache_resource
def load_validator_model():
    return load_model(VALIDATOR_MODEL_PATH)

# Load all models
model_class, device = load_classification_model()
model_detect = load_detection_model()
validator_model = load_validator_model()

# ------------------------------------------------------------------------------
# Image Transform for Classification
# ------------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ------------------------------------------------------------------------------
# Validator Function
# ------------------------------------------------------------------------------
def is_chest_xray(image_pil, validator_model):
    """Checks if an uploaded image is a chest X-ray using the validator model."""
    img = image_pil.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array_expanded)
    prediction = validator_model.predict(img_preprocessed, verbose=0)[0][0]
    # Interpretation: low score (~0) = chest X-ray, high (~1) = not chest X-ray
    return prediction < 0.5

# ------------------------------------------------------------------------------
# Main Interface
# ------------------------------------------------------------------------------
st.markdown("## CliniScan: Lung-Abnormality Analysis")
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -------------------- Step 1: Validate Image --------------------
    with st.spinner("Validating image type..."):
        valid = is_chest_xray(image, validator_model)

    if not valid:
        st.error("The uploaded image does not appear to be a chest X-ray.")
        st.warning("Please upload a valid chest X-ray image to continue.")
        st.stop()
    else:
        st.success("Image validated as Chest X-ray. Proceeding with analysis...")

    # -------------------- Step 2: Detection --------------------
    with st.spinner("Running detection in background..."):
        results = model_detect.predict(source=image, imgsz=640, conf=0.25)
    res = results[0]

    detected_classes = []
    if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
        cls_list = res.boxes.cls.cpu().numpy()
        conf_list = res.boxes.conf.cpu().numpy()
        for cls_id, conf in zip(cls_list, conf_list):
            if conf >= 0.25:
                detected_classes.append((classes[int(cls_id)], float(conf)))

    # -------------------- Step 3: Classification --------------------
    if mode == "Classification":
        if len(detected_classes) == 0:
            st.success("No abnormalities detected.")
        else:
            img_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = torch.sigmoid(model_class(img_tensor)).cpu().numpy()[0]
            class_preds = [(cls_name, float(conf)) for cls_name, conf in zip(classes, outputs) if conf >= confidence_threshold]

            for det_cls, det_conf in detected_classes:
                if det_cls not in [c[0] for c in class_preds]:
                    class_preds.append((det_cls, det_conf))

            if class_preds:
                st.subheader("Predicted Abnormalities")
                df = pd.DataFrame(sorted(class_preds, key=lambda x: x[1], reverse=True),
                                  columns=["Abnormality", "Confidence"])
                st.table(df)

                fig = go.Figure(go.Bar(
                    x=df["Confidence"],
                    y=df["Abnormality"],
                    orientation='h',
                    marker=dict(color=df["Confidence"], colorscale='YlGnBu'),
                    hovertemplate='<b>%{y}</b><br>Confidence: %{x:.2f}<extra></extra>'
                ))
                fig.update_layout(title="Confidence Scores by Abnormality",
                                  xaxis_title="Confidence",
                                  yaxis_title="Abnormality",
                                  template="plotly_white",
                                  height=500)
                st.plotly_chart(fig, use_container_width=True)

    # -------------------- Step 4: Detection Mode --------------------
    else:
        annotated = res.plot()
        st.image(annotated, caption="Detected Abnormalities", use_container_width=True)

        if len(detected_classes) > 0:
            st.subheader("Detected Findings")
            det_summary = {}
            for cls_name, conf in detected_classes:
                det_summary[cls_name] = max(det_summary.get(cls_name, 0), conf)
                st.write(f"- {cls_name} ({conf:.2f})")

            df = pd.DataFrame(sorted(det_summary.items(), key=lambda x: x[1], reverse=True),
                              columns=["Abnormality", "Confidence"])
            fig = go.Figure(go.Bar(
                x=df["Confidence"],
                y=df["Abnormality"],
                orientation='h',
                marker=dict(color=df["Confidence"], colorscale='YlGnBu',
                            line=dict(color='rgba(0,0,0,0.6)', width=1)),
                hovertemplate='<b>%{y}</b><br>Confidence: %{x:.2f}<extra></extra>'
            ))
            fig.update_layout(
                title="Confidence Scores by Abnormality",
                xaxis_title="Confidence",
                yaxis_title="Abnormality",
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No abnormalities detected.")

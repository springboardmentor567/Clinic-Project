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

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
# Classification model
CLASS_MODEL_PATH = "resnet50_multilabel_vinbigdata.pt"
CLASS_DRIVE_ID = "1CG3_0OJe5hc2JyXsyerHc0DZcSTfKt1v"
CLASS_MODEL_URL = f"https://drive.google.com/uc?id={CLASS_DRIVE_ID}"

# Detection model
DETECT_MODEL_PATH = "best.pt"
DETECT_DRIVE_ID = "1Tt7-qfGC8509TGZTMIT_cWIovgVyRiyc"
DETECT_MODEL_URL = f"https://drive.google.com/uc?id={DETECT_DRIVE_ID}"

LOGO_PATH = "assets/logo.jpg"  # Replace with your actual logo path

# Define the 14 class names
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
        "Select minimum confidence", 0.0, 1.0, 0.5 if mode=="Classification" else 0.25, 0.05
    )
    st.markdown("---")
    st.markdown("#### About the Model")
    if mode=="Classification":
        st.info("""
        This application uses a ResNet50-based multi-label classifier trained to identify multiple lung abnormalities from chest X-ray images.
        """)
    else:
        st.info("""
        This application uses a YOLOv8 object detection model trained to identify 14 types of lung abnormalities from chest X-ray images.
        Model architecture: YOLOv8n  
        Framework: Ultralytics YOLO
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

# Load models
if mode == "Classification":
    model, device = load_classification_model()
else:
    model = load_detection_model()

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
# Main Interface
# ------------------------------------------------------------------------------
st.markdown("## CliniScan: Lung-Abnormality Analysis")
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if mode == "Classification":
        # Multi-label classification
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = torch.sigmoid(model(img_tensor)).cpu().numpy()[0]

        preds = [(cls_name, float(conf)) for cls_name, conf in zip(classes, outputs) if conf >= confidence_threshold]

        if preds:
            st.subheader("Predicted Abnormalities")
            preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
            df = pd.DataFrame(preds_sorted, columns=["Abnormality", "Confidence"])
            st.table(df)

            # Horizontal bar chart
            fig = go.Figure(go.Bar(
                x=df["Confidence"],
                y=df["Abnormality"],
                orientation='h',
                marker=dict(color=df["Confidence"], colorscale='YlGnBu'),
                hovertemplate='<b>%{y}</b><br>Confidence: %{x:.2f}<extra></extra>'
            ))
            fig.update_layout(title="Confidence Scores by Abnormality",
                              xaxis_title="Confidence", yaxis_title="Abnormality",
                              template="plotly_white", height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No abnormalities detected above the selected confidence threshold.")

    else:
        # Object detection with YOLO
        results = model.predict(image, imgsz=640, conf=confidence_threshold)
        res = results[0]
        annotated = res.plot()
        st.image(annotated, caption="Detected Abnormalities", use_container_width=True)

        if len(res.boxes) > 0:
            st.subheader("Detected Findings")
            det_summary = {}
            for box in res.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if conf >= confidence_threshold:
                    cls_name = classes[cls_id]
                    det_summary[cls_name] = max(det_summary.get(cls_name, 0), conf)
                    st.write(f"- {cls_name} ({conf:.2f})")

            # Horizontal bar chart
            sorted_detections = sorted(det_summary.items(), key=lambda x: x[1], reverse=True)
            df = pd.DataFrame(sorted_detections, columns=["Abnormality", "Confidence"])
            fig = go.Figure(go.Bar(
                x=df["Confidence"],
                y=df["Abnormality"],
                orientation='h',
                marker=dict(color=df["Confidence"], colorscale='YlGnBu',
                            line=dict(color='rgba(0, 0, 0, 0.6)', width=1)),
                hovertemplate='<b>%{y}</b><br>Confidence: %{x:.2f}<extra></extra>'
            ))
            fig.update_layout(
                title="Confidence Scores by Abnormality",
                xaxis_title="Confidence",
                yaxis_title="Abnormality",
                template="plotly_white",
                height=500,
                margin=dict(l=100, r=40, t=60, b=40),
                font=dict(family="Helvetica", size=12, color="#212529"), 
                xaxis=dict(gridcolor='rgba(0,0,0,0.1)', zerolinecolor='rgba(0,0,0,0.2)'),
                yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No abnormalities detected above the selected confidence threshold.")

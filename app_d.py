# ==============================================================================
# CliniScan: Lung-Abnormality Detection on Chest X-rays
# ==============================================================================
import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import gdown
import plotly.graph_objects as go
import pandas as pd

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
MODEL_PATH = "best.pt"
DRIVE_ID = "1Tt7-qfGC8509TGZTMIT_cWIovgVyRiyc"
MODEL_URL = f"https://drive.google.com/uc?id={DRIVE_ID}"
LOGO_PATH = "assets/logo.jpg"  # Replace with your actual logo path

# ------------------------------------------------------------------------------
# Page Setup
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="CliniScan: Lung-Abnormality Detection on Chest X-rays",
    layout="centered",
    initial_sidebar_state="auto"
)

# ------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image(LOGO_PATH, use_container_width=True)
    st.markdown("### Confidence Threshold")
    confidence_threshold = st.slider("Select minimum confidence", 0.0, 1.0, 0.25, 0.05)
    st.markdown("---")
    st.markdown("#### About the Model")
    st.info("""
    This application uses a YOLOv8 object detection model trained to identify 14 types of lung abnormalities from chest X-ray images.
            

    Model architecture: YOLOv8n  

    Framework: Ultralytics YOLO  
    """)
    st.markdown("---")
    st.markdown("#### Disclaimer")
    st.warning("""
    This tool is intended for educational and research purposes only.  
    It is not a substitute for professional medical advice, diagnosis, or treatment.  
    Always consult a qualified healthcare provider for clinical decisions.
    """)

# ------------------------------------------------------------------------------
# Download model weights if not present
# ------------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... Please wait"):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ------------------------------------------------------------------------------
# Load YOLO model
# ------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ------------------------------------------------------------------------------
# Class Names
# ------------------------------------------------------------------------------
class_names = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis"
]

# ------------------------------------------------------------------------------
# Main Interface
# ------------------------------------------------------------------------------
st.markdown("## CliniScan: Lung-Abnormality Detection on Chest X-rays")
uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

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
                cls_name = class_names[cls_id]
                det_summary[cls_name] = max(det_summary.get(cls_name, 0), conf)
                st.write(f"- {cls_name} ({conf:.2f})")

        # Sort and prepare data
        sorted_detections = sorted(det_summary.items(), key=lambda x: x[1], reverse=True)
        df = pd.DataFrame(sorted_detections, columns=["Abnormality", "Confidence"])

        # Enhanced medical-grade horizontal bar chart
        fig = go.Figure(go.Bar(
            x=df["Confidence"],
            y=df["Abnormality"],
            orientation='h',
            marker=dict(
                color=df["Confidence"],
                colorscale='YlGnBu',
                line=dict(color='rgba(0, 0, 0, 0.6)', width=1)
            ),
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
            plot_bgcolor="#F0F2F6",
            paper_bgcolor="#FFFFFF",
            xaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                zerolinecolor='rgba(0,0,0,0.2)'
            ),
            yaxis=dict(
                gridcolor='rgba(0,0,0,0.1)'
            )
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No abnormalities detected above the selected confidence threshold.")

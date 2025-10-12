
# ==============================================================================
# CliniScan: Lung-Abnormality Classification on Chest X-rays (ResNet-18)
# ==============================================================================
import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import gdown
import plotly.graph_objects as go

# ------------------------------------------------------------------------------
# Page Setup
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="CliniScan-RN18",
    layout="centered",
    initial_sidebar_state="auto"
)

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
MODEL_PATH = "best_classification_model.pth"
DRIVE_ID = "1yW1qHFFwNO8BBxqoUCrFVBRrjwAqGDsJ"
MODEL_URL = f"https://drive.google.com/uc?id={DRIVE_ID}"
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.jpg")

CLASS_NAMES = ["Abnormal", "Normal"]
IMG_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# ------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    st.markdown("### Confidence Threshold")
    confidence_threshold = st.slider("Select minimum confidence", 0.0, 1.0, 0.5, 0.05)
    st.markdown("---")
    st.markdown("#### About the Model")
    st.info("""
    This tool uses a **ResNet-18** deep learning model trained on chest X-ray images 
    to classify them as **Normal** or **Abnormal**.
    

    Model: ResNet-18  

    
    Framework: PyTorch  
    """)
    st.markdown("---")
    st.markdown("#### Disclaimer")
    st.warning("""
    This application is for **educational and research** purposes only.  
    It is **not** a substitute for professional medical advice, diagnosis, or treatment.  
    Always consult a qualified healthcare provider for medical decisions.
    """)

# ------------------------------------------------------------------------------
# Download Model Weights
# ------------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ------------------------------------------------------------------------------
# Load Model
# ------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ------------------------------------------------------------------------------
# Image Preprocessing
# ------------------------------------------------------------------------------
@st.cache_data
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
    return transform(image).unsqueeze(0)

# ------------------------------------------------------------------------------
# Grad-CAM Implementation
# ------------------------------------------------------------------------------
def generate_gradcam(model, input_tensor, target_class):
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    layer = model.layer4[1].conv2
    fwd_handle = layer.register_forward_hook(forward_hook)
    bwd_handle = layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    model.zero_grad()
    loss = output[0, target_class]
    loss.backward()

    grads = gradients[0].detach().numpy()[0]
    acts = activations[0].detach().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i, :, :]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    fwd_handle.remove()
    bwd_handle.remove()

    return cam

def overlay_cam_on_image(img, cam):
    img = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return overlay

# ------------------------------------------------------------------------------
# Prediction Function
# ------------------------------------------------------------------------------
def predict(model, image):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_class = torch.argmax(probs).item()
    return probs, pred_class

# ------------------------------------------------------------------------------
# Main Interface
# ------------------------------------------------------------------------------
st.markdown("## CliniScan: Lung-Abnormality Classification on Chest X-rays")

uploaded_file = st.file_uploader("Upload a Chest X-ray image to classify it as **Normal** or **Abnormal**.", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display image
    st.subheader("Uploaded X-ray")
    st.image(image, use_container_width=True)

    # Prediction
    st.subheader("Prediction Results")
    probs, pred_class = predict(model, image)
    confidence = probs[pred_class].item()

    if confidence >= confidence_threshold:
        st.success(f"Prediction: **{CLASS_NAMES[pred_class]}**")
    else:
        st.warning(f"Low confidence prediction: **{CLASS_NAMES[pred_class]}**")
    st.markdown(f"**Confidence:** {confidence * 100:.2f}%")

    # Plotly horizontal bar chart below predictions
    prob_df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Confidence": probs.numpy()
    }).sort_values("Confidence", ascending=True)

    st.markdown("---")
    st.subheader("Class Probabilities")
    fig = go.Figure(go.Bar(
        x=prob_df["Confidence"],
        y=prob_df["Class"],
        orientation='h',
        marker=dict(
            color=prob_df["Confidence"],
            colorscale='YlGnBu',
            line=dict(color='rgba(0,0,0,0.6)', width=1)
        ),
        hovertemplate='<b>%{y}</b><br>Confidence: %{x:.3f}<extra></extra>'
    ))
    fig.update_layout(
        xaxis_title="Confidence",
        yaxis_title="Class",
        template="plotly_white",
        height=400,
        margin=dict(l=150, r=40, t=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Grad-CAM visualization below chart
    st.markdown("---")
    st.subheader("Grad-CAM Heatmap")
    cam = generate_gradcam(model, preprocess_image(image), pred_class)
    overlay = overlay_cam_on_image(image, cam)
    st.image(overlay, caption="Grad-CAM Visualization", use_container_width=True)


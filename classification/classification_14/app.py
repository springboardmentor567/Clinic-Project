# ==============================================================================
# CliniScan: Lung-Abnormality Classification on Chest X-rays (EfficientNet)
# ==============================================================================
import os
import streamlit as st
import torch
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd
import gdown
import plotly.graph_objects as go

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
IMG_SIZE = 512
N_CLASSES = 14
MODEL_NAME = 'efficientnet-b0'
MODEL_WEIGHTS_PATH = 'best_classification.pth'
BASE_DIR = os.path.dirname(os.path.dirname(__file__)) 
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.jpg")
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

DRIVE_ID = '1Mr4ojGw6djSPVBrPVFyiSDb0o9HHOji-'
MODEL_URL = f"https://drive.google.com/uc?id={DRIVE_ID}"

idx_to_class = {
    0: 'Aortic enlargement', 
    1: 'Cardiomegaly', 
    2: 'Pulmonary fibrosis', 
    3: 'Pneumothorax',
    4: 'Pleural thickening', 
    5: 'Pleural effusion', 
    6: 'No finding', 
    7: 'Nodule/Mass', 
    8: 'Infiltration',
    9: 'ILD', 
    10: 'Other lesion', 
    11: 'Atelectasis', 
    12: 'Emphysema', 
    13: 'Calcification'
}

# ------------------------------------------------------------------------------
# Page Setup
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="CliniScan-ENB0",
    layout="centered",
    initial_sidebar_state="auto"
)

# ------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image(LOGO_PATH, use_container_width=True)
    st.markdown("### Confidence Threshold")
    confidence_threshold = st.slider("Select minimum confidence", 0.0, 1.0, 0.5, 0.05)
    st.markdown("---")
    st.markdown("#### About the Model")
    st.info("""
    This application uses an EfficientNet-B0 classification model trained to identify 14 types of lung abnormalities from chest X-ray images.
    

    Model architecture: EfficientNet-B0  

    
    Framework: PyTorch  
    """)
    st.markdown("---")
    st.markdown("#### Disclaimer")
    st.warning("""
    This tool is intended for educational and research purposes only.  
    It is not a substitute for professional medical advice, diagnosis, or treatment.  
    Always consult a qualified healthcare provider for clinical decisions.
    """)

# ------------------------------------------------------------------------------
# Download Model if Not Present
# ------------------------------------------------------------------------------
if not os.path.exists(MODEL_WEIGHTS_PATH):
    with st.spinner("Downloading model weights..."):
        gdown.download(MODEL_URL, MODEL_WEIGHTS_PATH, quiet=False)

# ------------------------------------------------------------------------------
# Model Architecture
# ------------------------------------------------------------------------------
class VinBigDataClassifier(torch.nn.Module):
    def __init__(self, n_classes, model_name):
        super().__init__()
        self.model = EfficientNet.from_pretrained(model_name)
        num_ftrs = self.model._fc.in_features
        self.model._fc = torch.nn.Linear(num_ftrs, n_classes)

    def forward(self, x):
        return self.model(x)

# ------------------------------------------------------------------------------
# Load Model
# ------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = VinBigDataClassifier(n_classes=N_CLASSES, model_name=MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    try:
        checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith('model.'):
                new_state_dict['model.' + k] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device
    
    model.eval()
    return model, device

# ------------------------------------------------------------------------------
# Image Preprocessing
# ------------------------------------------------------------------------------
@st.cache_data
def preprocess_image(image):
    transforms_pipeline = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
    return transforms_pipeline(image).unsqueeze(0)

# ------------------------------------------------------------------------------
# Prediction Function
# ------------------------------------------------------------------------------
def predict(model, device, image):
    img_tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    predicted_indices = np.where(probabilities > confidence_threshold)[0]
    results = []

    if len(predicted_indices) == 0:
        max_idx = np.argmax(probabilities)
        results.append(f"{idx_to_class[max_idx]} (Fallback: {probabilities[max_idx]:.3f})")
    else:
        for i in predicted_indices:
            results.append(f"{idx_to_class[i]} ({probabilities[i]:.3f})")
    
    return results, probabilities

# ------------------------------------------------------------------------------
# Main Interface
# ------------------------------------------------------------------------------
st.markdown("## CliniScan: Lung-Abnormality Classification on Chest X-rays")
uploaded_file = st.file_uploader("Upload a Chest X-ray image to classify potential lung abnormalities and view their confidence scores.", type=["jpg", "jpeg", "png"])

model, device = load_model()
if model is None:
    st.stop()

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.subheader("Prediction Results")
    results, probabilities = predict(model, device, image)

    for res in results:
        st.write(f"- {res}")

    # Prepare DataFrame for Plotly bar chart
    prob_df = pd.DataFrame({
        "Abnormality": list(idx_to_class.values()),
        "Confidence": probabilities
    }).sort_values("Confidence", ascending=True)  # horizontal chart, smallest at bottom

    # Plotly horizontal bar chart
    fig = go.Figure(go.Bar(
        x=prob_df["Confidence"],
        y=prob_df["Abnormality"],
        orientation='h',
        marker=dict(
            color=prob_df["Confidence"],
            colorscale='YlGnBu',
            line=dict(color='rgba(0,0,0,0.6)', width=1)
        ),
        hovertemplate='<b>%{y}</b><br>Confidence: %{x:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title="Confidence Scores by Abnormality",
        xaxis_title="Confidence",
        yaxis_title="Abnormality",
        template="plotly_white",
        height=500,
        margin=dict(l=150, r=40, t=60, b=40),
        font=dict(family="Helvetica", size=12, color="#212529"),
        xaxis=dict(gridcolor='rgba(0,0,0,0.1)', zerolinecolor='rgba(0,0,0,0.2)'),
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
    )

    st.plotly_chart(fig, use_container_width=True)

import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import base64

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Import the model definition from model.py
from model import ResNetClassifier

# --- Configuration ---
MODEL_PATH = "model.pth"
NUM_CLASSES = 2
CLASS_NAMES = ["Normal", "Pneumonia"]
LOGO_PATH = "assets/logo.jpg"

# --- Helper Functions ---

@st.cache_data
def get_base64_of_bin_file(bin_file):
    """Encodes a binary file to base64 to embed in HTML."""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def create_html_confidence_bar(probabilities):
    """
    Creates a robust, pure HTML/CSS confidence bar.
    This avoids Matplotlib rendering issues.
    """
    predicted_class_index = np.argmax(probabilities.numpy())
    confidence = probabilities[predicted_class_index].item() * 100
    
    # Determine color based on prediction
    main_color = '#4CAF50' if predicted_class_index == 0 else '#F44336' # Green for Normal, Red for Pneumonia
    
    # Simplified HTML structure for the bar
    bar_html = f"""
    <div style="background-color: #262730; border-radius: 5px; height: 30px; width: 100%; padding: 0;">
        <div style="background-color: {main_color}; width: {confidence}%; border-radius: 5px; height: 100%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
            {confidence:.2f}%
        </div>
    </div>
    """
    return bar_html

# --- Core Functions ---

@st.cache_resource
def load_model(model_path, num_classes):
    """Loads the trained PyTorch model."""
    model = ResNetClassifier(num_classes=num_classes)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except Exception:
        pass # A warning is shown in the main UI if the model fails to load.
    model.eval()
    return model

def preprocess_image(image):
    """Applies the necessary transformations to the uploaded image."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def get_grad_cam(model, input_tensor, rgb_img, pred_class_index):
    """Generates and overlays a Grad-CAM heatmap on the image."""
    target_layers = [model.model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred_class_index)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization

# --- Main Application UI ---
st.set_page_config(page_title="CliniScan", page_icon="🩻", layout="wide")

# Custom CSS for the UI
st.markdown("""
    <style>
    .block-container {
        padding: 2rem 3rem 2rem 3rem;
    }
    #MainMenu, footer, .stDeployButton { display: none; }
    .stApp > header { background-color: transparent; }
    .info-box {
        background-color: #1c4e80; color: white; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; text-align: center;
    }
    .results-container {
        border: 1px solid #262730; border-radius: 0.5rem; padding: 1.5rem; margin-top: 1.5rem;
    }
    [data-testid="stMetric"] {
        background-color: #262730; border-radius: 0.5rem; padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
try:
    logo_base64 = get_base64_of_bin_file(LOGO_PATH)
    st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <img src="data:image/png;base64,{logo_base64}" width="70" style="margin-right: 20px;">
            <h1 style="margin: 0;">CliniScan: Lung Abnormality Detection</h1>
        </div>
        """, unsafe_allow_html=True)
except FileNotFoundError:
    st.error(f"Logo file not found at '{LOGO_PATH}'.")
    st.header("CliniScan: Lung Abnormality Detection")

# --- Description ---
st.markdown('<div class="info-box">Upload a chest X-ray to get a prediction for Normal vs. Pneumonia.</div>', unsafe_allow_html=True)

# --- Load Model and Check Status ---
model = load_model(MODEL_PATH, NUM_CLASSES)
model_loaded = any(p.requires_grad for p in model.parameters())
if not model_loaded:
    st.error("Model file 'model.pth' not found or is invalid. The app is in demonstration mode.")

# --- Main App Logic ---
st.subheader("1. Upload Chest X-Ray Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        input_tensor = preprocess_image(image)
        rgb_img_for_display = np.array(image.resize((224, 224))) / 255.0

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class_index = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class_index].item()
            predicted_class_name = CLASS_NAMES[predicted_class_index]

        # --- Display Results ---
        st.subheader("2. Analysis Results")
        with st.container():
            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            col1.metric(label="Predicted Class", value=predicted_class_name)
            col2.metric(label="Confidence", value=f"{confidence:.2%}")
            
            # Display the robust HTML confidence bar
            st.markdown(create_html_confidence_bar(probabilities), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("3. Explainability with Grad-CAM")
        grad_cam_image = get_grad_cam(model, input_tensor, rgb_img_for_display, predicted_class_index)
        
        col_orig, col_cam = st.columns(2)
        col_orig.image(image, caption="Original X-Ray", use_container_width=True)
        col_cam.image(grad_cam_image, caption=f"Grad-CAM Heatmap for '{predicted_class_name}'", use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
else:
    st.info("Awaiting image upload to begin analysis.")

# --- Sidebar ---
st.sidebar.title("About CliniScan")
st.sidebar.info("This app uses a ResNet18 model to classify chest X-rays. Grad-CAM visualizes the areas the model focused on for its prediction.")
st.sidebar.markdown("**Disclaimer:** This tool is for educational purposes only and is not a substitute for professional medical advice.")
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by: Shambhavi Singh")

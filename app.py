# app.py — CliniScan: Chest X-Ray Classification + Grad-CAM
# Put this file in the project root.

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import io
import cv2
import pandas as pd
import os
import tempfile

# Optional Grad-CAM imports (installed from GitHub if needed)
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except Exception as e:
    GRADCAM_AVAILABLE = False
    gradcam_import_error = e

# -----------------------
# DICOM -> PIL helper
# -----------------------
import pydicom
from io import BytesIO

def dicom_bytes_to_pil(dcm_bytes):
    """Convert DICOM bytes to PIL.Image (RGB)."""
    ds = pydicom.dcmread(BytesIO(dcm_bytes))
    arr = ds.pixel_array
    arr_norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    img = Image.fromarray(arr_norm).convert("RGB")
    return img

# -----------------------
# Model (ResNet18 wrapper)
# -----------------------
from torchvision.models import resnet18
try:
    from torchvision.models import ResNet18_Weights
    RESNET_WEIGHTS = ResNet18_Weights.DEFAULT
except Exception:
    RESNET_WEIGHTS = None

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        if RESNET_WEIGHTS is not None:
            base = resnet18(weights=RESNET_WEIGHTS)
        else:
            base = resnet18(pretrained=True)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model_from_path(model_path=None, num_classes=2):
    """Return model (attempt to load model_path if provided)."""
    model = ResNetClassifier(num_classes=num_classes)
    if model_path:
        try:
            sd = torch.load(model_path, map_location="cpu")
            if isinstance(sd, dict):
                model.load_state_dict(sd)
            else:
                model = sd
            st.success("✅ model loaded from file.")
        except Exception as e:
            st.warning(f"Could not load model.pth: {e}")
            st.info("Falling back to fresh ResNet.")
    model.eval()
    return model

# -----------------------
# Preprocessing
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="CliniScan — Chest X-Ray", layout="wide")
st.markdown('<h1 style="text-align:center;color:#00BFFF;">CLINISCAN — Chest X-Ray Abnormality Detection</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png", width=120)
    st.markdown("### Options")
    num_classes = st.number_input("Number of classes", value=2, min_value=2, max_value=20)
    upload_model = st.file_uploader("Upload trained model (.pt/.pth)", type=["pt","pth"])
    enable_gradcam = st.checkbox("Enable Grad-CAM", value=True)
    show_probs = st.checkbox("Show probabilities", value=True)

model_path = None
if upload_model is not None:
    tempf = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
    tempf.write(upload_model.getvalue())
    tempf.flush()
    model_path = tempf.name

model = load_model_from_path(model_path=model_path, num_classes=int(num_classes))

uploaded = st.file_uploader("Upload chest X-ray (jpg/png) or DICOM (.dcm)", type=["jpg","jpeg","png","dcm"])
if uploaded is None:
    st.info("Upload an image or DICOM to run prediction.")
else:
    try:
        name = uploaded.name.lower()
        if name.endswith(".dcm"):
            img = dicom_bytes_to_pil(uploaded.getvalue())
        else:
            img = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")

        st.image(img, caption="Input image", use_column_width=True)

        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_class = int(torch.argmax(probs).item())

        class_names = [f"Class_{i}" for i in range(int(num_classes))]
        if int(num_classes) == 2:
            class_names = ["Normal", "Pneumonia"]

        if show_probs:
            st.subheader("Prediction")
            st.write(f"**{class_names[pred_class]}** — {probs[pred_class].item()*100:.2f}%")
            df = pd.DataFrame({"Class": class_names, "Probability": [float(probs[i].item()) for i in range(len(probs))]})
            st.bar_chart(df.set_index("Class"))

        if enable_gradcam:
            if not GRADCAM_AVAILABLE:
                st.error("Grad-CAM not installed. Install via `pip install git+https://github.com/jacobgil/pytorch-grad-cam.git`")
            else:
                try:
                    # pick the last layer of ResNet
                    target_layers = [model.model.layer4[-1]]
                    cam = GradCAM(model=model.model, target_layers=target_layers)
                    rgb = np.array(img.resize((224,224))) / 255.0
                    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
                    visualization = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(img, caption="Original", use_column_width=True)
                    with col2:
                        st.image(visualization, caption="Grad-CAM", use_column_width=True)
                except Exception as e:
                    st.error(f"Grad-CAM failed: {e}")

        st.markdown("<p style='text-align:center;color:gray;'>Made with Streamlit & PyTorch</p>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Failed to process uploaded file: {e}")

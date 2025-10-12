import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os

# -------------------- Streamlit Page Config --------------------
st.set_page_config(
    page_title="Chest X-Ray Pneumonia Detection",
    layout="wide",
    initial_sidebar_state="auto",
    page_icon="🩺"
)

# -------------------- Custom CSS --------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #181c24;
        color: #f5f6fa;
    }
    .stButton>button {
        color: white;
        background: linear-gradient(90deg, #1e90ff 0%, #00b894 100%);
        border: none;
        border-radius: 8px;
        padding: 0.5em 2em;
        font-weight: bold;
    }
    .stFileUploader>div>div {
        background: #232946;
        color: #f5f6fa;
    }
    .stAlert {
        background: #232946;
        color: #f5f6fa;
    }
    .confidence-bar {
        width: 100%;
        height: 32px;
        background: #232946;
        border-radius: 8px;
        margin-bottom: 1em;
        position: relative;
    }
    .confidence-bar-inner {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #ff7675 0%, #00b894 100%);
        text-align: right;
        color: #fff;
        font-weight: bold;
        font-size: 1.1em;
        padding-right: 12px;
        line-height: 32px;
    }
    .disclaimer {
        background: #22313f;
        color: #f5f6fa;
        border-radius: 6px;
        padding: 0.7em 1em;
        margin: 1em 0;
        font-size: 1em;
        border-left: 6px solid #ff7675;
    }
    .localization-header {
        background: #feca57;
        color: #222f3e;
        border-radius: 6px;
        padding: 0.5em 1em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- Model Loading --------------------
MODEL_PATH = 'model.pth'

@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# -------------------- Grad-CAM Implementation --------------------
def generate_gradcam(model, img_tensor, target_class):
    gradients = []
    activations = []

    # Forward and backward hooks
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    final_conv = model.layer4[1].conv2
    handle_fw = final_conv.register_forward_hook(forward_hook)
    handle_bw = final_conv.register_backward_hook(backward_hook)

    # Ensure input requires grad
    img_tensor.requires_grad_()

    # Forward pass
    output = model(img_tensor)
    model.zero_grad()

    # Select target class (tensor, not .item())
    class_loss = output[0, target_class]

    # Backward pass
    class_loss.backward(retain_graph=True)

    # Compute Grad-CAM
    grads = gradients[0][0].cpu().detach().numpy()
    acts = activations[0][0].cpu().detach().numpy()
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam / cam.max() if cam.max() != 0 else cam

    # Remove hooks
    handle_fw.remove()
    handle_bw.remove()

    return cam

# -------------------- Image Preprocessing --------------------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# -------------------- Streamlit App UI --------------------
st.markdown("""
# 🩺 Chest X-Ray Pneumonia Detection
Upload a chest X-ray image to get a prediction and see the Grad-CAM heatmap.
""")

uploaded_file = st.file_uploader('**Choose a chest X-ray image**', type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded X-ray Image', use_column_width=True)

    img_tensor = preprocess_image(image)
    model = load_model()

    # -------------------- Inference --------------------
    output = model(img_tensor)
    pred_tensor = torch.argmax(output, dim=1)  # keep as tensor
    pred = pred_tensor.item()  # for display

    classes = ['Normal', 'Pneumonia']
    confidence = torch.softmax(output, dim=1).detach().cpu().numpy()[0]
    conf_percent = confidence[pred] * 100

    st.markdown(f"<h3>Prediction: {classes[pred]}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3>Confidence: {conf_percent:.2f}%</h3>", unsafe_allow_html=True)

    # Confidence bar
    st.markdown(f'''
    <div class="confidence-bar">
        <div class="confidence-bar-inner" style="width: {confidence[1]*100:.1f}%">
            {'PNEUMONIA' if pred==1 else 'NORMAL'}
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Confidence comparison
    st.markdown("**Model Confidence Comparison**")
    st.progress(float(confidence[1]))
    st.caption(f"PNEUMONIA: {confidence[1]*100:.2f}% | NORMAL: {confidence[0]*100:.2f}%")

    # Disclaimer
    st.markdown(
        '<div class="disclaimer">Disclaimer: This tool is an Prediction   and is <b>NOT</b> a substitute for professional medical diagnosis.</div>',
        unsafe_allow_html=True
    )

    # -------------------- Grad-CAM Visualization --------------------
    st.markdown("---")
    st.markdown('<div class="localization-header">Localization Map (Area of Interest)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="background:#feca57;color:#222f3e;padding:0.5em 1em;border-radius:6px;margin-bottom:0.5em;font-size:1em;">'
        'The area highlighted in <b style="color:#d63031">red/yellow</b> below visually identifies the most suspicious region the model focused on. This is the likely location of the pneumonia.'
        '</div>', unsafe_allow_html=True
    )
    # --------------------detection--------------------
    cam = generate_gradcam(model, img_tensor, pred_tensor)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    img_np = np.array(image.resize((224, 224)))
    overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
    st.image(overlay, caption='Grad-CAM Heatmap', use_column_width=True)
    st.caption('Red/yellow regions indicate areas important for the model prediction.')

else:
    st.info('Please upload a chest X-ray image.')

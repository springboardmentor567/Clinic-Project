"""
Chest X-Ray Pneumonia Detection - Streamlit Application
Created by: Ekta Joge

A comprehensive AI-powered medical imaging application for detecting pneumonia
from chest X-ray images using ResNet18 deep learning model with Grad-CAM visualization.
"""

import streamlit as st

st.set_page_config(page_title="Chest X-Ray Detection", layout="wide", page_icon="🩺")

import os
import numpy as np
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2

# Path to trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pth')


@st.cache_resource
def load_model():
    """Load the ResNet18 model and its trained weights."""
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    state = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_resource
def get_transform(img_size):
    """Return preprocessing transform for inference."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def preprocess_image(image, img_size):
    transform = get_transform(img_size)
    return transform(image).unsqueeze(0)


def generate_gradcam(model, img_tensor, target_index):
    """Compute a Grad-CAM heatmap."""
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    final_conv = model.layer4[1].conv2
    handle_fw = final_conv.register_forward_hook(forward_hook)
    handle_bw = final_conv.register_backward_hook(backward_hook)

    img_tensor = img_tensor.requires_grad_()
    outputs = model(img_tensor)
    model.zero_grad()
    score = outputs[0, target_index]
    score.backward(retain_graph=True)

    grads = gradients[0][0].detach().cpu().numpy()
    acts = activations[0][0].detach().cpu().numpy()
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = np.maximum(cam, 0)
    cam = cam / cam.max() if cam.max() != 0 else cam

    handle_fw.remove()
    handle_bw.remove()
    return cam


def overlay_cam_on_image(image_rgb_np, cam):
    """Overlay CAM heatmap on image."""
    h, w = image_rgb_np.shape[:2]
    cam_resized = cv2.resize((cam * 255).astype(np.uint8), (w, h))
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_rgb_np, 0.5, heatmap, 0.5, 0)
    return overlay


def predict_single(model, image: Image.Image, img_size: int):
    tensor = preprocess_image(image.convert('RGB'), img_size)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
    return pred_idx, probs


def main():
    """Main app UI and logic."""

    # Simple CSS
    st.markdown("""
    <style>
    .header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .result-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .normal {
        border-left-color: #28a745;
        background: #d4edda;
    }
    .pneumonia {
        border-left-color: #dc3545;
        background: #f8d7da;
    }
    </style>
    """, unsafe_allow_html=True)

    # 1. MAIN HEADING
    st.markdown("""
    <div class="header">
        <h1>🩺 Chest X-Ray Pneumonia Detection</h1>
        <p>AI-powered medical imaging analysis by Ekta Joge</p>
    </div>
    """, unsafe_allow_html=True)

    # 2. UPLOAD SECTION
    st.markdown("### Upload Images")
    
    # Simple settings
    show_cam = st.checkbox("Show Grad-CAM", value=True)
    
    # Load model
    model = load_model()
    st.success("✅ Model loaded successfully")
    
    # Source selection
    source = st.radio("Input Source", ["Upload Images", "Sample Images"], horizontal=True)
    
    files = []
    sample_items = []
    
    if source == "Upload Images":
        uploaded_files = st.file_uploader(
            "Choose chest X-ray images", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True
        )
        if uploaded_files:
            files = uploaded_files
            st.success(f"✅ {len(files)} image(s) uploaded")
        else:
            st.info("👆 Please upload image files to begin analysis")
            return
    else:
        # Sample images
        normal_dir = os.path.join('data', 'Images', 'test', 'NORMAL')
        pneu_dir = os.path.join('data', 'Images', 'test', 'PNEUMONIA')
        
        if os.path.isdir(normal_dir) and os.path.isdir(pneu_dir):
            cat = st.selectbox("Category", ["NORMAL", "PNEUMONIA"])
            folder = normal_dir if cat == 'NORMAL' else pneu_dir
            
            try:
                all_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if all_files:
                    selected_file = st.selectbox("Select image", all_files[:20])
                    sample_items = [{"name": selected_file, "path": os.path.join(folder, selected_file)}]
                    st.success("✅ Sample image selected")
                else:
                    st.warning("No images found in selected category")
                    return
            except Exception:
                st.warning("Could not access sample images")
                return
        else:
            st.warning("Sample images not found. Please use Upload mode.")
            return

    # 3. OUTPUT SECTION - 2x2 Grid Layout
    st.markdown("### Analysis Results")
    
    results = []
    iterable = [{"name": getattr(f, 'name', f"image_{i+1}"), "file": f} for i, f in enumerate(files)] if source == "Upload Images" else sample_items

    # Process all images first
    processed_images = []
    for idx, item in enumerate(iterable):
        try:
            img = Image.open(item['file']).convert('RGB') if 'file' in item else Image.open(item['path']).convert('RGB')
        except Exception as e:
            st.error(f"❌ Could not read image {item.get('name','(unknown)')}: {e}")
            continue

        # Prediction
        pred_idx, probs = predict_single(model, img, 224)  # Fixed image size
        pneumonia_prob = float(probs[1])
        normal_prob = float(probs[0])
        predicted_label = "Pneumonia" if pneumonia_prob >= normal_prob else "Normal"

        processed_images.append({
            'img': img,
            'item': item,
            'idx': idx,
            'pneumonia_prob': pneumonia_prob,
            'normal_prob': normal_prob,
            'predicted_label': predicted_label
        })

        results.append({
            'filename': item.get('name', f"image_{idx+1}.png"),
            'predicted_label': predicted_label,
            'prob_pneumonia': pneumonia_prob,
            'prob_normal': normal_prob,
        })

    # Display each image in a 2x2 grid layout
    for idx, data in enumerate(processed_images):
        st.markdown(f"#### 📸 {data['item'].get('name', f'Image {data['idx']+1}')}")
        
        # Create 2x2 grid: Image | Grad-CAM, Prediction | Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Position 11: Original Image
            st.markdown("**Original Image**")
            st.image(data['img'], use_container_width=True)
            
            # Position 21: Prediction Results
            st.markdown("**Prediction Results**")
            result_class = "pneumonia" if data['predicted_label'] == "Pneumonia" else "normal"
            st.markdown(f'<div class="result-box {result_class}">', unsafe_allow_html=True)
            st.markdown(f"#### 🎯 {data['predicted_label']}")
            st.markdown(f"**Pneumonia:** {data['pneumonia_prob']*100:.1f}%")
            st.markdown(f"**Normal:** {data['normal_prob']*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Position 12: Grad-CAM Overlay
            if show_cam:
                st.markdown("**Grad-CAM Overlay**")
                try:
                    tensor = preprocess_image(data['img'], 224)
                    cam = generate_gradcam(model, tensor, 1 if data['predicted_label'] == 'Pneumonia' else 0)
                    overlay = overlay_cam_on_image(np.array(data['img']), cam)
                    st.image(overlay, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate Grad-CAM: {e}")
            else:
                st.info("Enable Grad-CAM to see heatmap overlay")
            
            # Position 22: Probability Visualization
            st.markdown("**Probability Visualization**")
            import pandas as pd
            prob_data = pd.DataFrame({
                "Class": ["Normal", "Pneumonia"],
                "Probability": [data['normal_prob'], data['pneumonia_prob']]
            })
            st.bar_chart(prob_data.set_index("Class"), height=300)
        
        st.markdown("---")  # Separator between images

    # 4. DATA SECTION
    if results:
        st.markdown("### Summary & Data")
        
        # Summary statistics
        total_images = len(results)
        pneumonia_count = sum(1 for r in results if r['predicted_label'] == 'Pneumonia')
        normal_count = total_images - pneumonia_count
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Images", total_images)
        with col2:
            st.metric("Pneumonia Detected", pneumonia_count)
        with col3:
            st.metric("Normal Cases", normal_count)
        
        # Results table
        import pandas as pd
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        
        # Download
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button(
            label="📥 Download CSV Report",
            data=csv_bytes,
            file_name=f"pneumonia_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    # 6. DESCRIPTION AT THE END
    st.markdown("---")
    st.markdown("### About This Application")
    st.markdown("""
    **Streamlit application for Chest X-Ray Pneumonia detection.**
    
    **Created by:** Ekta Joge
    
    **Features:**
    - AI-powered chest X-ray analysis using ResNet18 deep learning model
    - Single and batch image uploads with sample image support
    - Real-time prediction with confidence metrics
    - Grad-CAM visualization for model interpretability
    - Results table and downloadable CSV export
    - Professional 2x2 grid layout for optimal user experience
    
    **⚠️ Medical Disclaimer:** This tool is for research and educational purposes only. 
    Not a substitute for professional medical diagnosis.
    """)


if __name__ == "__main__":
    main()
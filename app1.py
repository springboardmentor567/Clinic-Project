import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import altair as alt
import numpy as np
import cv2 # Used for image processing and blending (make sure to install: pip install opencv-python)
import io

# --- Configuration ---
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
MODEL_PATH = 'model.pth' # Path to your trained PyTorch model weights

# --- Utility Functions ---

@st.cache_resource
def load_model(model_path):
    """Loads the trained PyTorch model and sets it to evaluation mode."""
    try:
        # Initialize ResNet18 structure
        model = models.resnet18(weights=None)
        # Adjust the final fully connected layer for binary classification
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        
        # Load the state dictionary (weights)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at '{model_path}'. Please ensure 'model.pth' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_transforms():
    """Defines the necessary image transformations for model input."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def predict(model, image: Image.Image, transform):
    """Performs classification prediction on the input image."""
    try:
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        predicted_index = torch.argmax(probabilities).item()
        
        prediction_class = CLASS_NAMES[predicted_index]
        confidence = probabilities[predicted_index].item()
        
        return prediction_class, confidence, probabilities

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return "ERROR", 0.0, torch.tensor([0.0, 0.0])

def apply_grad_cam(model, image: Image.Image, transform, target_class_idx):
    """Generates a Grad-CAM heatmap for the predicted class on the image.
    
    NOTE: This implementation is specific to ResNet18 structure targeting layer4[-1].
    """
    
    # 1. Prepare image and model
    input_tensor = transform(image).unsqueeze(0)
    
    # Target the last convolutional block of ResNet18
    target_layer = model.layer4[-1] 
    
    # Variables to store feature maps and gradients
    feature_map = None
    gradient = None

    def save_feature_map(module, input, output):
        nonlocal feature_map
        feature_map = output

    def save_gradient(module, grad_input, grad_output):
        nonlocal gradient
        gradient = grad_output[0]

    # Set up hooks to capture intermediate data
    hook_handle_fm = target_layer.register_forward_hook(save_feature_map)
    hook_handle_grad = target_layer.register_full_backward_hook(save_gradient)

    # 2. Forward pass
    model.zero_grad()
    output = model(input_tensor)
    
    # 3. Backward pass
    # Create a one-hot vector for the target class prediction score
    one_hot = torch.zeros_like(output)
    one_hot[0][target_class_idx] = 1
    
    # Backpropagate the gradient from the target class score
    output.backward(gradient=one_hot, retain_graph=True)

    # 4. Calculate Grad-CAM
    # Global average pooling on the gradients (alpha weights)
    weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
    
    # Weighted combination of feature maps
    cam = weights * feature_map
    cam = torch.sum(cam, dim=1).squeeze()
    
    # Apply ReLU to the CAM (only positive contributions matter)
    cam = torch.relu(cam)
    
    # 5. Normalize and resize heatmap
    heatmap = cam.cpu().data.numpy()
    # Normalize to 0-1
    heatmap = heatmap / np.max(heatmap)
    
    # Resize heatmap to the original image dimensions
    img_width, img_height = image.size
    heatmap = cv2.resize(heatmap, (img_width, img_height))
    
    # Convert to 8-bit unsigned integer and apply a color map (JET is common)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Clean up the hooks
    hook_handle_fm.remove()
    hook_handle_grad.remove()
    
    return heatmap

# --- Streamlit Main Function ---

def main():
    st.set_page_config(page_title="Pneumonia Classifier", layout="wide")
    st.title("Chest X-Ray Pneumonia Classification & Localization")
    st.markdown("Upload a chest X-ray image to classify it and visualize the region the model focused on.")

    model = load_model(MODEL_PATH)
    data_transform = get_transforms()

    if model is None:
        return

    uploaded_file = st.file_uploader("Choose a Chest X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])

        # --- Column 1: Image Display and Prediction ---
        with col1:
            try:
                # Load the image using PIL
                image = Image.open(uploaded_file).convert("RGB")
                original_image = image.copy() # Keep a copy for blending later
                # Updated: replaced use_column_width=True with use_container_width=True
                st.image(image, caption='Uploaded X-Ray Image', use_container_width=True)
            except Exception as e:
                st.error(f"Could not load image: {e}")
                return

        # --- Column 2: Results and Chart ---
        with col2:
            st.header("Classification Result:")
            
            # Run prediction
            prediction_class, confidence, probabilities = predict(model, image, data_transform)

            # Display prediction
            if prediction_class == "PNEUMONIA":
                st.markdown(f"**<span style='color:red; font-size: 30px;'>PNEUMONIA DETECTED</span>**", unsafe_allow_html=True)
                st.markdown("⚠️ **Action Recommended:** Please consult a medical professional immediately with this result.")
            elif prediction_class == "NORMAL":
                st.markdown(f"**<span style='color:green; font-size: 30px;'>NORMAL</span>**", unsafe_allow_html=True)
                st.markdown("✅ The image appears normal based on the model's analysis.")
            else:
                st.markdown(f"**<span style='color:orange; font-size: 30px;'>{prediction_class}</span>**", unsafe_allow_html=True)

            st.markdown(f"Confidence: **{confidence:.2%}**")
            
            # Comparison Chart
            prob_data = pd.DataFrame({
                'Class': CLASS_NAMES,
                'Probability': [p.item() for p in probabilities]
            })

            highlight_color = '#FF4B4B' if prediction_class == 'PNEUMONIA' else '#008000'

            chart = alt.Chart(prob_data).mark_bar().encode(
                x=alt.X('Probability', axis=alt.Axis(format='%', title='Confidence')),
                y=alt.Y('Class', title='Classification', sort='-x'),
                color=alt.condition(
                    alt.datum.Class == prediction_class,
                    alt.value(highlight_color),
                    alt.value('lightgray')
                ),
                tooltip=['Class', alt.Tooltip('Probability', format='.2%')]
            ).properties(
                title='Model Confidence Comparison'
            )
            st.altair_chart(chart, use_container_width=True)
            
            st.info("Disclaimer: This tool is an AI aid and is NOT a substitute for professional medical diagnosis.")

        # --- Grad-CAM Heatmap Visualization ---
        if prediction_class == "PNEUMONIA":
            st.markdown("---")
            st.header("Localization Map (Area of Interest)")

            # The target index for Grad-CAM is the index of 'PNEUMONIA' (which is 1)
            pneumonia_idx = CLASS_NAMES.index('PNEUMONIA')
            
            with st.spinner('Generating localization map...'):
                # 1. Generate the heatmap
                heatmap_cv = apply_grad_cam(model, original_image, data_transform, pneumonia_idx)
                
                # 2. Convert PIL image to numpy array for blending with OpenCV
                img_np = np.array(original_image.convert("RGB"))
                
                # 3. Blend the heatmap with the original image (0.7 weight to image, 0.3 to heatmap)
                # Note: OpenCV uses BGR by default, so we use cv2.cvtColor later.
                superimposed_img = cv2.addWeighted(img_np, 0.7, heatmap_cv, 0.3, 0)
                
                # 4. Convert back to RGB for Streamlit display
                superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
            
            # Enhanced text to specify the place
            st.warning("The area highlighted in **red/yellow** below visually identifies the most suspicious region the model focused on. This is the **likely location of the pneumonia**.")

            # Use columns to reduce the size and center the Grad-CAM image
            col_left, col_center, col_right = st.columns([1, 2, 1])

            with col_center:
                # Use use_container_width=True, but constrained by the smaller column width
                st.image(superimposed_img, caption="Grad-CAM Heatmap Overlay (Likely Pneumonia)", use_container_width=True)


if __name__ == '__main__':
    main()

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import onnxruntime as ort
import numpy as np
import altair as alt 
import os 
import onnx 
import cv2 
import io 
import gdown 
import base64 

device = torch.device("cpu") 


MODEL_PTH_PATH = "model_stable.pth"

GOOGLE_DRIVE_FILE_ID = "1FN8UG5pJiKPT8_yE8CkvlTC2DthCxbVR"


DEFAULT_IMAGE_PATH = "image_cfd0a2.jpg" 


BASE64_PLACEHOLDER_IMAGE = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwAB/2+Bw/oAAAAASUVORK5CYII="


transform_gradcam = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class_names = ["NORMAL", "PNEUMONIA"]


def generate_grad_cam(model, target_layer, img_tensor, original_img):
    model.eval()
    target_class_idx = 1 # Assuming PNEUMONIA is class 1
    
    feature_maps = []
    def forward_hook(module, input, output):
        feature_maps.append(output)
    
    gradients = []
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())


    hook_handle_fm = target_layer.register_forward_hook(forward_hook)
    hook_handle_grad = target_layer.register_backward_hook(backward_hook)

    img_input = img_tensor.clone().detach().requires_grad_(True)
    
    output = model(img_input)
    model.zero_grad()
    
    target_score = output[0, target_class_idx]
    
  
    if target_score.numel() == 1:
        target_score.backward(retain_graph=True) 
    
    hook_handle_fm.remove()
    hook_handle_grad.remove()
    
    if not gradients or len(gradients[0]) == 0:
     
        return Image.new('RGB', (original_img.width, original_img.height), color = 'gray')

    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    feature_map = feature_maps[0].squeeze(0)
    
    for i in range(feature_map.shape[0]):
        feature_map[i, :, :] *= pooled_gradients[i]
    
    heatmap = torch.mean(feature_map, dim=0).relu()
    
   
    max_val = torch.max(heatmap)
    if max_val > 0:
        heatmap /= max_val
    
    heatmap_np = heatmap.detach().cpu().numpy() 
    
 
    heatmap_resized = cv2.resize(heatmap_np, (original_img.width, original_img.height))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    

    img_np = np.array(original_img.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) 
    

    superimposed_img = cv2.addWeighted(img_bgr, 0.5, heatmap_colored, 0.5, 0)
    
    superimposed_img_pil = Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    
    return superimposed_img_pil


def get_status_color(class_name):
    
    return "red" if class_name == "PNEUMONIA" else "green"

def create_conditional_bar_chart(df, model_name):
    
    
    color_scale = alt.condition(
        alt.datum.Class == "PNEUMONIA",
        alt.value("red"),
        alt.value("green") 
    )

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Class", sort="-y", title=None, axis=alt.Axis(labels=True, title=None)), 
        y=alt.Y("Probability", axis=alt.Axis(format=".0%", title="Probability")),
        color=color_scale,
        tooltip=["Class", alt.Tooltip("Probability", format=".4%")] 
    ).properties(
        title=f"{model_name} Class Probabilities" 
    ).interactive()
    
    return chart


@st.cache_resource
def load_pytorch_model():
    
    
    if not os.path.exists(MODEL_PTH_PATH) or os.path.getsize(MODEL_PTH_PATH) < 100000:
        with st.spinner("Model weights not found/corrupted. Downloading from Google Drive..."):
            try:
               
                gdown.download(id=GOOGLE_DRIVE_FILE_ID, output=MODEL_PTH_PATH, quiet=True)
                
               
                if not os.path.exists(MODEL_PTH_PATH) or os.path.getsize(MODEL_PTH_PATH) < 100000:
                     raise Exception("Downloaded file is empty or corrupted.")

            except Exception as e:
                st.error(f"Failed to download model from Google Drive. Check the file ID and permissions. Error: {e}")
                return None 
   

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    try:
        with st.spinner(f"Loading PyTorch weights from {MODEL_PTH_PATH}..."):
            model.load_state_dict(torch.load(MODEL_PTH_PATH, map_location=device))
    except Exception as e:
        st.error(f"Failed to load model weights. Ensure '{MODEL_PTH_PATH}' is a valid PyTorch state dict. Error: {e}")
        return None
        
    model.to(device)
    model.eval()
    return model

pytorch_model = load_pytorch_model()


if pytorch_model is None:
    st.error("Cannot proceed without PyTorch model. Please check logs for download errors.")
    st.stop()


@st.cache_resource
def load_onnx_model(_model_pt, device):
   
    ONNX_PATH = "model_onnx_new.onnx"
    
  
    if not os.path.exists(ONNX_PATH) or os.path.getsize(ONNX_PATH) < 100000:
        
        dummy_input = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float32)

        try:
            with st.spinner("EXPORTING PYTORCH TO ONNX (This is CPU-intensive and might take time)..."):
                # Export the PyTorch model to ONNX format
                torch.onnx.export(
                    _model_pt, 
                    dummy_input,
                    ONNX_PATH,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'},
                                  'output': {0: 'batch_size'}}
                )
           
            onnx_model = onnx.load(ONNX_PATH)
            onnx.checker.check_model(onnx_model)
        except Exception as e:
            st.warning(f"Failed to export ONNX model. Running in PyTorch-only mode. Error: {e}")
            return None 

   
    try:
        with st.spinner(f"Loading ONNX session from {ONNX_PATH}..."):
            return ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    except Exception as e:
        st.warning(f"Failed to load ONNX session. Running in PyTorch-only mode. Error: {e}")
        return None


onnx_session = load_onnx_model(pytorch_model, device)



st.markdown(
    """
    <style>
    /* Main Streamlit container */
    .stApp {
        background-color: #000000;
        color: white; 
    }
    /* Set the main content background to black */
    [data-testid="stVerticalBlock"] {
        background-color: #000000;
    }
    /* Ensure markdown text is visible on dark background */
    h1, h2, h3, h4, p, label {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.header("CHEST X-RAY PNEUMONIA CLASSIFIER") 
st.markdown("---")



uploaded_file = st.file_uploader("Choose a Chest X-ray image", type=["jpg","jpeg","png"])



if uploaded_file is None:
    
    image_to_display = None
    caption_text = "Example X-ray Image (Sample)"
    
    try:
        
        if os.path.exists(DEFAULT_IMAGE_PATH):
            image_to_display = Image.open(DEFAULT_IMAGE_PATH).convert("RGB")
        
     
        if image_to_display is None:
            image_data = base64.b64decode(BASE64_PLACEHOLDER_IMAGE)
            image_to_display = Image.open(io.BytesIO(image_data)).convert("RGB")
            caption_text = "Placeholder X-ray (Upload an image)"
            
    except Exception:
       
        pass
        
    if image_to_display:
        st.image(image_to_display, caption=caption_text, use_container_width=True)
        
    st.stop()


if uploaded_file:
    image_bytes = uploaded_file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    current_model = pytorch_model 
    target_layer = current_model.layer4[-1].conv2 
    
    img_tensor = transform(img).unsqueeze(0).to(device).to(torch.float32) 
    gradcam_tensor = transform_gradcam(img).unsqueeze(0).to(device).to(torch.float32)
    
    
  
    with torch.no_grad():
        outputs_pt = current_model(img_tensor)
    probs_pt = torch.softmax(outputs_pt, dim=1)[0]
    pred_idx_pt = torch.argmax(probs_pt).item()
    pred_class_pt = class_names[pred_idx_pt]

    
   
    probs_onnx = None
    if onnx_session:
        x = transform(img).unsqueeze(0).numpy().astype(np.float32)
       
        outputs_onnx_list = onnx_session.run(None, {"input": x})
        
        if outputs_onnx_list and len(outputs_onnx_list) > 0:
            outputs_onnx = outputs_onnx_list[0][0]
            probs_onnx = np.exp(outputs_onnx) / np.sum(np.exp(outputs_onnx))
        else:
            st.warning("ONNX session ran but returned no valid output.")
    
    
    final_diagnosis_class = pred_class_pt
    max_prob = probs_pt[pred_idx_pt].item()

    final_diagnosis_color = get_status_color(final_diagnosis_class)

    
    st.markdown("### Classification Result:")
    
 
    if final_diagnosis_class == "PNEUMONIA":
        st.markdown(f"## :{final_diagnosis_color}[PNEUMONIA DETECTED]")
        st.warning("Action Recommended: Please consult a medical professional immediately with this result.")
    else:
        st.markdown(f"## :{final_diagnosis_color}[NORMAL FINDING]")
    
        st.info("**Model Analysis:** No Sign of Pneumonia Detected.") 

    st.markdown(f"Confidence: **{max_prob*100:.2f}%**")
    st.markdown("---")

    st.markdown("### Uploaded X-ray")
    st.image(img, use_container_width=True)
    st.markdown("---")

    
    st.markdown("### Model Focus (Grad-CAM Visualization)")
  
    if pred_class_pt == "PNEUMONIA":
        with st.spinner("Generating Explainability Heatmap..."):
          
            gradcam_tensor.requires_grad_(True)
            heatmap_img = generate_grad_cam(current_model, target_layer, gradcam_tensor, img)
            st.image(heatmap_img, caption="Areas contributing to PNEUMONIA diagnosis (Red/Yellow)", use_container_width=True)
    else:
        st.info("Grad-CAM visualization skipped as the PyTorch diagnosis is NORMAL.")

    st.markdown("---")

    
 
    st.markdown("### Model Prediction Scores")
    
 
    if onnx_session:
        col_pt, col_onnx = st.columns(2)
    else:
        col_pt, _ = st.columns([1, 0])
        st.warning("ONNX predictions are currently unavailable because the export/load process failed due to resource limits or an unknown issue. Running in PyTorch-only mode.")


    with col_pt:
        st.markdown("#### PyTorch Prediction")
        pred_prob_pt = probs_pt[pred_idx_pt].item()
        color_pt = get_status_color(pred_class_pt)

        st.markdown(
            f"Highest Confidence: **:{color_pt}[{pred_class_pt}]** ({pred_prob_pt*100:.4f}%)"
        )
        df_pt = pd.DataFrame({"Class": class_names, "Probability":[p.item() for p in probs_pt]})
    
        st.altair_chart(create_conditional_bar_chart(df_pt, "PyTorch"), use_container_width=True) 

    if onnx_session and probs_onnx is not None:
        with col_onnx:
            st.markdown("#### ONNX Prediction")
            
            onnx_pred_idx = np.argmax(probs_onnx)
            onnx_pred_class = class_names[onnx_pred_idx]
            onnx_pred_prob = probs_onnx[onnx_pred_idx]
            color_onnx = get_status_color(onnx_pred_class)

            st.markdown(
                f"Highest Confidence: **:{color_onnx}[{onnx_pred_class}]** ({onnx_pred_prob*100:.4f}%)"
            )
            df_onnx = pd.DataFrame({"Class": class_names, "Probability": probs_onnx})
    
            st.altair_chart(create_conditional_bar_chart(df_onnx, "ONNX"), use_container_width=True)

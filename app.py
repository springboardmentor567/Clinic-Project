# app.py

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pydicom
import matplotlib.pyplot as plt
import os

# ----------------------------
# 1️⃣ Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="VinDr-CXR Detector",
    page_icon="🩺",
    layout="wide"
)

# ----------------------------
# 2️⃣ Dark Mode CSS
# ----------------------------
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #2ecc71;
        color: #000000;
    }
    .stFileUploader>div>div>div {
        background-color: #333333;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🩺 VinDr-CXR Chest X-ray Abnormality Detector")
st.write("Upload a chest X-ray image and detect abnormalities with YOLOv8.")

# ----------------------------
# 3️⃣ Load YOLOv8 Model
# ----------------------------
MODEL_PATH = r"C:\Users\kesha\OneDrive\Desktop\info\lung-abnormality-detection-using-yolov8n\runs\detect\VinDr_CXR_model2\weights\best.pt"
model = YOLO(MODEL_PATH)

# ----------------------------
# 4️⃣ Image Upload
# ----------------------------
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["png", "jpg", "jpeg", "dcm"])

if uploaded_file is not None:
    # Handle DICOM
    if uploaded_file.name.endswith(".dcm"):
        dicom_data = pydicom.dcmread(uploaded_file)
        image = dicom_data.pixel_array
        if np.max(image) > 0:
            image = cv2.convertScaleAbs(image, alpha=(255.0 / np.max(image)))
        else:
            image = np.zeros_like(image, dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = Image.open(uploaded_file).convert("RGB")
        image = np.array(image)

    img_array = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # ----------------------------
    # 5️⃣ YOLO Prediction
    # ----------------------------
    results = model(img_array)
    result = results[0]

    annotated_img = img_array.copy()
    affected_area = 0
    total_area = img_array.shape[0] * img_array.shape[1]
    class_conf_map = {}

    if len(result.boxes) == 0:
        st.success("✅ No abnormalities detected — X-ray appears normal")
        affected_pct = 0
        healthy_pct = 100
    else:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0]) * 100
            label = model.names[cls]

            # Max confidence per class
            if label not in class_conf_map or class_conf_map[label] < conf:
                class_conf_map[label] = conf
            

            # Approx affected area
            affected_area += (x2 - x1) * (y2 - y1)

            # Red bounding box and label
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box
            cv2.putText(annotated_img, label, (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)  # Red text

        affected_pct = (affected_area / total_area) * 100
        healthy_pct = 100 - affected_pct


    # ----------------------------
    # 6️⃣ Display Results Side by Side
    # ----------------------------
    col1, col2 = st.columns(2)

    # Left: Original + affected vs healthy
    with col1:
        st.image(image, caption="📷 Original X-ray", use_container_width=True)

        fig1, ax1 = plt.subplots(figsize=(4, 2), facecolor="#000000")
        ax1.set_facecolor("#000000")
        categories = ["Affected Area", "Healthy Area"]
        values = [affected_pct, healthy_pct]
        colors = ["#e74c3c", "#2ecc71"]

        ax1.barh(categories, values, color=colors, height=0.4)
        ax1.set_xlim(0, 100)
        ax1.set_xticks([0, 25, 50, 75, 100])
        ax1.set_xlabel("Percentage (%)", fontsize=9, color="white")
        ax1.tick_params(axis='x', colors='white')
        ax1.tick_params(axis='y', colors='white')
        for spine in ax1.spines.values():
            spine.set_color("white")
        for i, v in enumerate(values):
            ax1.text(v + 1, i, f"{v:.1f}%", va='center', fontsize=8, color='white')
        st.pyplot(fig1)

    # Right: Annotated + confidence
    with col2:
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                 caption="✅ Detected Abnormalities", use_container_width=True)

        if class_conf_map:
            fig2, ax2 = plt.subplots(figsize=(4, 3), facecolor="#000000")
            ax2.set_facecolor("#000000")
            classes = list(class_conf_map.keys())
            conf_scores = list(class_conf_map.values())

            x = np.arange(len(classes))
            bars = ax2.bar(x, conf_scores, color="#e74c3c", width=0.5)
            ax2.set_ylabel("Confidence (%)", fontsize=9, color="white")
            ax2.set_xticks(x)
            ax2.set_xticklabels(classes, fontsize=9, rotation=20, color="white")
            ax2.set_ylim(0, 110)
            ax2.tick_params(axis='y', colors='white')

            for spine in ax2.spines.values():
                spine.set_color("white")

            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height + 1,
                         f"{height:.1f}%", ha='center', va='bottom', fontsize=8, color='white')

            st.pyplot(fig2)

            st.markdown("### 🦠 Affected by:")
            st.write(", ".join(classes))

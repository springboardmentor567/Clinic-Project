# 🩺 LungScan AI: Automated Chest X-Ray Abnormality Detection

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://infosys-project-awctidx2cpngvabdrxdq8k.streamlit.app/)

**LungScan AI** is a deep-learning–powered web app that detects and localizes abnormalities in chest X-rays.  
Built with **Python**, **PyTorch**, and **Streamlit**, it demonstrates an end-to-end medical imaging pipeline — from data preprocessing to visualization.

---

## 🎬 Live Demonstration

The app is deployed and publicly accessible.  
Click the badge above ☝️ or [open it directly here](https://infosys-project-awctidx2cpngvabdrxdq8k.streamlit.app/).

> 🎥 *A short walkthrough video and sample screenshots will be added here once uploaded.*

| Application Home | Prediction Output |
|------------------|------------------|
| ![Home Screenshot](assets/home_screenshot.png) | ![Prediction Screenshot](assets/prediction_screenshot.png) |

---

## 🧠 About the Project

**LungScan AI** is a portfolio-level proof of concept designed to demonstrate how AI can assist in chest-X-ray screening by identifying potential lung abnormalities automatically.

**Goal:** To provide an intuitive interface for clinicians, researchers, and students to visualize how machine learning can be used for diagnostic image analysis.

> ⚠️ *This project is for educational and demonstration purposes only and is **not intended for medical use**.*

---

## ✨ Key Features

- **Dual-Model Pipeline:** Combines classification (ResNet-based) and detection (YOLOv8) stages.  
- **Interactive Streamlit UI:** Upload an image and view both class prediction and bounding-box localization.  
- **Grad-CAM Visualization:** Highlights the key image regions influencing the model’s decision.  
- **Performance Dashboard:** Displays metrics, confusion matrices, and detected anomalies.  
- **Downloadable Reports:** Allows exporting model results and interpretations.

---

## 🧩 Methodology: AI Pipeline

The analysis follows a **three-stage sequential pipeline:**

1. **Classification:**  
   - Uses a **ResNet-50** CNN to classify each X-ray as “Normal” or “Abnormal”.  
2. **Detection:**  
   - If “Abnormal”, runs a **YOLOv8** detector to localize regions of concern (e.g., nodules, opacity).  
3. **Visualization:**  
   - Generates **Grad-CAM heatmaps** for interpretability.

---

## ⚙️ Technology Stack

| Layer | Tools |
|-------|-------|
| **Backend / Web** | Python, Streamlit |
| **AI / ML Frameworks** | PyTorch, Ultralytics YOLOv8 |
| **Classification Model** | ResNet-50 |
| **Data Processing** | Pandas, NumPy, OpenCV, Pillow |
| **Visualization** | Matplotlib, Grad-CAM |
| **Deployment** | Streamlit Community Cloud |
| **Version Control** | Git + GitHub |

---

## 🗂️ Project Structure


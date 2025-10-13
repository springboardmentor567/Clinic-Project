# 🩺 LungScan AI: Automated Chest X-Ray Abnormality Detection

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://infosys-project-awctidx2cpngvabdrxdq8k.streamlit.app/)

LungScan AI is an end-to-end deep learning application for detecting and localizing abnormalities in chest X-rays using advanced CNN and object detection models.  
Built with **Python**, **PyTorch**, and **Streamlit**, it provides an intuitive interface for exploring AI-driven medical image analysis.

---

## 🎥 Live Demonstration

**CliniScan Demo Video:** A complete walkthrough of the application’s features, workflow, and results.  
📺 *Watch the video below once uploaded.*

🔗 [**Open Live Application Here →**](https://infosys-project-awctidx2cpngvabdrxdq8k.streamlit.app/)

| Application Home | Detection Output |
|------------------|------------------|
| ![Home Screenshot](assets/home_screenshot.png) | ![Detection Screenshot](assets/prediction_screenshot.png) |

---

## 🧠 About the Project

LungScan AI demonstrates the integration of deep learning in medical image processing through classification, detection, and interpretability techniques.  
The goal is to showcase a streamlined AI workflow — from data preprocessing to final model deployment — in an interactive and visually interpretable form.

---

## ✨ Key Features

- **Dual-Model Workflow:** Classification (ResNet-50) + Detection (YOLOv8).  
- **Interactive Dashboard:** Upload, analyze, and visualize results in real-time.  
- **Grad-CAM Heatmaps:** Highlight important regions influencing model predictions.  
- **Performance Metrics:** Evaluate model accuracy and detection quality.  
- **Streamlit Deployment:** Lightweight and fully hosted web interface.

---

## 🧩 Methodology: AI Pipeline

The project follows a **three-stage process**:

1. **Classification:**  
   A **ResNet-50** model identifies whether the X-ray is *Normal* or *Abnormal*.  

2. **Detection:**  
   If *Abnormal*, a **YOLOv8** detector localizes suspicious regions (bounding boxes).  

3. **Visualization:**  
   **Grad-CAM** overlays visualize where the model focused during prediction.

---

## ⚙️ Technology Stack

| Layer | Tools |
|-------|-------|
| **Frameworks** | Python, Streamlit |
| **Deep Learning** | PyTorch, Ultralytics YOLOv8 |
| **Classification Model** | ResNet-50 |
| **Data Handling** | Pandas, NumPy, OpenCV, Pillow |
| **Visualization** | Matplotlib, Grad-CAM |
| **Deployment** | Streamlit Community Cloud |
| **Version Control** | Git + GitHub |

---

## 🗂️ Project Structure


# 🩺 LungScan AI: Automated Chest X-Ray Abnormality Detection

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://infosys-project-awctidx2cpngvabdrxdq8k.streamlit.app/)

LungScan AI is an end-to-end deep learning application for detecting and localizing abnormalities in chest X-rays using advanced CNN and object detection models.  
Built with **Python**, **PyTorch**, and **Streamlit**, it provides an intuitive interface for exploring AI-driven medical image analysis.

---

## 🎥 Live Demonstration

**CliniScan Demo Video:** A complete walkthrough of the application’s features, workflow, and results.  
📺 *Watch the video below once uploaded.*

🔗 [**Open Live Application Here →**](https://infosys-project-awctidx2cpngvabdrxdq8k.streamlit.app/)

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

lung_detection/
│
├── streamlit_app.py # Main Streamlit UI script
├── /models/ # Trained YOLOv8 / ResNet weights
├── /src/ # Helper modules and utility functions
├── /data/ # Processed dataset or samples
├── /assets/ # Images, Grad-CAMs, screenshots
├── /scripts/ # Model training and annotation conversion
├── requirements.txt # Dependencies
└── .streamlit/config.toml # App theme configuration

---

## ⚙️ Setup Instructions

### Clone the repository:
```bash
git clone https://github.com/Ektajoge55/lung_detection.git
cd lung_detection
Install dependencies (preferably in a virtual environment):
pip install -r requirements.txt

Launch the Streamlit app:
streamlit run streamlit_app.py


Then open the local URL displayed in your terminal or
View Live App Here →

🚀 Development Milestones
Milestone	Description
1️⃣ Data Preparation & Setup	Downloaded and converted VinDr-CXR DICOMs to PNG/JPEG; converted annotations to YOLO format.
2️⃣ Model Training & Evaluation	Trained baseline classification (ResNet-50) and detection (YOLOv8) models.
3️⃣ Optimization & Visualization	Applied transfer learning, augmentations, and Grad-CAM interpretability.
4️⃣ Deployment	Combined models in a unified Streamlit dashboard and deployed the web app.
📊 Results & Insights

Achieved strong baseline accuracy on VinDr-CXR subsets.

Effective localization of common abnormalities such as opacity, infiltration, and nodules.

Grad-CAM outputs align well with expected regions of interest.

Grad-CAM Example	YOLOv8 Detection

	
🌱 Future Enhancements

Multi-class disease classification

Integration with lung segmentation models

Model explainability dashboard

Batch prediction and automated reporting

👩‍💻 Developed By

Ekta Joge
🔗 Live App
 | GitHub Repository



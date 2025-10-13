# CliniScan AI: AI-Powered Lung Abnormality Detection

An end-to-end deep learning portfolio project demonstrating a complete pipeline for detecting and classifying abnormalities in chest X-ray images, built with Python, YOLOv8, TensorFlow, and deployed as an interactive web application with Streamlit.

## Live Demonstration

The application is deployed and publicly accessible. Click the link below to interact with the live app:

[Open in Streamlit](https://infosys-project-awctidx2cpngvabdrxdq8k.streamlit.app/)

<img width="1893" height="837" alt="Screenshot 2025-10-13 203502" src="https://github.com/user-attachments/assets/267be7f8-9a5c-4669-9299-6a481ba92cf6" />


https://github.com/user-attachments/assets/d2164893-43ab-427e-ae53-4d3dea62b537


## Table of Contents

- [About The Project](#about-the-project)
- [Key Features](#key-features)
- [Methodology: The AI Pipeline](#methodology-the-ai-pipeline)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Development Milestones](#development-milestones)

## About The Project

CliniScan AI is a proof-of-concept platform designed to showcase the power of deep learning in medical imaging. The primary goal is to provide an intuitive tool that assists in the analysis of chest radiographs by identifying and localizing potential pathological findings.

This project was developed to tackle the challenges of working with complex medical data and to build a full-stack AI application from data collection to final deployment.

This tool is for educational and demonstration purposes only and is not a certified medical device intended for real-world diagnostic use.

## Key Features

- **Dual-Model System:** Utilizes two distinct AI models for a comprehensive, two-stage analysis.
- **Interactive Web Interface:** A user-friendly application built with Streamlit that allows for easy image upload and analysis.
- **Explainable AI:** Incorporates Grad-CAM visualizations to provide insights into the classification model's decision-making process.
- **Flexible Detection:** Allows users to switch between a general single-class detector and a more specific multi-class detector to compare results.
- **Performance Dashboard:** A dedicated page to display and review the performance metrics (like Confusion Matrices and training data) for all integrated models.
- **Downloadable Reports:** Users can download a text-based summary of the analysis for any given X-ray.

## Methodology: The AI Pipeline

The application uses a sequential, two-stage process for analysis:

1. **Classification:** A ResNet50-based Convolutional Neural Network (CNN) first analyzes the entire image to determine if it is 'Normal' or 'Abnormal'. This acts as an initial, high-level screening.
2. **Detection:** If the image is deemed 'Abnormal', a YOLOv8 object detection model is then used to draw bounding boxes around the specific regions of interest that it identifies as potential abnormalities.
3. **Visualization:** For all 'Abnormal' classifications, a Grad-CAM (Gradient-weighted Class Activation Mapping) heatmap is generated. This visualization highlights the pixels the classification model focused on most, providing insight into its decision-making process.

## Technology Stack

- **Backend & Web Framework:** Python, Streamlit
- **Object Detection:** PyTorch, Ultralytics YOLOv8
- **Classification:** TensorFlow, Keras (ResNet50)
- **Data Handling & Image Processing:** Pandas, NumPy, OpenCV, Pillow
- **Deployment:** Streamlit Community Cloud
- **Version Control:** Git & GitHub (with Git LFS for large model handling)

## Project Structure

```
lung_detection/
├── streamlit_app.py        # Main Streamlit UI script
├── models/                 # Trained YOLOv8 / ResNet weights
├── src/                    # Helper modules and utility functions
├── data/                   # Processed dataset or samples
├── assets/                 # Images, Grad-CAMs, screenshots
├── scripts/                # Model training and annotation conversion
├── requirements.txt        # Dependencies
└── streamlit/config.toml   # App theme configuration
```

## Setup Instructions

Clone the repository:

```
git clone https://github.com/Ektajoge55/lung_detection.git
cd lung_detection
```

Install dependencies (preferably in a virtual environment):

```
pip install -r requirements.txt
```

Launch the Streamlit app:

```
streamlit run streamlit_app.py
```

## Development Milestones

This project was developed through a series of key milestones:

- **Milestone 1: Data Preprocessing & Setup:** Processed and cleaned distinct chest X-ray datasets, converting annotations to a unified format.
- **Milestone 2: Model Training & Experimentation:** Trained baseline classification (ResNet-50) and detection (YOLOv8) models. Applied transfer learning, augmentations, and Grad-CAM interpretability.
- **Milestone 3: Application Development:** Built the core Streamlit application, integrating the best-performing classifier and detectors.
- **Milestone 4: Final Polish & Deployment:** Refined the UI/UX, added professional features like Grad-CAM and downloadable reports, and deployed the final application to Streamlit Community Cloud.

(Placeholder for additional photos or screenshots - add your images here to illustrate milestones or app features)


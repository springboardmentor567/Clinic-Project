## CliniScan: Advanced Chest X-ray Analysis with Deep Learning 

This repository presents **CliniScan**, a collection of three distinct deep learning projects for analyzing **Chest X-ray (CXR)** images to detect and classify lung abnormalities. Each project utilizes a different state-of-the-art model and machine learning task to provide comprehensive diagnostic support, from pinpointing specific lesions to general abnormality screening.



### Project Gist

The projects leverage the **VinBigData Chest X-ray** dataset to demonstrate complete end-to-end pipelines: from data preprocessing to practical deployment via interactive **Streamlit** web applications.

### Model Summaries & Live Demos

| Project Name | ML Task | Model | Output | Explainability | Live App Link |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Abnormality Detection** | Object Detection | **YOLOv8n** | Bounding boxes for 14 specific abnormalities | Bounding Boxes | [Try the App](https://vimedhpbqdfx8fp59ynu3y.streamlit.app) |
| **Abnormality Classification** | Multi-Label Classification | **EfficientNet-B0** | Confidence scores for presence of 14 abnormalities | Grad-CAM | [Try the App](https://xdhu4se4zcpsgxg2aj82s6.streamlit.app) |
| **Binary Classification (CliniScan)** | Binary Classification | **ResNet-18** | Predicts: **Normal** vs. **Abnormal** | Grad-CAM | [Try the App](https://fcc2y5jgqqbry5suwaiyur.streamlit.app) |



### Getting Started

Detailed, model-specific instructions for installation, data preparation, training, and deployment (including required dependencies and scripts) are provided in the separate README files:

* **Object Detection (YOLOv8n):** [`README_detection.md`](README_detection.md)
* **Multi-Label Classification (EfficientNet-B0):** [`README_classification.md`](README_classification.md)
* **Binary Classification (ResNet-18):** [`README_binary_classification.md`](README_binary_classification.md)

### Installation (General)

For a general setup across all projects, you will need to install the following core dependencies:

```bash
# Core Machine Learning Frameworks
pip install torch ultralytics pytorch-lightning efficientnet-pytorch

# Data Science and Utilities
pip install numpy pandas scikit-learn tqdm

# Image Processing and Deployment
pip install streamlit pillow opencv-python matplotlib seaborn albumentations grad-cam

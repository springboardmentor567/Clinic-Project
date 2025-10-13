# CliniScan: Lung-Abnormality Classificaton & Detection on Chest X-rays

This project demonstrates two distinct but complementary approaches for analyzing Chest X-ray (CXR) images using state-of-the-art Deep Learning models: **Object Detection** (YOLOv8) to localize abnormalities and **Multi-Label Classification** (ResNet50) to identify their presence.

Both tasks use the **VinBigData Chest X-ray** dataset, targeting **14 different lung abnormalities**.

---

## Project Overview

The project is structured into two main parts, both deployed within a single, unified Streamlit application:

### Part 1: Abnormality Detection (YOLOv8)

The goal is to accurately **locate and identify** the 14 specific lung abnormalities within a chest X-ray image using **bounding boxes**.

| Feature | Description |
| :--- | :--- |
| **Model** | **YOLOv8n** (Nano) |
| **Task** | **Object Detection** (Detecting 14 abnormalities) |
| **Image Size** | $512 \times 512$ pixels |

### Part 2: Multi-Label Classification (ResNet)

The goal is to **classify** an entire CXR image to determine the **presence or absence** of the 14 abnormalities simultaneously.

| Feature | Description |
| :--- | :--- |
| **Model** | **ResNet50** (Pretrained) |
| **Task** | **Multi-Label Image Classification** (14 classes) |
| **Image Size** | $224 \times 224$ pixels |

### Shared Features
* **Dataset:** VinBigData Chest X-ray (Filtered)
* **Deployment:** Unified Streamlit Web Application (Streamlit Community Cloud / Hugging Face Spaces)

---

## 🚀 Live Demo

You can test the unified AI tool directly in your browser. The application hosts both the YOLOv8 Detection model and the ResNet Classification model.

**Application Link:** **[INSERT LIVE APP LINK HERE (e.g., Streamlit Share URL or HF Spaces URL)]**

---

## Technologies & Dependencies

* **Python:** $3.8+$
* **Deep Learning:** `torch`, `torchvision`, `ultralytics` (**YOLOv8**)
* **Data Science:** `numpy`, `pandas`, `sklearn`, `tqdm`
* **Visualization:** `matplotlib`, `seaborn`
* **Deployment:** `streamlit`, `Pillow` (PIL)
* **Utilities:** `os`, `shutil`, `glob`, `yaml`, `cv2`

### Installation

1.  **Clone the repository** (if applicable).
2.  **Install the main ML framework:**
    ```bash
    pip install ultralytics
    ```
3.  **Install other dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
*(Note: A `requirements.txt` file detailing all listed dependencies is assumed.)*

---

## Usage: Training & Deployment

### Phase 1: Data Preparation & Training (Kaggle/Colab)

This phase involves running the separate training pipelines for the two models.

1.  **Prepare datasets** (YOLO format for detection; one-hot labels for classification).
2.  **Configure and Train YOLOv8** for object detection, saving the best weights (`yolov8n_trained.pt`).
3.  **Configure and Train ResNet50** for multi-label classification, saving the best weights (`resnet50_multilabel_vinbigdata.pt`).
*(All training code and preprocessing steps are typically included in an accompanying notebook.)*

---

### Phase 2: Unified Deployment (Cloud/Local)

The **`app.py`** script serves as the unified interface, using Streamlit Tabs to allow users to switch between the two trained models.

1.  **Model Setup:**
    * Download both trained weights: `yolov8n_trained.pt` and `resnet50_multilabel_vinbigdata.pt`.
    * Ensure `app.py` loads both models efficiently using Streamlit's caching (`@st.cache_resource`).

2.  **Run Locally (Development/Testing):**
    ```bash
    streamlit run app.py
    ```
    * Access locally at: `http://localhost:8501`.

3.  **Cloud Deployment (Public Access):**
    * Push the `app.py`, `requirements.txt`, and trained weights to a public repository (e.g., GitHub).
    * Deploy the repository using a service like **Streamlit Community Cloud** or **Hugging Face Spaces** to generate the permanent, public URL for the **Live Demo** section above.

---

## Application Snapshots

The application uses tabs to cleanly separate the two functionalities. These images illustrate the Object Detection tab (YOLOv8):



# Chest X-Ray Abnormality Detection (YOLOv8)

This project demonstrates a complete end-to-end pipeline for **object detection** on Chest X-ray (CXR) images. Using the **VinBigData Chest X-ray** dataset, a **YOLOv8n (Nano)** model is trained to detect and localize **14 different lung abnormalities** using bounding boxes.

The trained model is then deployed via an interactive **Streamlit** web application for local inference.

---

## Project Overview

The goal is to accurately locate and identify **14 specific lung abnormalities** within a chest X-ray image.

### Workflow Highlights:
* **Data Preprocessing:** Cleaning and preparing the VinBigData annotations.
* **Custom Split:** Using **GroupKFold** for a robust train/validation split to prevent data leakage (since multiple annotations can belong to one image).
* **Model Training:** Training the state-of-the-art **YOLOv8** model.
* **Deployment:** Simple, fast deployment via a Streamlit web app.

### Feature Details

| Feature | Description |
| :--- | :--- |
| **Model** | **YOLOv8n** (Nano) |
| **Dataset** | VinBigData Chest X-ray (Filtered) |
| **Task** | **Object Detection** (Detecting 14 abnormalities) |
| **Training Env** | Online IDE (Kaggle/Colab) |
| **Deployment** | Streamlit (Local Host) |
| **Image Size** | $512 \times 512$ pixels |

---

## Technologies & Dependencies

* **Python:** $3.8+$
* **Deep Learning:** `torch`, `ultralytics` (**YOLOv8**)
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

---

## Usage

The project consists of two main phases: Training and Deployment.

### Phase 1: Data Preparation & Training (Kaggle/Colab)

1.  **Prepare dataset in YOLO format:** Convert the bounding box coordinates into the normalized YOLO format (`.txt` files).
2.  **Configure training:** Set up paths and classes in the `vinbigdata.yaml` configuration file.
3.  **Train YOLOv8 model:** Use the base `yolov8n.pt` weights for training.
4.  **Run inference** on validation images to check performance.
*(All code and preprocessing steps are included in the accompanying notebook.)*

### Phase 2: Deployment (Local Machine)

**Step 1: Model Setup**

* **Model Weights:** Download the trained weights from the kaggle runtime.

* **Deployment Script:** Give the path to the trained model.Save the deployment script as `app.py`.

**Step 2: Run the Application**
```bash
streamlit run app.py
```

Open your browser at: http://localhost:8501

**Deployment Features**
Allows a user to upload a chest X-ray image.

Runs inference locally using the deployed YOLOv8n model.

Displays the image with predicted bounding boxes and corresponding abnormality labels.




**Snapshots of the application:**
![CliniScan-YOLOv8_page-0001](https://github.com/user-attachments/assets/88ab5ba5-0ccb-4ec5-af0b-917c70f48bcf)

![CliniScan-YOLOv8_page-0002](https://github.com/user-attachments/assets/b373281b-0a37-4a0e-89b9-1c4080052335)

![CliniScan-YOLOv8_page-0003](https://github.com/user-attachments/assets/f073d84a-62f8-4e6a-8001-d89549657b8f)











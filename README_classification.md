# Chest X-Ray Abnormality Detection and Classification

This project demonstrates two distinct but complementary approaches for analyzing Chest X-ray (CXR) images using state-of-the-art Deep Learning models: **Object Detection** to localize abnormalities and **Multi-Label Classification** to identify their presence.

Both tasks use the **VinBigData Chest X-ray** dataset, targeting **14 different lung abnormalities**.

---

## Project Overview

The project is structured into two main parts:

### Part 1: Abnormality Detection (YOLOv8)

The goal of this part is to accurately **locate and identify** the 14 specific lung abnormalities within a chest X-ray image using **bounding boxes**.

#### Workflow Highlights (Detection):
* **Data Preprocessing:** Cleaning and preparing the VinBigData annotations into the YOLO format (`.txt` files).
* **Custom Split:** Using **GroupKFold** for a robust train/validation split to prevent data leakage (since multiple annotations can belong to one image).
* **Model Training:** Training the state-of-the-art **YOLOv8** model.
* **Deployment:** Simple, fast deployment via an interactive **Streamlit** web app for local inference.

| Feature | Description |
| :--- | :--- |
| **Model** | **YOLOv8n** (Nano) |
| **Dataset** | VinBigData Chest X-ray (Filtered) |
| **Task** | **Object Detection** (Detecting 14 abnormalities) |
| **Deployment** | Streamlit (Local Host) |
| **Image Size** | $512 \times 512$ pixels |

### Part 2: Multi-Label Classification (ResNet)

The goal of this part is to **classify** an entire CXR image to determine the **presence or absence** of the 14 abnormalities simultaneously.

#### Workflow Highlights (Classification):
* **Label Preparation:** Creating **one-hot encoded** labels for each image to handle the multi-label nature (an image can have $0$, $1$, or multiple abnormalities).
* **Custom Data Handler:** Implementing a custom PyTorch `Dataset` to load images and their corresponding multi-labels.
* **Transfer Learning:** Utilizing a **pretrained ResNet50** model.
* **Evaluation:** Using appropriate multi-label metrics like **BCEWithLogitsLoss**, **Classification Report**, and **ROC-AUC** per class.

| Feature | Description |
| :--- | :--- |
| **Model** | **ResNet50** (Pretrained) |
| **Dataset** | VinBigData Chest X-ray (Filtered) |
| **Task** | **Multi-Label Image Classification** (14 classes) |
| **Image Size** | $224 \times 224$ pixels |

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

## Usage

The project consists of two main analytical phases: Object Detection and Multi-Label Classification.

### Phase 1: Abnormality Detection (YOLOv8)

This phase involves training the object detection model and deploying it.

1.  **Data Preparation & Training (Kaggle/Colab):**
    * Prepare dataset in YOLO format: Convert the bounding box coordinates into the normalized YOLO format (`.txt` files).
    * Configure training: Set up paths and classes in the `vinbigdata.yaml` configuration file.
    * Train YOLOv8 model: Use the base `yolov8n.pt` weights for training.
    * Run inference on validation images to check performance.
    *(All code and preprocessing steps are typically included in an accompanying notebook.)*

2.  **Deployment (Local Machine):**
    * **Model Setup:** Download the trained weights and save the deployment script as `app.py`.
    * **Run the Application:**
        ```bash
        streamlit run app.py
        ```
    * Open your browser at: `http://localhost:8501`.
    * **Deployment Features:** Allows users to upload a chest X-ray image, runs local inference using the deployed YOLOv8n model, and displays the image with predicted bounding boxes and corresponding abnormality labels.

### Phase 2: Multi-Label Classification (ResNet)

This phase focuses on the multi-label image classification pipeline.

1.  **Target Preparation:** Run the initial data processing to create the **one-hot encoded** label matrix (`multi_labels_df`) for all unique images in the dataset.
2.  **Data Loading:** Use the **custom `MultiLabelDataset`** class to handle image loading and multi-label tensor preparation.
3.  **Split and Load:** Split the data into training and validation sets (e.g., $80/20$) and set up the PyTorch `DataLoader`s.
4.  **Model Initialization:** Load the pretrained **ResNet50** model and modify the final layer (`model.fc`) to output a vector of size $14$ (one for each class).
5.  **Training:** Train the model using **`nn.BCEWithLogitsLoss()`** as the criterion, which is suitable for multi-label classification.
6.  **Evaluation:** After training, use the validation set to calculate performance metrics:
    * **Classification Report:** Provides precision, recall, and F1-score for each of the $14$ abnormality classes based on a $0.5$ threshold.
    * **ROC-AUC:** Calculates the Area Under the Curve (AUC) for the Receiver Operating Characteristic (ROC) curve for each class, providing a robust measure of performance.
7.  **Model Saving:** The trained model weights are saved (e.g., `/kaggle/working/resnet50_multilabel_vinbigdata.pt`).

---

## Application Snapshots (YOLOv8 Deployment)

The following images illustrate the deployed Streamlit application in action for the Object Detection task:

| Description | Snapshot |
| :--- | :--- |
| **Application Welcome Page** | ![CliniScan-YOLOv8_page-0001](https://github.com/user-attachments/assets/88ab5ba5-0ccb-4ec5-af0b-917c70f48bcf) |
| **Image Upload Interface** | ![CliniScan-YOLOv8_page-0002](https://github.com/user-attachments/assets/b373281b-0a37-4a0e-89b9-1c4080052335) |
| **Inference Result (Image with Bounding Boxes)** | ![CliniScan-YOLOv8_page-0003](https://github.com/user-attachments/assets/f073d84a-62f8-4e6a-8001-d89549657b8f) |

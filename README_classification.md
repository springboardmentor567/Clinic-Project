# Chest X-Ray Abnormality Classification (EfficientNet) 

This project demonstrates a complete end-to-end pipeline for **multi-label image classification** on Chest X-ray (CXR) images. It uses the **VinBigData Chest X-ray** dataset to train an **EfficientNet-B0** model to classify the presence of **14 different lung abnormalities** simultaneously.

The trained model is deployed via an interactive **Streamlit** web application, featuring **Grad-CAM** explainability for crucial model interpretability.



## Project Overview

The goal is to accurately identify and classify **multiple thoracic abnormalities** from a single chest X-ray image.

### Workflow Highlights:
* **Multi-label Target Engineering:** Preparing the data for 14 simultaneous classifications.
* **Robust Training:** Leveraging **PyTorch Lightning** for structured and efficient model training.
* **Deployment:** Interactive web application using Streamlit.
* **Interpretability:** **Grad-CAM** visualizations to highlight important image regions driving the model's prediction.

### Feature Details

| Feature | Description |
| :--- | :--- |
| **Model** | **EfficientNet-B0** (Pre-trained) |
| **Dataset** | VinBigData Chest X-ray |
| **Task** | **Multi-Label Image Classification** (14 abnormalities) |
| **Training Env** | Kaggle/Colab (PyTorch Lightning) |
| **Deployment** | Streamlit (Local Host) |
| **Explainability** | **Grad-CAM** |
| **Image Size** | $512 \times 512$ pixels |



## Technologies & Dependencies

* **Python:** $3.8+$
* **Deep Learning:** `torch`, `pytorch-lightning`
* **Model Architecture:** `efficientnet-pytorch`
* **Explainability:** `grad-cam`
* **Data Science:** `numpy`, `pandas`, `sklearn`, `tqdm`
* **Image Handling:** `Pillow` (PIL), `torchvision`
* **Deployment:** `streamlit`

### Installation

1.  **Clone the repository** (if applicable).
2.  **Install dependencies using `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

The project consists of two main phases: Training and Deployment.

### Phase 1: Data Preparation & Training (Kaggle/Colab)

1.  **Convert object detection metadata** into multi-label classification targets.
2.  **Define PyTorch Dataset and DataLoaders** with appropriate augmentations.
3.  **Train EfficientNet-B0** using PyTorch Lightning (Notebook Steps 6-7).
4.  **Save the optimized weights file:** `finalvinbigdataclassifier.pth`
5.  **Generate Grad-CAM visualizations** for interpretability.

### Phase 2: Deployment (Local Machine)

**Step 1: Model Setup**

* **Model Weights:** Download the trained weights from the kaggle runtime.

* **Deployment Script:** Give the path to the trained model.Save the deployment code as `app.py`.

**Step 2: Run the Application**
```bash
streamlit run app.py
```
Open your browser at: http://localhost:8501

**Deployment Features**
Predicts 14 lung abnormalities from a single X-ray.

Provides multi-label predictions with confidence scores.

Grad-CAM visualizations highlight important image regions contributing to the classification.

Simple Streamlit UI for easy image upload and inference.

**Snapshots of the application:**
![CliniScan-ENB0_page-0001](https://github.com/user-attachments/assets/102a42b2-b326-496a-bcb9-1a8cbd2a4ad4)

![CliniScan-ENB0_page-0002](https://github.com/user-attachments/assets/246568c1-7d90-48c1-9f34-ed3dea1d6ff8)




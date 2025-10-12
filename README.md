# Lung Detection – Streamlit-Based Pneumonia Classification App

This project focuses on detecting **Pneumonia** from chest X-ray images using a **ResNet18 convolutional neural network** integrated with a **Streamlit web interface**. The application supports both single and batch image inference and provides Grad-CAM visualizations for interpretability.

## Features
- **Classification:** Predicts whether an uploaded X-ray image is Normal or Pneumonia.
- **Confidence Metrics:** Displays class probabilities and confidence bars.
- **Grad-CAM Visualization:** Highlights the most influential regions in the X-ray that contributed to the model’s prediction.
- **Batch Inference:** Enables simultaneous processing of multiple images with summarized results.
- **Result Export:** Allows downloading of prediction outputs in CSV format.

## Directory Structure
- `app.py`: Baseline Streamlit interface.
- `streamlit_app.py`: Enhanced interface with visualization controls and batch functionality.
- `train.py`: Contains the training pipeline for ResNet18 and model checkpoint saving.
- `data_preparation.py`: Handles dataset loading, preprocessing, and splitting into train, validation, and test sets.
- `data/Images`: Dataset directory organized by NORMAL and PNEUMONIA categories.
- `model.pth`: Pretrained weights of the final model.
- `requirements.txt`: List of required Python dependencies.

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/Ektajoge55/lung_detection.git
   cd lung_detection
   ```
2. Install dependencies (preferably in a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

## Application Usage
1. Access the application through the local URL displayed in the terminal.
2. Configure sidebar parameters such as:
   - Image resize dimensions for model input.
   - Grad-CAM overlay toggle.
   - Probability threshold for Pneumonia detection.
   - Batch mode activation for multiple uploads.
3. Upload X-ray images to perform inference.
4. Review predicted results, class probabilities, and Grad-CAM overlays.
5. Export results to CSV if needed.

## Technical Overview
### Model
- **Architecture:** ResNet18 with a modified final fully connected layer for binary classification.
- **Weights:** Pretrained model weights stored in `model.pth`.

### Preprocessing
- Images resized to 224×224 pixels.
- Normalization applied using ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.

### Inference Process
- Model produces logits for `[Normal, Pneumonia]`.
- Softmax function converts logits into probabilities.
- The class with the highest probability is displayed as the final prediction.

### Grad-CAM (Explainability)
- Gradients and activations captured from the final convolutional block (`layer4[1].conv2`).
- Weighted activation maps are combined and overlaid on the original image to visualize decision regions.

## Model Training (Optional)
To train or fine-tune the model:
```bash
python train.py
```
- Dataset must be placed in the path `data/Images/{train,val,test}/{NORMAL,PNEUMONIA}`.
- Hyperparameters such as learning rate, epochs, or device can be adjusted within `train.py`.


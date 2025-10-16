

import os
import cv2
import pydicom
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

# --- DICOM to PNG Conversion ---

def dicom_to_png(dicom_path: str, png_path: str) -> bool:
    """
    Convert a single DICOM file to a PNG image file.

    Args:
        dicom_path (str): Path to the input DICOM file.
        png_path (str): Path to save the output PNG file.

    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    try:
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array

        # Normalize pixel values to 0-255 and convert to 8-bit unsigned integer
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img_uint8 = img_norm.astype("uint8")

        cv2.imwrite(png_path, img_uint8)
        return True
    except Exception as e:
        print(f"❌ Failed to convert {dicom_path}: {e}")
        return False


def convert_dataset(dicom_dir: str, output_dir: str, annotations_csv: str):
    """
    Batch convert a DICOM dataset to PNG format based on an annotations file.

    Args:
        dicom_dir (str): Directory containing the DICOM files.
        output_dir (str): Directory to save the converted PNG files.
        annotations_csv (str): Path to the CSV file with image IDs.
    """
    os.makedirs(output_dir, exist_ok=True)
    annotations = pd.read_csv(annotations_csv)

    for _, row in annotations.iterrows():
        image_id = row['image_id']
        dicom_file = os.path.join(dicom_dir, f"{image_id}.dicom")
        png_file = os.path.join(output_dir, f"{image_id}.png")

        if os.path.exists(dicom_file):
            if dicom_to_png(dicom_file, png_file):
                print(f"✅ Converted: {dicom_file} → {png_file}")
        else:
            print(f"⚠️ Missing file: {dicom_file}")


# --- Custom PyTorch Dataset Class ---

class VinDrCXRDataset(Dataset):
    """Custom PyTorch Dataset for the VinDr-CXR dataset."""
    def __init__(self, df, img_dir, transform=None, task='classification'):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.task = task  # 'classification' or 'detection'

        # Get unique labels and create a mapping to integer IDs
        self.labels = sorted(df['label'].unique())
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}

        # Group annotations by image ID
        self.grouped_images = self.df.groupby('image_id')
        self.image_ids = list(self.grouped_images.groups.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        image = Image.open(img_path).convert("RGB")
        annotations = self.grouped_images.get_group(img_id)

        if self.task == 'classification':
            # Create a multi-hot encoded label vector
            label_vector = torch.zeros(len(self.labels), dtype=torch.float)
            for _, row in annotations.iterrows():
                label_vector[self.label_to_id[row['label']]] = 1.0

            if self.transform:
                image = self.transform(image)
            return image, label_vector

        elif self.task == 'detection':
            # Extract bounding boxes and labels for object detection
            boxes, labels = [], []
            for _, row in annotations.iterrows():
                boxes.append([row['x_min'], row['y_min'], row['x_max'], row['y_max']])
                labels.append(self.label_to_id[row['label']])

            target = {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64)
            }

            if self.transform:
                image = self.transform(image)
            return image, target

import os
import shutil
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
png_dir = "png_train"  # Directory containing all PNG images
csv_path = "vinbigdata-chest-xray-abnormalities-detection/train.csv"  # Metadata file path
output_dir = "cls_dataset_binary"  # Output directory for processed dataset
VAL_RATIO = 0.2  # 20% of data reserved for validation
random.seed(42)  # Reproducibility

# ------------------------------------------------------------------------------
# Load and preprocess CSV file
# ------------------------------------------------------------------------------
df = pd.read_csv(csv_path)

# Create a new column: "binary_label"
# If "class_name" == "No finding", label as "normal"; otherwise "abnormal"
df["binary_label"] = df["class_name"].apply(lambda x: "normal" if x == "No finding" else "abnormal")

# Get unique image IDs for each category
normal_ids = df[df["binary_label"] == "normal"]["image_id"].unique().tolist()
abnormal_ids = df[df["binary_label"] == "abnormal"]["image_id"].unique().tolist()

# ------------------------------------------------------------------------------
# Balance the dataset
# ------------------------------------------------------------------------------
# Ensure equal number of normal and abnormal samples by random downsampling
if len(normal_ids) > len(abnormal_ids):
    normal_ids = random.sample(normal_ids, len(abnormal_ids))

# Combine and label all selected IDs
all_ids = normal_ids + abnormal_ids
labels_map = {img_id: ("abnormal" if img_id in abnormal_ids else "normal") for img_id in all_ids}

# ------------------------------------------------------------------------------
# Train-validation split (stratified)
# ------------------------------------------------------------------------------
# Create a binary list for stratification (1 = abnormal, 0 = normal)
stratify_labels = [1 if labels_map[i] == "abnormal" else 0 for i in all_ids]

# Split into train and validation sets
train_ids, val_ids = train_test_split(
    all_ids, test_size=VAL_RATIO, random_state=42, stratify=stratify_labels
)

# ------------------------------------------------------------------------------
# Function to copy images to respective folders
# ------------------------------------------------------------------------------
def copy_files(ids, split):
    """
    Copies images corresponding to given IDs into structured directories.
    Example directory structure:
        cls_dataset_binary/
            train/
                normal/
                abnormal/
            val/
                normal/
                abnormal/
    """
    for img_id in ids:
        img_src = os.path.join(png_dir, f"{img_id}.png")
        if not os.path.exists(img_src):
            continue  # Skip missing files gracefully
        label = labels_map[img_id]
        dst_dir = os.path.join(output_dir, split, label)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(img_src, os.path.join(dst_dir, f"{img_id}.png"))

# ------------------------------------------------------------------------------
# Execute dataset preparation
# ------------------------------------------------------------------------------
copy_files(train_ids, "train")
copy_files(val_ids, "val")

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
print("✅ Balanced binary classification dataset prepared:")
print(f"   Train: {len(train_ids)} images "
      f"({len([i for i in train_ids if labels_map[i]=='normal'])} normal, "
      f"{len([i for i in train_ids if labels_map[i]=='abnormal'])} abnormal)")
print(f"   Val:   {len(val_ids)} images "
      f"({len([i for i in val_ids if labels_map[i]=='normal'])} normal, "
      f"{len([i for i in val_ids if labels_map[i]=='abnormal'])} abnormal)")

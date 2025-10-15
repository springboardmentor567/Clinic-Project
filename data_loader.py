#data.py
import os
import zipfile
import pandas as pd
import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt
import random

# --------------------
# 1. Paths
# --------------------
zip_path = r"D:\downloads\vinbigdata-chest-xray-abnormalities-detection.zip"
extract_dir = r"D:\vinbigdata_extracted"
output_images = r"D:\vinbigdata_processed\images"
output_labels = r"D:\vinbigdata_processed\labels"

os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

# --------------------
# 2. Extract zip (only once)
# --------------------
if not os.path.exists(extract_dir):
    print("Extracting dataset... (this may take a while)")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
else:
    print("Dataset already extracted.")

# --------------------
# 3. Load CSV
# --------------------
csv_path = os.path.join(extract_dir, "train.csv")
df = pd.read_csv(csv_path)
print("CSV loaded with shape:", df.shape)

# --------------------
# 4. Generate YOLO classes.txt (include both abnormal + normal)
# --------------------
classes = sorted(df["class_name"].unique())
classes_path = os.path.join(output_labels, "classes.txt")
with open(classes_path, "w") as f:
    for c in classes:
        f.write(c + "\n")
print("YOLO classes saved at:", classes_path)

# --------------------
# 5. Take smaller subset to save storage
# --------------------
sample_df = df.sample(600, random_state=42).reset_index(drop=True)  
# includes both abnormal + normal

print("Processing", len(sample_df), "samples")

# --------------------
# 6. Process images
# --------------------
for idx, row in sample_df.iterrows():
    image_id = row["image_id"]
    dicom_path = os.path.join(extract_dir, "train", image_id + ".dicom")
    png_path = os.path.join(output_images, image_id + ".png")
    label_path = os.path.join(output_labels, image_id + ".txt")

    # Skip if already processed
    if os.path.exists(png_path):
        continue

    if not os.path.exists(dicom_path):
        continue

    try:
        # Convert DICOM → PNG
        dcm = pydicom.dcmread(dicom_path)
        img = dcm.pixel_array
        img = cv2.convertScaleAbs(img, alpha=255.0/np.max(img))
        cv2.imwrite(png_path, img)

        # Create YOLO label (if bbox exists)
        if pd.notnull(row["x_min"]) and pd.notnull(row["y_min"]) and pd.notnull(row["x_max"]) and pd.notnull(row["y_max"]):
            h, w = img.shape
            x_min, y_min, x_max, y_max = row[["x_min","y_min","x_max","y_max"]]
            x_center = (x_min + x_max) / (2*w)
            y_center = (y_min + y_max) / (2*h)
            bw = (x_max - x_min) / w
            bh = (y_max - y_min) / h
            with open(label_path, "w") as f:
                f.write(f"{row['class_id']} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")
        else:
            # If no finding → write empty file
            open(label_path, "w").close()

    except Exception as e:
        print("Error with", image_id, e)

print("✅ Processing complete.")

# --------------------
# 7. Show one random sample (with bbox if abnormal)
# --------------------
valid_samples = sample_df.sample(20, random_state=42)
for _, sample in valid_samples.iterrows():
    sample_path = os.path.join(output_images, sample["image_id"] + ".png")
    if not os.path.exists(sample_path):
        continue

    img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        continue

    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap="gray")
    plt.title(f"Sample: {sample['class_name']}")

    if pd.notnull(sample["x_min"]) and pd.notnull(sample["y_min"]):
        x_min, y_min, x_max, y_max = sample[["x_min","y_min","x_max","y_max"]]
        plt.gca().add_patch(
            plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                          edgecolor="red", facecolor="none", linewidth=2)
        )

    plt.axis("off")
    plt.show()
    break

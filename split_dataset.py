#split_dataset.py
import os
import random
import shutil

# Paths
BASE_DIR = r"D:\vinbigdata_processed"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
LABEL_DIR = os.path.join(BASE_DIR, "labels")

OUTPUT_DIR = r"D:\vinbigdata_yolo"
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

# Parameters
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Collect all images
all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".png")]
random.shuffle(all_images)

# Split
n_total = len(all_images)
n_train = int(n_total * train_ratio)
n_val = int(n_total * val_ratio)

train_files = all_images[:n_train]
val_files = all_images[n_train:n_train+n_val]
test_files = all_images[n_train+n_val:]

splits = {"train": train_files, "val": val_files, "test": test_files}

# Move files
for split, files in splits.items():
    for img_file in files:
        label_file = img_file.replace(".png", ".txt")
        
        # Copy image
        shutil.copy(os.path.join(IMAGE_DIR, img_file),
                    os.path.join(OUTPUT_DIR, split, "images", img_file))
        
        # Copy label
        src_label = os.path.join(LABEL_DIR, label_file)
        if os.path.exists(src_label):
            shutil.copy(src_label,
                        os.path.join(OUTPUT_DIR, split, "labels", label_file))

print("✅ Dataset split completed!")
print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

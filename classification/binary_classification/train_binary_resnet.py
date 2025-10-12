# ==============================================================================
# train_resnet_binary.py
# ------------------------------------------------------------------------------
# Purpose:
#     Train a ResNet-18 model for binary classification ("normal" vs "abnormal")
#     using chest X-ray images from the VinBigData dataset.
#
# Description:
#     - Loads dataset prepared by `prepare_binary_dataset.py`
#     - Applies data augmentation using Albumentations
#     - Uses ResNet-18 with pretrained weights
#     - Tracks F1-score and saves the best model checkpoint
#     - Includes early stopping to prevent overfitting
#
# Dependencies:
#     torch, torchvision, albumentations, sklearn, numpy, cv2, pandas
# ==============================================================================

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau  # Optional scheduler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score, roc_auc_score
import cv2
import numpy as np
import os

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
train_dir = "cls_dataset_binary/train"     # Training data folder
val_dir = "cls_dataset_binary/val"         # Validation data folder
batch_size = 32
num_epochs = 20
lr = 0.001
device = torch.device("cpu")               # Force CPU usage (can switch to GPU)
model_save_path = "best_classification_model.pth"

print(f"Using device: {device}")

# ------------------------------------------------------------------------------
# Data Augmentation using Albumentations
# ------------------------------------------------------------------------------
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(p=0.2),
    # A.CoarseDropout(p=0.2) can be added for stronger regularization
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ------------------------------------------------------------------------------
# Custom Dataset Class
# ------------------------------------------------------------------------------
class AlbumentationsDataset(Dataset):
    """
    Custom dataset class to handle images and labels from directory structure:
        cls_dataset_binary/
            train/
                normal/
                abnormal/
            val/
                normal/
                abnormal/
    """
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        # List all subdirectories as class names
        self.classes = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d))])
        # Gather all image paths and corresponding class labels
        for label_idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls)
            for f in os.listdir(cls_dir):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(cls_dir, f), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, torch.tensor(label, dtype=torch.long)

# ------------------------------------------------------------------------------
# Dataset and DataLoader Setup
# ------------------------------------------------------------------------------
try:
    train_dataset = AlbumentationsDataset(train_dir, transform=train_transform)
    val_dataset = AlbumentationsDataset(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    num_classes = len(train_dataset.classes)
    if num_classes < 2:
        raise ValueError("Must have at least 2 classes for classification.")

    print(f"Classes: {train_dataset.classes}, Number of classes: {num_classes}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

except Exception as e:
    print(f"Error during dataset setup: {e}")
    exit()

# ------------------------------------------------------------------------------
# Model Setup and Resume Logic
# ------------------------------------------------------------------------------
model = resnet18(weights=ResNet18_Weights.DEFAULT)  # Load pretrained ResNet18
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)     # Replace final layer

# Trackers for best validation performance
best_val_f1 = 0.0
best_model_wts = model.state_dict()
patience = 5
epochs_no_improve = 0

# Resume from checkpoint if exists
if os.path.exists(model_save_path):
    try:
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"\nLoaded existing model: {model_save_path}, evaluating baseline F1...")

        # Evaluate model to initialize F1 baseline
        model.eval()
        all_labels, all_preds = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                all_labels.extend(labels.numpy())
                all_preds.extend(preds.numpy())

        best_val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        best_model_wts = model.state_dict()
        print(f"✅ Baseline F1-score: {best_val_f1:.4f}")

    except Exception as e:
        print(f"⚠️ Could not load checkpoint. Starting fresh. Error: {e}")

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ------------------------------------------------------------------------------
# Training & Validation Loop with Early Stopping
# ------------------------------------------------------------------------------
for epoch in range(num_epochs):
    print(f"\n===== Epoch [{epoch+1}/{num_epochs}] =====")

    # ---------------- Training ----------------
    model.train()
    running_loss = 0.0
    for step, (images, labels) in enumerate(train_loader, start=1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

        if step % 10 == 0 or step == len(train_loader):
            print(f"[Train] Step {step}/{len(train_loader)}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_dataset)
    print(f"[Train] Epoch Loss: {epoch_loss:.4f}")

    # ---------------- Validation ----------------
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for step, (images, labels) in enumerate(val_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            if num_classes == 2:
                all_probs.extend(probs[:, 1].cpu().numpy())

    val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    val_auc = None
    if num_classes == 2 and len(np.unique(all_labels)) == 2:
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            pass

    print(f"[Val] Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc if val_auc else 'N/A'}")

    # ---------------- Early Stopping ----------------
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_wts = model.state_dict()
        torch.save(best_model_wts, model_save_path)
        print(f"[Save] ✅ Model improved (F1: {best_val_f1:.4f}) — saved checkpoint.")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"[Early Stop] No improvement for {epochs_no_improve}/{patience} epochs.")
        if epochs_no_improve >= patience:
            print("\n⏹️ Early stopping triggered.")
            break

# ------------------------------------------------------------------------------
#  Restore Best Weights
# ------------------------------------------------------------------------------
model.load_state_dict(best_model_wts)
print("\n✅ Training Complete — Restored best model weights.")
print(f"Best Validation F1-score: {best_val_f1:.4f}")

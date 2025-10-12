"""
Chest X-Ray Pneumonia Detection - Model Training Script
Created by: Ekta Joge
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights
from data_preparation import train_loader, val_loader

print("🚀 train.py started...")

# Model setup (ResNet18)
def get_model():
    print("📥 Loading ResNet18 with pretrained weights...")
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: Pneumonia / Normal
    print("✅ Model ready")
    return model

def train_model(model, train_loader, val_loader, num_epochs=5, lr=1e-4, device='cpu'):
    print(f"📌 Training on {device} for {num_epochs} epochs...")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\n🔄 Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"📊 Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'model.pth')
            print("💾 Best model saved.")

    print("🎉 Training complete.")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Using device: {device}")
    model = get_model()
    print("📊 Train loader size:", len(train_loader.dataset), "Val loader size:", len(val_loader.dataset))
    train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4, device=device)

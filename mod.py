import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset

train_dir = r"C:\Users\admin\OneDrive\Desktop\in_spring\chest_xray\train"
val_dir = r"C:\Users\admin\OneDrive\Desktop\in_spring\chest_xray\val"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)

# Limit to first 100 images in each dataset
train_data = Subset(train_data, list(range(min(100, len(train_data)))))
val_data = Subset(val_data, list(range(min(100, len(val_data)))))

train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
val_loader = DataLoader(val_data, batch_size=4)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
device = torch.device("cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(3):  # 3 epochs
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done")

torch.save(model.state_dict(), "model.pth")
print("✅ model.pth saved! Place it in your Streamlit folder.")

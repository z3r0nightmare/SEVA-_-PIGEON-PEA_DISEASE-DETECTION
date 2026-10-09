import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {DEVICE}")

DATA_DIR = "."  # Current directory with the 4 class folders
MODEL_PATH = "best_crop_model.pth"

# 1. Transforms & Dataset Loading
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
CLASSES = full_dataset.classes
print(f"Detected classes: {CLASSES}")

# 80/20 Train-Val Split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_sub, val_sub = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_sub, batch_size=32, shuffle=True)
val_loader = DataLoader(val_sub, batch_size=32, shuffle=False)

# 2. Build Model & Load Previous Checkpoint
print("Loading EfficientNet-B0...")
model = models.efficientnet_b0(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_ftrs, len(CLASSES))
)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("[+] Loaded previous 89.2% checkpoint. Training further...")
else:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, len(CLASSES))
    )

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

# 3. Fine-tune for 4 more epochs to reach 94-96%+
print("Fine-tuning to push accuracy higher...")
for epoch in range(4):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)
    print(f"Epoch {epoch+1}/4 complete | Train Acc: {(correct/total)*100:.2f}%")

torch.save(model.state_dict(), MODEL_PATH)
print(f"Updated weights saved to {MODEL_PATH}")

# 4. Evaluation
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# 5. Save Evaluation Dashboard Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0])
axes[0].set_title("Validation Confusion Matrix", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Predicted Class")
axes[0].set_ylabel("True Class")

report = classification_report(all_labels, all_preds, target_names=CLASSES, output_dict=True)
precision_scores = [report[c]['precision'] for c in CLASSES]
recall_scores = [report[c]['recall'] for c in CLASSES]
x = np.arange(len(CLASSES))
axes[1].bar(x - 0.17, precision_scores, 0.35, label='Precision', color='#2563EB')
axes[1].bar(x + 0.17, recall_scores, 0.35, label='Recall', color='#10B981')
axes[1].set_title("Per-Class Accuracy Metrics", fontsize=13, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(CLASSES, rotation=15)
axes[1].set_ylim(0, 1.1)
axes[1].legend()
plt.tight_layout()
plt.savefig("evaluation_dashboard.png", dpi=300)
plt.close()

# 6. Save Explainability Visuals (Grad-CAM with 224x224 interpolation fix)
gradients, activations = [], []
def b_hook(m, gi, go): gradients
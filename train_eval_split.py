import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["Healthy", "Leaf_Spot", "Leaf_webber", "Sterilic_mosaic"]
MODEL_PATH = "best_crop_model.pth"

print(f"[*] Compute device: {DEVICE}")

# 1. Load Model Weights
print("[*] Loading model weights from:", MODEL_PATH)
model = models.efficientnet_b0(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_ftrs, len(CLASSES))
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# 2. Setup Hooks
gradients = []
activations = []

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

def forward_hook(module, input, output):
    activations.append(output)

model.features[-1].register_forward_hook(forward_hook)
model.features[-1].register_full_backward_hook(backward_hook)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Locate Sample Images from the 4 folders
fig, axes = plt.subplots(2, len(CLASSES), figsize=(16, 7))

for idx, cls in enumerate(CLASSES):
    # Check current directory and subdirectories for the class folder
    candidates = [
        cls,
        os.path.join(".", cls),
        os.path.join("dataset_split", "val", cls),
        os.path.join("dataset_split", "train", cls)
    ]
    
    cls_dir = None
    for c in candidates:
        if os.path.isdir(c):
            cls_dir = c
            break

    if cls_dir is None:
        print(f"[-] Could not find folder for class: {cls}")
        continue

    valid_imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not valid_imgs:
        print(f"[-] No images found in: {cls_dir}")
        continue

    target_img_path = os.path.join(cls_dir, valid_imgs[0])
    print(f"[+] Processing {cls} using: {target_img_path}")

    raw_img = Image.open(target_img_path).convert('RGB').resize((224, 224))
    inp_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)

    gradients.clear()
    activations.clear()

    # Forward
    output = model(inp_tensor)
    pred_idx = torch.argmax(output).item()

    # Backward
    model.zero_grad()
    output[0, pred_idx].backward()

    # Grad-CAM computation
    grads = gradients[0]
    acts = activations[0]
    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    
    cam = torch.sum(weights * acts, dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().detach().numpy()
    
    if np.max(cam) != 0:
        cam = cam / np.max(cam)

    heatmap = plt.cm.jet(cam)[:, :, :3]
    img_arr = np.array(raw_img) / 255.0
    overlay = 0.5 * img_arr + 0.5 * heatmap

    # Row 1: Original
    axes[0, idx].imshow(raw_img)
    axes[0, idx].set_title(f"True: {cls}", fontsize=11, fontweight='bold')
    axes[0, idx].axis('off')

    # Row 2: Heatmap Overlay
    axes[1, idx].imshow(overlay)
    status_color = 'green' if pred_idx == idx else 'red'
    axes[1, idx].set_title(f"Pred: {CLASSES[pred_idx]}", fontsize=11, fontweight='bold', color=status_color)
    axes[1, idx].axis('off')

plt.suptitle("Model Diagnostic Focus (Grad-CAM Saliency Maps)", fontsize=15, fontweight='bold')
plt.tight_layout()

save_dest = os.path.abspath("gradcam_samples.png")
plt.savefig(save_dest, dpi=300)
plt.close()

print(f"\n[+] SUCCESS! Grad-CAM image written to:\n    {save_dest}")
# band_importance_analysis.py
import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import tifffile
import numpy as np
from captum.attr import IntegratedGradients
import pandas as pd
import matplotlib.pyplot as plt

# Define band labels
band_labels_short = [
    "B1 Coast", 
    "B2 Blue", 
    "B3 Green", 
    "B4 Red", 
    "B5 RE1", 
    "B6 RE2", 
    "B7 RE3", 
    "B8 NIR", 
    "B8A NIR2", 
    "B9 WV", 
    "B10 Cirrus", 
    "B11 SWIR1", 
    "B12 SWIR2"
]

band_labels_wavelength = [
    "B1 Coast (443)", 
    "B2 Blue (490)", 
    "B3 Green (560)", 
    "B4 Red (665)", 
    "B5 RE1 (705)", 
    "B6 RE2 (740)", 
    "B7 RE3 (783)", 
    "B8 NIR (842)", 
    "B8A NIR2 (865)", 
    "B9 WV (945)", 
    "B10 Cirrus (1380)", 
    "B11 SWIR1 (1610)", 
    "B12 SWIR2 (2190)"
]

# -----------------------------
# 1. Parameters to change
# -----------------------------
model_path = 'TrainedModels/MS_Deepest_40epoch_Trained.pth'  # path to your trained model
image_dir = 'EuroSAT_MS'                             # path to your dataset
batch_size = 32
num_bands = 13                                       # input channels
num_classes = 10                                     # output classes

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 2. Define your CNN class
# -----------------------------
# Make sure you have this definition (or import it if in image_models.py)
from image_models import Deepest_CNN
model = Deepest_CNN(num_bands, num_classes).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# -----------------------------
# 3. Load test dataset
# -----------------------------
def tiff_loader(path):
    img = tifffile.imread(path)
    if img.shape[0] == num_bands:
        img = np.transpose(img, (1, 2, 0))
    return img.astype(np.float32)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64))
])

full_dataset = ImageFolder(root=image_dir, transform=transform, loader=tiff_loader)

# Stratified split: 80% train, 10% val, 10% test
indices = list(range(len(full_dataset)))
labels = np.array(full_dataset.targets)

from sklearn.model_selection import train_test_split
train_idx, temp_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=42)
test_dataset = Subset(full_dataset, test_idx)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("step 3 complete")

# -----------------------------
# 4. Compute band importance
# -----------------------------
ig = IntegratedGradients(model)
total_attr = torch.zeros((num_bands,), device=device)
total_samples = 0

for batch_idx, (images, labels) in enumerate(test_loader):
    images = images.to(device)
    baseline = torch.zeros_like(images)

    preds = model(images).argmax(dim=1)

    for i in range(images.shape[0]):
        target_class = preds[i].item()
        attr, _ = ig.attribute(
            images[i:i+1],
            baselines=baseline[i:i+1],
            target=target_class,
            return_convergence_delta=True
        )
        total_attr += attr.abs().mean(dim=(2, 3)).squeeze()
        total_samples += 1

band_importance = total_attr / total_attr.sum()
band_importance_cpu = band_importance.detach().cpu().numpy()

# -----------------------------
# 5. Save CSV
# -----------------------------
csv_path = "band_importance.csv"
df_importance = pd.DataFrame({
    "Band": list(range(1, num_bands + 1)),
    "Importance": band_importance_cpu
})
df_importance.to_csv(csv_path, index=False)
print(f"Saved band importances to {csv_path}")

# -----------------------------
# 6. Plot
# -----------------------------
plt.figure(figsize=(12, 5))
plt.bar(band_labels_wavelength, band_importance_cpu)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Spectral Band (nm)")
plt.ylabel("Relative Importance")
plt.title("Spectral Band Importance (Integrated Gradients)")
plt.tight_layout()
plt.show()
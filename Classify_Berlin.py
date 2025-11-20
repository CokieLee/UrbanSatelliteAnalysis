# import os, numpy as np
# from PIL import Image
# import scipy.io as sio
# import torch
# from torchvision import transforms

# # ============ CONFIGURATION ============
# MAT_PATH = "/Users/ryanlutz/Desktop/berlin_multimodal.mat"
# OUTPUT_DIR = "Berlin_RGB_Predicted"
# MODEL_PATH = "Deepest_CNN/model_fin.pth"  # Path to your trained model

# # Import your model architecture
# from image_models import Deepest_CNN

# # Device
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # EuroSAT classes (must match training order)
# eurosat_classes = [
#     "AnnualCrop","Forest","HerbaceousVegetation","Highway","Industrial",
#     "Pasture","PermanentCrop","Residential","River","SeaLake"
# ]

# # ============ LOAD MODEL ============
# print("Loading trained model...")
# model = Deepest_CNN(3, 10).to(device)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
# model.eval()
# print(f"Model loaded from {MODEL_PATH}")

# # Option 2: Calculate from EuroSAT (safer - ensures consistency)
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader

# print("Calculating normalization from EuroSAT dataset...")
# eurosat_dataset = ImageFolder(root='EuroSAT_RGB', transform=transforms.ToTensor())
# eurosat_loader = DataLoader(eurosat_dataset, batch_size=32, shuffle=False, num_workers=0)

# from image_normalization import get_mean_stdev
# mean, stdev = get_mean_stdev(eurosat_loader)
# print(f'Using mean: {mean}, stdev: {stdev}')

# transform = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=mean, std=stdev)
# ])

# # ============ LOAD BERLIN DATA ============
# print("Loading Berlin dataset...")
# m = sio.loadmat(MAT_PATH)
# MSI = np.array(m["MSI"])
# H, W, C = MSI.shape
# print(f"MSI: {MSI.shape}, {MSI.dtype}")

# # ============ CREATE RGB ============
# print("Creating RGB composite...")
# RGB_IDX = (2, 1, 0)
# rgb = MSI[..., list(RGB_IDX)].astype(np.float32)

# for b in range(3):
#     band = rgb[..., b]
#     nz = band[band > 0]
#     if nz.size > 1000:
#         p1, p99 = np.percentile(nz, (1, 99))
#     else:
#         p1, p99 = np.percentile(band, (1, 99))
#     band = np.clip((band - p1) / (p99 - p1 + 1e-8), 0, 1)
#     rgb[..., b] = band

# rgb_u8 = (rgb * 255).astype(np.uint8)

# # ============ CREATE OUTPUT FOLDERS ============
# for cls in eurosat_classes:
#     os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

# # ============ EXTRACT AND CLASSIFY PATCHES ============
# print("Extracting patches and running CNN predictions...")
# PATCH, STRIDE = 64, 64
# idx = 0
# counts = {c: 0 for c in eurosat_classes}

# with torch.no_grad():  # No gradients needed for inference
#     for y in range(0, H - PATCH + 1, STRIDE):
#         if y % 320 == 0:  # Progress update every ~5 rows
#             print(f"Processing row {y}/{H-PATCH}...")
        
#         for x in range(0, W - PATCH + 1, STRIDE):
#             # Extract patch
#             tile = rgb_u8[y:y+PATCH, x:x+PATCH, :]
            
#             # Convert to PIL Image and apply transform
#             pil_img = Image.fromarray(tile, "RGB")
#             tensor_img = transform(pil_img).unsqueeze(0)  # Add batch dimension
#             tensor_img = tensor_img.to(device)
            
#             # Run prediction
#             output = model(tensor_img)
#             predicted_idx = output.argmax(1).item()
#             predicted_class = eurosat_classes[predicted_idx]
            
#             # Save image
#             filename = f"{predicted_class}_{idx:06d}.png"
#             filepath = os.path.join(OUTPUT_DIR, predicted_class, filename)
#             pil_img.save(filepath)
            
#             counts[predicted_class] += 1
#             idx += 1

# # ============ SUMMARY ============
# print("\n" + "="*50)
# print("PREDICTION COMPLETE!")
# print("="*50)
# print(f"Total patches extracted: {idx}")
# print("\nPatches per predicted class:")
# for cls, count in counts.items():
#     print(f"  {cls:22s} {count:5d}")
# print(f"\nOutput directory: {os.path.abspath(OUTPUT_DIR)}")

import os, numpy as np
from PIL import Image
import scipy.io as sio
import torch
from torchvision import transforms

# ============ CONFIGURATION ============
MAT_PATH = "/Users/ryanlutz/Desktop/berlin_multimodal.mat"
OUTPUT_DIR = "Berlin_RGB_Predicted"
MODEL_PATH = "Deepest_CNN/model_fin.pth"  # Path to your trained model

# Import your model architecture
from image_models import Deepest_CNN

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# EuroSAT classes (must match training order)
eurosat_classes = [
    "AnnualCrop","Forest","HerbaceousVegetation","Highway","Industrial",
    "Pasture","PermanentCrop","Residential","River","SeaLake"
]

# ============ LOAD MODEL ============
print("Loading trained model...")
model = Deepest_CNN(3, 10).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
print(f"Model loaded from {MODEL_PATH}")

# Option 2: Calculate from EuroSAT (safer - ensures consistency)
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

print("Calculating normalization from EuroSAT dataset...")
eurosat_dataset = ImageFolder(root='EuroSAT_RGB', transform=transforms.ToTensor())
eurosat_loader = DataLoader(eurosat_dataset, batch_size=32, shuffle=False, num_workers=0)

from image_normalization import get_mean_stdev
mean, stdev = get_mean_stdev(eurosat_loader)
print(f'Using mean: {mean}, stdev: {stdev}')

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=stdev)
])

# ============ LOAD BERLIN DATA ============
print("Loading Berlin dataset...")
m = sio.loadmat(MAT_PATH)
MSI = np.array(m["MSI"])
H, W, C = MSI.shape
print(f"MSI: {MSI.shape}, {MSI.dtype}")

# ============ CREATE RGB ============
print("Creating RGB composite...")
RGB_IDX = (2, 1, 0)
rgb = MSI[..., list(RGB_IDX)].astype(np.float32)

for b in range(3):
    band = rgb[..., b]
    nz = band[band > 0]
    if nz.size > 1000:
        p1, p99 = np.percentile(nz, (1, 99))
    else:
        p1, p99 = np.percentile(band, (1, 99))
    band = np.clip((band - p1) / (p99 - p1 + 1e-8), 0, 1)
    rgb[..., b] = band

rgb_u8 = (rgb * 255).astype(np.uint8)

# ============ CREATE OUTPUT FOLDERS ============
for cls in eurosat_classes:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

# ============ EXTRACT AND CLASSIFY PATCHES ============
print("Extracting patches and running CNN predictions...")
PATCH, STRIDE = 64, 64
idx = 0
counts = {c: 0 for c in eurosat_classes}

with torch.no_grad():  # No gradients needed for inference
    for y in range(0, H - PATCH + 1, STRIDE):
        if y % 320 == 0:  # Progress update every ~5 rows
            progress = (y / (H - PATCH + 1)) * 100
            print(f"Progress: {progress:.1f}% (row {y}/{H-PATCH})")
        
        for x in range(0, W - PATCH + 1, STRIDE):
            # Extract patch
            tile = rgb_u8[y:y+PATCH, x:x+PATCH, :]
            
            # Convert to PIL Image and apply transform
            pil_img = Image.fromarray(tile, "RGB")
            tensor_img = transform(pil_img).unsqueeze(0)  # Add batch dimension
            tensor_img = tensor_img.to(device)
            
            # Run prediction
            output = model(tensor_img)
            predicted_idx = output.argmax(1).item()
            predicted_class = eurosat_classes[predicted_idx]
            
            # Save image
            filename = f"{predicted_class}_{idx:06d}.png"
            filepath = os.path.join(OUTPUT_DIR, predicted_class, filename)
            pil_img.save(filepath)
            
            counts[predicted_class] += 1
            idx += 1

# ============ SUMMARY ============
print("\n" + "="*50)
print("PREDICTION COMPLETE!")
print("="*50)
print(f"Total patches extracted: {idx}")
print("\nPatches per predicted class:")
for cls, count in counts.items():
    percentage = (count / idx * 100) if idx > 0 else 0
    print(f"  {cls:22s} {count:5d} ({percentage:5.1f}%)")
print(f"\nOutput directory: {os.path.abspath(OUTPUT_DIR)}")

# ============ PLOT RESULTS ============
import matplotlib.pyplot as plt

print("\nGenerating histogram...")

# Prepare data for plotting
classes = list(counts.keys())
values = list(counts.values())
percentages = [(v / idx * 100) if idx > 0 else 0 for v in values]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Count histogram
bars1 = ax1.bar(range(len(classes)), values, color='steelblue', alpha=0.8, edgecolor='black')
ax1.set_xlabel('Land Cover Class', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Patches', fontsize=12, fontweight='bold')
ax1.set_title('Berlin Classification Results - Patch Counts', fontsize=14, fontweight='bold')
ax1.set_xticks(range(len(classes)))
ax1.set_xticklabels(classes, rotation=45, ha='right')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar, val in zip(bars1, values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(val)}',
             ha='center', va='bottom', fontsize=9)

# Plot 2: Percentage histogram
bars2 = ax2.bar(range(len(classes)), percentages, color='coral', alpha=0.8, edgecolor='black')
ax2.set_xlabel('Land Cover Class', fontsize=12, fontweight='bold')
ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax2.set_title('Berlin Classification Results - Percentages', fontsize=14, fontweight='bold')
ax2.set_xticks(range(len(classes)))
ax2.set_xticklabels(classes, rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(0, max(percentages) * 1.15)  # Add some headroom

# Add percentage labels on bars
for bar, pct in zip(bars2, percentages):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{pct:.1f}%',
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()

# Save the figure
plot_path = os.path.join(OUTPUT_DIR, 'classification_histogram.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Histogram saved to: {os.path.abspath(plot_path)}")

# Display the plot
plt.show()

print("\nDone!")
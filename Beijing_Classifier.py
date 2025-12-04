import os, numpy as np
from PIL import Image
import h5py  # Changed from scipy.io
import torch
from torchvision import transforms

# ============ CONFIGURATION ============
MAT_PATH = "/Users/ryanlutz/Desktop/beijing.mat"
OUTPUT_DIR = "Beijing_RGB_Predicted"
MODEL_PATH = "Deepest_CNN/model_fin.pth"  # Or Basic_CNN

from image_models import Deepest_CNN  # Change if needed
device = "cuda" if torch.cuda.is_available() else "cpu"

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

# ============ SETUP TRANSFORM ============
print("Calculating normalization from EuroSAT dataset...")
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

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

# ============ LOAD BEIJING DATA ============
print("Loading Beijing dataset...")
f = h5py.File(MAT_PATH, 'r')  # Changed to h5py
MSI = np.array(f["MSI"])  # Shape: (4, 8706, 13474)

# CRITICAL: Transpose from (C,H,W) to (H,W,C)
MSI = np.transpose(MSI, (1, 2, 0))  # Now: (8706, 13474, 4)
H, W, C = MSI.shape
print(f"MSI: {MSI.shape}, {MSI.dtype}")

f.close()  # Close the file

# ============ CREATE RGB ============
print("Creating RGB composite...")
RGB_IDX = (2, 1, 0)  # Try this first, adjust if colors look wrong
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
print(f"WARNING: Beijing is large ({H}x{W}). This will take a while!")
PATCH, STRIDE = 64, 64
idx = 0
counts = {c: 0 for c in eurosat_classes}

with torch.no_grad():
    for y in range(0, H - PATCH + 1, STRIDE):
        if y % 320 == 0:
            progress = (y / (H - PATCH + 1)) * 100
            print(f"Progress: {progress:.1f}% (row {y}/{H-PATCH})")
        
        for x in range(0, W - PATCH + 1, STRIDE):
            tile = rgb_u8[y:y+PATCH, x:x+PATCH, :]
            
            pil_img = Image.fromarray(tile, "RGB")
            tensor_img = transform(pil_img).unsqueeze(0)
            tensor_img = tensor_img.to(device)
            
            output = model(tensor_img)
            predicted_idx = output.argmax(1).item()
            predicted_class = eurosat_classes[predicted_idx]
            
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
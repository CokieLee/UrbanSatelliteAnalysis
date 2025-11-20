#!/usr/bin/env python3
import os
import glob
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import rasterio

# ============ CONFIGURATION ============
# Folder where your 2018 NYC orthoimagery .tif tiles live
ORTHO_DIR = "/Users/ryanlutz/Downloads/NYC_Data"  # <-- change this

OUTPUT_DIR = "NYC_2018_RGB_Predicted"
MODEL_PATH = "Deepest_CNN/model_fin.pth"
EUROSAT_ROOT = "EuroSAT_RGB"  # your existing EuroSAT RGB folder

from image_models import Deepest_CNN
from image_normalization import get_mean_stdev

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

# ============ SETUP TRANSFORM (EuroSAT normalization) ============
print("Calculating normalization from EuroSAT dataset...")
eurosat_dataset = ImageFolder(root=EUROSAT_ROOT, transform=transforms.ToTensor())
eurosat_loader = DataLoader(eurosat_dataset, batch_size=32, shuffle=False, num_workers=0)

mean, stdev = get_mean_stdev(eurosat_loader)
print(f"Using mean: {mean}, stdev: {stdev}")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=stdev)
])

# ============ CREATE OUTPUT FOLDERS ============
for cls in eurosat_classes:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

# ============ HELPER: CONTRAST STRETCH TO [0,1] ============
def contrast_stretch_rgb(rgb_float):
    """
    rgb_float: H x W x 3, float32, arbitrary range
    returns: H x W x 3, float32 in [0,1]
    """
    rgb = rgb_float.copy()
    for b in range(3):
        band = rgb[..., b]
        nz = band[band > 0]
        if nz.size > 1000:
            p1, p99 = np.percentile(nz, (1, 99))
        else:
            p1, p99 = np.percentile(band, (1, 99))
        denom = (p99 - p1) + 1e-8
        band = np.clip((band - p1) / denom, 0, 1)
        rgb[..., b] = band
    return rgb

# ============ PROCESS ALL ORTHO TILES ============
PATCH, STRIDE = 64, 64
global_counts = {c: 0 for c in eurosat_classes}
total_patches = 0

tif_paths = sorted(glob.glob(os.path.join(ORTHO_DIR, "*.jp2")))
print(f"Found {len(tif_paths)} JP2 tiles in {ORTHO_DIR}.")
if not tif_paths:
    raise SystemExit(f"No .jp2 files found in {ORTHO_DIR} — double-check ORTHO_DIR.")

print(f"Found {len(tif_paths)} GeoTIFF tiles to process.")

with torch.no_grad():
    for tile_idx, tif_path in enumerate(tif_paths):
        print(f"\nProcessing tile {tile_idx+1}/{len(tif_paths)}: {os.path.basename(tif_path)}")
        # --- LOAD RASTER WITH RASTERIO ---
        with rasterio.open(tif_path) as src:
            # Expect at least 3 bands (R,G,B). If more, ignore extra.
            if src.count < 3:
                print(f"  Skipping: {tif_path} has only {src.count} bands.")
                continue

            # Read first 3 bands: shape (bands, H, W)
            arr = src.read([1, 2, 3])  # 1-based indexing
            # Move to H x W x C
            arr = np.transpose(arr, (1, 2, 0)).astype(np.float32)
            H, W, C = arr.shape
            print(f"  Tile shape: {H} x {W} x {C}")

        # --- CONTRAST STRETCH TO [0,1] AND CONVERT TO 0-255 UINT8 ---
        rgb_01 = contrast_stretch_rgb(arr)
        rgb_u8 = (rgb_01 * 255).astype(np.uint8)

        # --- EXTRACT PATCHES AND CLASSIFY ---
        local_counts = {c: 0 for c in eurosat_classes}
        patch_idx = 0

        for y in range(0, H - PATCH + 1, STRIDE):
            # progress every ~5 tile heights
            if H > PATCH and y % max(PATCH * 5, 1) == 0:
                tile_progress = (y / (H - PATCH + 1)) * 100
                print(f"  Tile progress: {tile_progress:5.1f}% (row {y}/{H-PATCH})")

            for x in range(0, W - PATCH + 1, STRIDE):
                tile_rgb = rgb_u8[y:y+PATCH, x:x+PATCH, :]
                if tile_rgb.shape != (PATCH, PATCH, 3):
                    continue  # skip incomplete edge patches

                pil_img = Image.fromarray(tile_rgb, "RGB")
                tensor_img = transform(pil_img).unsqueeze(0).to(device)

                output = model(tensor_img)
                predicted_idx = output.argmax(1).item()
                predicted_class = eurosat_classes[predicted_idx]

                # Save patch to class folder
                filename = f"tile{tile_idx:03d}_{predicted_class}_{total_patches:06d}.png"
                filepath = os.path.join(OUTPUT_DIR, predicted_class, filename)
                pil_img.save(filepath)

                global_counts[predicted_class] += 1
                local_counts[predicted_class] += 1
                total_patches += 1
                patch_idx += 1

        print(f"  Finished tile {tile_idx+1}: {patch_idx} patches classified.")
        print("  Per-class counts for this tile:")
        for cls, cnt in local_counts.items():
            if cnt > 0:
                print(f"    {cls:22s} {cnt:6d}")

# ============ SUMMARY ============
print("\n" + "="*60)
print("NYC 2018 ORTHO PREDICTION COMPLETE!")
print("="*60)
print(f"Total patches extracted: {total_patches}")
print("\nPatches per predicted class (all tiles):")
for cls, count in global_counts.items():
    perc = (count / total_patches * 100) if total_patches > 0 else 0
    print(f"  {cls:22s} {count:7d} ({perc:5.1f}%)")
print(f"\nOutput directory: {os.path.abspath(OUTPUT_DIR)}")

import matplotlib.pyplot as plt

print("\nGenerating histogram...")

# Prepare data for plotting
classes = list(global_counts.keys())
values = list(global_counts.values())
percentages = [(v / total_patches * 100) if total_patches > 0 else 0 for v in values]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Count histogram
bars1 = ax1.bar(range(len(classes)), values, color='steelblue', alpha=0.8, edgecolor='black')
ax1.set_xlabel('Land Cover Class', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Patches', fontsize=12, fontweight='bold')
ax1.set_title('NYC 2018 Ortho Classification Results - Patch Counts', fontsize=14, fontweight='bold')
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
bars2 = ax2.bar(range(len(classes)), percentages, color='forestgreen', alpha=0.8, edgecolor='black')
ax2.set_xlabel('Land Cover Class', fontsize=12, fontweight='bold')
ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax2.set_title('NYC 2018 Ortho Classification Results - Percentages', fontsize=14, fontweight='bold')
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

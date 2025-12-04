import os, numpy as np
from PIL import Image
import scipy.io as sio

# --- 1) load ---
MAT_PATH = "/Users/ryanlutz/Desktop/berlin_multimodal.mat"   # TODO: set your path
m = sio.loadmat(MAT_PATH)
MSI = np.array(m["MSI"])         # (2465, 811, 4) uint16
LABEL = np.array(m["label"])     # (2465, 811)   uint8
H, W, C = MSI.shape
print("MSI:", MSI.shape, MSI.dtype, "LABEL:", LABEL.shape, LABEL.dtype)
print("Label IDs:", np.unique(LABEL))

# --- 2) pick RGB bands ---
# Try bands 0,1,2 (Python is 0-based). If colors look odd, try (2,1,0).
RGB_IDX = (2, 1, 0)  # TODO: change if needed
assert max(RGB_IDX) < C
rgb = MSI[..., list(RGB_IDX)].astype(np.float32)  # (H,W,3)

# # Per-band robust scaling to 0..255
# for b in range(3):
#     p1, p99 = np.percentile(rgb[..., b], (1, 99))
#     rgb[..., b] = np.clip((rgb[..., b] - p1) / (p99 - p1 + 1e-8), 0, 1)
# rgb_u8 = (rgb * 255).astype(np.uint8)

# Per-band robust scaling to 0..255 (ignore zeros)
for b in range(3):
    band = rgb[..., b].astype(np.float32)
    nz = band[band > 0]
    if nz.size > 1000:
        p1, p99 = np.percentile(nz, (1, 99))
    else:  # fallback if band is almost all zeros
        p1, p99 = np.percentile(band, (1, 99))
    band = np.clip((band - p1) / (p99 - p1 + 1e-8), 0, 1)
    rgb[..., b] = band
rgb_u8 = (rgb * 255).astype(np.uint8)


# --- 3) label mapping: Berlin ID -> EuroSAT class name ---
# TODO: fill this with the IDs you printed above that you can map sensibly.
label_to_eurosat = {
    0:"SeaLake", 1:"Residential", 2:"Highway", 3:"Industrial",
    4:"Highway", 5:"HerbaceousVegetation", 6:"Forest", 7:"Pasture",
    8:"AnnualCrop", 9:"River", 10:"Forest", 11:"Residential",
    12:"Industrial", 13:"SeaLake",
}
eurosat_classes = [
    "AnnualCrop","Forest","HerbaceousVegetation","Highway","Industrial",
    "Pasture","PermanentCrop","Residential","River","SeaLake"
]
for c in eurosat_classes:
    os.makedirs(os.path.join("Berlin_RGB", c), exist_ok=True)

# --- 4) extract pure 64x64 patches ---
PATCH, STRIDE, PURITY = 64, 64, 0.70
idx = 0
counts = {c:0 for c in eurosat_classes}

for y in range(0, H - PATCH + 1, STRIDE):
    for x in range(0, W - PATCH + 1, STRIDE):
        lab = LABEL[y:y+PATCH, x:x+PATCH]
        vals, cnts = np.unique(lab, return_counts=True)
        j = np.argmax(cnts); maj_id = int(vals[j]); frac = cnts[j] / (PATCH*PATCH)
        if frac < PURITY: 
            continue
        if maj_id not in label_to_eurosat:
            continue
        cls = label_to_eurosat[maj_id]
        if cls not in eurosat_classes:
            continue
        tile = rgb_u8[y:y+PATCH, x:x+PATCH, :]
        Image.fromarray(tile, "RGB").save(os.path.join("Berlin_RGB", cls, f"{cls}_{idx:06d}.png"))
        counts[cls] += 1; idx += 1

print("Saved patches per class:")
for k,v in counts.items(): print(f"{k:22s} {v}")
print("Output:", os.path.abspath("Berlin_RGB"))



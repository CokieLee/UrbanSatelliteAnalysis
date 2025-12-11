#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
from torch import nn
from torch import optim
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import rasterio
import numpy as np


# In[2]:


device = "cuda" if torch.cuda.is_available() else "cpu"

from image_models import Deepest_CNN
model = Deepest_CNN(3,10).to(device)
model.load_state_dict(torch.load("Trained_40epoch_Deepest.pth", weights_only=True))
model.eval()


# In[123]:


class_names = ['AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial',
               'Pasture','PermanentCrop','Residential','River','SeaLake']

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3443, 0.3801, 0.4085], std=[0.1522, 0.1190, 0.0860])  
])


# In[146]:


with rasterio.open("NY_B04_10m.jp2") as r, \
     rasterio.open("NY_B03_10m.jp2") as g, \
     rasterio.open("NY_B02_10m.jp2") as b:
    red = r.read(1); green = g.read(1); blue = b.read(1)

rgb = np.stack([red, green, blue], axis=0)  # [3,H,W]
H, W = rgb.shape[1], rgb.shape[2]
print(rgb.shape)


# In[148]:


#/3000
win = 64
stride = 32

num_h = (H - win) // stride + 1
num_w = (W - win) // stride + 1

pred_map = np.zeros((num_h, num_w), dtype=np.uint8)

for i in range(0, H - win + 1, stride):
    for j in range(0, W - win + 1, stride):

        patch = rgb[:, i:i+win, j:j+win].astype(np.float32)
        
        patch = np.clip(patch / 3000.0, 0, 1)  
        rgb_patch = np.transpose(patch, (1, 2, 0))  
        
        x = transform(rgb_patch).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x).argmax().item()

        pred_map[i//stride, j//stride] = pred

np.save("pred_map_NY32.npy", pred_map)
print("Done. pred_map shape:", pred_map.shape)


# In[149]:


x


# In[144]:


pred_map=np.load('pred_map_NC32.npy')

unique, counts = np.unique(pred_map, return_counts=True)

for value, count in zip(unique, counts):
    print(f"{value}: {class_names[value]} — {count}")


# In[145]:


import matplotlib.pyplot as plt
plt.imshow(pred_map)
plt.colorbar()
plt.show()


# In[142]:


pred_map1 = np.where(pred_map == 3, pred_map, 0)
H1, W1 = pred_map1.shape
top_right = pred_map1[H1//2:, :W1//2]   

plt.figure(figsize=(6, 6))
plt.imshow(top_right, cmap='viridis')
#plt.imshow(pred_map1, cmap='viridis')
#plt.colorbar(label='Class Index')
cbar = plt.colorbar(label="Class Index", shrink=0.8)
#plt.title("Where are the highways?",fontsize=16)
plt.show()


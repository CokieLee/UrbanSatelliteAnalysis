#!/usr/bin/env python
# coding: utf-8

# In[8]:


import torch
from torch import nn
from torch import optim
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import tifffile
import numpy as np
from image_normalization import get_mean_stdev


# In[9]:


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ### This is the section with variables to change when running different models

    # Import your model here after adding it to image_models.py 
    from image_models import Basic_CNN, Deep_CNN, Deepest_CNN
    model = Deepest_CNN(13,10).to(device)

    # for data loader:
    dl_batch_size = 32 # sort of hardware specifc
    dl_num_cores = 0 # hardware specific, change this to the number of cores on your cpu

    # Do we want to normalize the dataset based off of the per-pixel average and stdev?
    Do_Image_Normalization = True

    # image file paths
    image_dir = 'EuroSAT_MS'
    # image type (either set this to 'MS' or 'RGB')
    image_type = 'MS'

    # Training parameters
    num_epochs=40
    learnrate = 0.001
    save_interval = 1
model.load_state_dict(torch.load("MS_Deepest_40epoch_Trained.pth", weights_only=True))
model.eval()


# In[110]:


def tiff_loader(path):
    """
    Custom loader for 13-channel TIFF files.
    """
    img = tifffile.imread(path)

    # 确保通道在前 (C,H,W)
    if img.shape[0] != 13 and img.shape[-1] == 13:
        img = np.transpose(img, (2, 0, 1))

    img = img.astype(np.float32)
    
    if img.max() > 1:
        img /= 10000.0 

    return torch.from_numpy(img)


# In[111]:


def Get_Dataset(image_dir,transform,image_type):
    if image_type == 'MS':
        return ImageFolder(root=image_dir, transform=transform, loader=tiff_loader)
    elif image_type == 'RGB':
        return ImageFolder(root=image_dir, transform=transform)
    else:
        print("Set a valid image type")


# In[112]:


if Do_Image_Normalization:
    full_dataset = Get_Dataset(image_dir, transforms.Resize((64, 64)), image_type)

    from image_normalization import get_mean_stdev
    full_dataloader = DataLoader(
        full_dataset,
        batch_size=dl_batch_size,
        shuffle=True,
        num_workers=dl_num_cores,
        pin_memory=True
    )

    mean, stdev = get_mean_stdev(full_dataloader)
    print(f'mean: {mean}, stdev: {stdev}')

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Normalize(mean=mean, std=stdev)
    ])

else:
    transform = transforms.Compose([
        transforms.Resize((64, 64))
    ])


# In[113]:


full_dataset = Get_Dataset(image_dir,transform,image_type)


# In[114]:


indices = list(range(len(full_dataset)))
labels = np.array(full_dataset.targets)  # ImageFolder stores class labels here
class_names = full_dataset.classes

print(full_dataset.classes)
print(indices[0],indices[9000])
print(labels[0],labels[9000])


# In[115]:


train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=42  # for reproducibility
    )
temp_labels = labels[temp_idx]
val_idx, test_idx = train_test_split(temp_idx, 
                                    test_size = 0.5, 
                                    stratify = temp_labels, 
                                    random_state = 42)
train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
test_dataset = Subset(full_dataset, test_idx)


# In[116]:


test_loader = DataLoader(
        test_dataset,
        batch_size=dl_batch_size,
        shuffle=True,  # Shuffle for training
        num_workers=dl_num_cores,  # Parallel data loading (adjust based on CPU cores)
        pin_memory=True  # Faster data transfer to GPU (if using GPU)
    )


# In[117]:


import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import functional as TF
import tifffile
model.eval()

def get_classes(dset):
    if hasattr(dset, "classes"):
        return dset.classes
    if hasattr(dset, "dataset"):
        return get_classes(dset.dataset)
    return None

def resolve_base_and_index(dset, idx):
    cur, cur_idx = dset, idx
    while hasattr(cur, "dataset"):
        if hasattr(cur, "indices"):
            cur_idx = cur.indices[cur_idx]
        cur = cur.dataset
    return cur, cur_idx

def path_for_index(dset, idx):
    base, base_idx = resolve_base_and_index(dset, idx)
    if hasattr(base, "samples"):
        return base.samples[base_idx][0]
    if hasattr(base, "imgs"):
        return base.imgs[base_idx][0]
    return None

ds = test_loader.dataset
classes = get_classes(ds) or [str(i) for i in range(1000)]
idx_to_class = {i: c for i, c in enumerate(classes)}

n_show = 5
all_idx = list(range(len(ds)))
random.shuffle(all_idx)
pick_idx = all_idx[:n_show]

xs, ys, paths = [], [], []
for idx in pick_idx:
    x, y = ds[idx][0], int(ds[idx][1])
    p = path_for_index(ds, idx)
    xs.append(x.unsqueeze(0))
    ys.append(y)
    paths.append(p)

xb = torch.cat(xs, dim=0).to(device)
yb = torch.tensor(ys, device=device)
B, C, H_img, W_img = xb.shape  

features = []
def hook_fn(m, i, o):
    features.append(o.detach())
hook = model.bn4.register_forward_hook(hook_fn)

with torch.no_grad():
    logits = model(xb)
pred_idx = logits.argmax(dim=1)

A_pre = features.pop()
hook.remove()

A = model.pool(A_pre)                       # [B,C,Hf,Wf]
W_fc = model.fc1.weight.detach()            
C2, Hf, Wf = A.size(1), A.size(2), A.size(3)

fig, axes = plt.subplots(2, B, figsize=(2.2*B, 5))

for i in range(B):
    cls_pred = int(pred_idx[i])
    Wc = W_fc[cls_pred].view(C2, Hf, Wf)
    cam = (Wc * A[i]).sum(dim=0)
    cam = F.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_up = F.interpolate(
        cam[None, None],
        size=(int(H_img), int(W_img)),
        mode="bilinear",
        align_corners=False
    )[0, 0].cpu().numpy() 

    if paths[i] is not None and paths[i].endswith((".tiff", ".tif")):
        img_ms = tifffile.imread(paths[i])   # (13,H,W) or (H,W,13)
        if img_ms.shape[0] != 13 and img_ms.shape[-1] == 13:
            img_ms = np.transpose(img_ms, (2, 0, 1))
        img_ms = img_ms.astype(np.float32)
        if img_ms.max() > 0:
            img_ms /= img_ms.max()

        rgb_idx = [3, 2, 1]
        img_rgb = np.stack([img_ms[rgb_idx[0]], img_ms[rgb_idx[1]], img_ms[rgb_idx[2]]], axis=-1)
    else:
        x_cpu = xb[i].detach().cpu()
        x_cpu = (x_cpu - x_cpu.min()) / (x_cpu.max() - x_cpu.min() + 1e-8)
        if x_cpu.shape[0] >= 3:
            img_rgb = x_cpu[:3].permute(1, 2, 0).numpy()
        else:
            img_rgb = x_cpu.permute(1, 2, 0).numpy()

    file_name = paths[i].split("/")[-1] if paths[i] is not None else "unknown.tif"


    title = f"{file_name}"

    axes[0, i].imshow(img_rgb)
    axes[0, i].set_title(title, fontsize=9)  
    axes[0, i].axis("off")

    axes[1, i].imshow(img_rgb)
    axes[1, i].imshow(cam_up, alpha=0.45, cmap="jet")
    axes[1, i].axis("off")

plt.suptitle("Grad-CAM | 13-channel pseudo-RGB (top: composite image)")
plt.tight_layout()
plt.show()


# In[118]:


### same as last cell but have designated image id for comparison

import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import functional as TF
import tifffile
model.eval()

def get_classes(dset):
    if hasattr(dset, "classes"):
        return dset.classes
    if hasattr(dset, "dataset"):
        return get_classes(dset.dataset)
    return None

def resolve_base_and_index(dset, idx):
    cur, cur_idx = dset, idx
    while hasattr(cur, "dataset"):
        if hasattr(cur, "indices"):
            cur_idx = cur.indices[cur_idx]
        cur = cur.dataset
    return cur, cur_idx

def path_for_index(dset, idx):
    base, base_idx = resolve_base_and_index(dset, idx)
    if hasattr(base, "samples"):
        return base.samples[base_idx][0]
    if hasattr(base, "imgs"):
        return base.imgs[base_idx][0]
    return None

ds = test_loader.dataset
classes = get_classes(ds) or [str(i) for i in range(1000)]
idx_to_class = {i: c for i, c in enumerate(classes)}

target_names = [
    "PermanentCrop_911.jpg", "Industrial_1285","HerbaceousVegetation_234.jpg","Residential_1730","River_1817"
]

pick_idx = []
for i in range(len(ds)):
    p = path_for_index(ds, i)
    if p is not None and any(name in p for name in target_names):
        pick_idx.append(i)

xs, ys, paths = [], [], []
for idx in pick_idx:
    x, y = ds[idx][0], int(ds[idx][1])
    p = path_for_index(ds, idx)
    xs.append(x.unsqueeze(0))
    ys.append(y)
    paths.append(p)

xb = torch.cat(xs, dim=0).to(device)
yb = torch.tensor(ys, device=device)
B, C, H_img, W_img = xb.shape  

features = []
def hook_fn(m, i, o):
    features.append(o.detach())
hook = model.bn4.register_forward_hook(hook_fn)

with torch.no_grad():
    logits = model(xb)
pred_idx = logits.argmax(dim=1)

A_pre = features.pop()
hook.remove()

A = model.pool(A_pre)                       # [B,C,Hf,Wf]
W_fc = model.fc1.weight.detach()            
C2, Hf, Wf = A.size(1), A.size(2), A.size(3)

fig, axes = plt.subplots(2, B, figsize=(2.2*B, 5))
if B == 1:
    axes = np.array(axes).reshape(2, 1)

for i in range(B):
    cls_pred = int(pred_idx[i])
    Wc = W_fc[cls_pred].view(C2, Hf, Wf)
    cam = (Wc * A[i]).sum(dim=0)
    cam = F.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_up = F.interpolate(
        cam[None, None],
        size=(int(H_img), int(W_img)),
        mode="bilinear",
        align_corners=False
    )[0, 0].cpu().numpy() 

    if paths[i] is not None and paths[i].endswith((".tiff", ".tif")):
        img_ms = tifffile.imread(paths[i])   # (13,H,W) or (H,W,13)
        if img_ms.shape[0] != 13 and img_ms.shape[-1] == 13:
            img_ms = np.transpose(img_ms, (2, 0, 1))
        img_ms = img_ms.astype(np.float32)
        if img_ms.max() > 0:
            img_ms /= img_ms.max()

        rgb_idx = [3, 2, 1]
        img_rgb = np.stack([img_ms[rgb_idx[0]], img_ms[rgb_idx[1]], img_ms[rgb_idx[2]]], axis=-1)
    else:
        x_cpu = xb[i].detach().cpu()
        x_cpu = (x_cpu - x_cpu.min()) / (x_cpu.max() - x_cpu.min() + 1e-8)
        if x_cpu.shape[0] >= 3:
            img_rgb = x_cpu[:3].permute(1, 2, 0).numpy()
        else:
            img_rgb = x_cpu.permute(1, 2, 0).numpy()

    file_name = paths[i].split("/")[-1] if paths[i] is not None else "unknown.tif"

    title = f"{file_name}"

    axes[0, i].imshow(img_rgb)
    axes[0, i].set_title(title, fontsize=9)  # ✅ 多行显示
    axes[0, i].axis("off")

    axes[1, i].imshow(img_rgb)
    axes[1, i].imshow(cam_up, alpha=0.45, cmap="jet")
    axes[1, i].axis("off")

plt.suptitle("Grad-CAM | 13-channel pseudo-RGB (top: composite image)")
plt.tight_layout()
plt.show()


# In[109]:


test_dataset


# In[ ]:





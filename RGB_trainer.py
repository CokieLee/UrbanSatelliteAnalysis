import torch
from torch import nn
from torch import optim
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, Subset

from torchvision import transforms

from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"

### This is the section with variables to change when running different models

# Import your model here after adding it to image_models.py 
from image_models import Basic_CNN
model = Basic_CNN(3,10).to(device)

# for data loader:
dl_batch_size = 32 # sort of hardware specifc
dl_num_cores = 0 # hardware specific, change this to the number of cores on your cpu

# Do we want to normalize the dataset based off of the per-pixel average and stdev?
Do_Image_Normalization = True

# image file paths
image_dir = 'EuroSAT_RGB'

# Training parameters
num_epochs=12
learnrate = 0.001
save_interval = 1
saved_model_states = "Basic_CNN"

### End of modifiable variables

transform = transforms.ToTensor()

if Do_Image_Normalization:
    full_dataset = ImageFolder(root=image_dir, transform=transforms.ToTensor())
    #set up normalization from loader on all data
    from image_normalization import get_mean_stdev

    # Create DataLoader for normalization
    full_dataloader = DataLoader(
        full_dataset,
        batch_size=dl_batch_size,
        shuffle=True,  # Shuffle for training
        num_workers=dl_num_cores,  # Parallel data loading (adjust based on CPU cores)
        pin_memory=True  # Faster data transfer to GPU (if using GPU)
    )

    mean, stdev = get_mean_stdev(full_dataloader)
    print(f'mean: {mean}, stdev: {stdev}')

    transform = transforms.Compose(
        [transforms.Resize((64, 64)), # resize the images to ensure 64x64 pixels
        transforms.ToTensor(), #convert to a tensor
        transforms.Normalize(mean=mean, std=stdev)])

full_dataset = ImageFolder(root=image_dir, transform=transform)

import numpy as np
# Get all indices and their corresponding labels
indices = list(range(len(full_dataset)))
labels = np.array(full_dataset.targets)  # ImageFolder stores class labels here
class_names = full_dataset.classes

print(full_dataset.classes)
print(indices[0],indices[9000])
print(labels[0],labels[9000])

# Stratified split: 80% train, 10% val, 10% test
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

# create subsets
train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
test_dataset = Subset(full_dataset, test_idx)



# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=learnrate)

# Create DataLoader
dataloader = DataLoader(
    train_dataset,
    batch_size=dl_batch_size,
    shuffle=True,  # Shuffle for training
    num_workers=dl_num_cores,  # Parallel data loading (adjust based on CPU cores)
    pin_memory=True  # Faster data transfer to GPU (if using GPU)
)

def Do_Training(model,num_epochs,save_interval,save_dir):
        
    torch.save(model.state_dict(), os.path.join(save_dir, 'model_init.pth'))

    for epoch in range(num_epochs):
        # Iterate over training batches
        print(f"Epoch [{epoch + 1}/{num_epochs}]")

        for images, labels in dataloader:
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            scores = model(images)  # Shape: [batch_size, num_classes]
            loss = criterion(scores, labels)  # labels should be shape [batch_size]

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch) % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'epoch_{epoch+1}.pth'))
    torch.save(model.state_dict(), os.path.join(save_dir, 'model_fin.pth'))


import math
def Training_Is_Done(num_epochs,save_interval,saved_model_states):
    if os.path.isdir(saved_model_states):
        save_names = [os.path.basename(f) for f in os.listdir(saved_model_states)]
        if len(save_names) >= math.ceil(num_epochs / save_interval) + 2:
            return True
    else:
        os.mkdir(saved_model_states)
    return False


import os

# need to determine if the training has already happened, 
# otherwise we can just go straight to model analysis
model.train()
if not Training_Is_Done(num_epochs,save_interval,saved_model_states):
    Do_Training(model,num_epochs,save_interval,saved_model_states)

print("Training complete!!!")
model.eval()
# Lets evaluate the results

import polars as pl

test_loader = DataLoader(
    test_dataset,
    batch_size=dl_batch_size,
    shuffle=True,  # Shuffle for training
    num_workers=dl_num_cores,  # Parallel data loading (adjust based on CPU cores)
    pin_memory=True  # Faster data transfer to GPU (if using GPU)
)


save_names = [os.path.join(saved_model_states,f) for f in os.listdir(saved_model_states)]
results_df = pl.DataFrame({
    "Category" : class_names
})
for path in save_names:
    model.load_state_dict(torch.load(path, weights_only=True))

    correct_pred = {classname: 0 for classname in class_names}
    total_pred = {classname: 0 for classname in class_names}

    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[class_names[label]] += 1
                total_pred[class_names[label]] += 1
    # pl.DataFrame()
    # results_df.insert_column(os.path.basename(path),)
    print(f"Evaluating accuracy of model state: {os.path.basename(path)}...")
    model_accuracy = {}
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        model_accuracy[classname] = accuracy
        # print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    # print(f"model:{os.path.basename(path)}",model_accuracy)

    model_df = pl.DataFrame({
        "Category": list(model_accuracy.keys()),
        os.path.basename(path): list(model_accuracy.values())
    })
    results_df = results_df.join(model_df, on="Category", how="left")

results_df.write_csv(saved_model_states + '_accuracies.csv')

# ---- Evaluate trained EuroSAT model on Berlin_RGB (no retraining) ----
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

berlin_dir = "Berlin_RGB"  # folder you just created with patches
if os.path.isdir(berlin_dir):
    # Reuse the SAME transform you used for EuroSAT (resize + EuroSAT mean/std)
    berlin_dataset = ImageFolder(root=berlin_dir, transform=transform)
    berlin_loader = DataLoader(
        berlin_dataset,
        batch_size=dl_batch_size,
        shuffle=False,
        num_workers=dl_num_cores,   # set to 0 on macOS if you hit worker errors
        pin_memory=(device == "cuda"),
    )

    # Pick a checkpoint to test (final is fine)
    ckpt_path = os.path.join(saved_model_states, "model_fin.pth")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in berlin_loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu()
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    print(f"[Berlin] accuracy @ {os.path.basename(ckpt_path)}: {100.0*correct/total:.2f}% ({correct}/{total})")
else:
    print("Berlin_RGB/ not found. Run your converter to create it.")


# ---- Grab images the model predicted as a specific class (e.g., "Pasture") ----
import shutil
import torch

target_class = "Pasture"  # <- change if you want another class later
assert target_class in berlin_dataset.class_to_idx, f"{target_class} not in {berlin_dataset.classes}"
target_idx = berlin_dataset.class_to_idx[target_class]

pred_out_dir = f"berlin_pred_{target_class}"
pred_mis_dir = f"berlin_pred_{target_class}_mis"  # predicted as target but actually another class
os.makedirs(pred_out_dir, exist_ok=True)
os.makedirs(pred_mis_dir, exist_ok=True)

# list of file paths in dataset order
all_paths = [p for p,_ in berlin_dataset.samples]

saved_pred = 0
saved_mis  = 0
max_save   = 200  # cap how many we copy to keep things light

idx0 = 0
model.eval()
with torch.no_grad():
    for images, labels in berlin_loader:
        bpaths = all_paths[idx0: idx0 + images.size(0)]
        idx0  += images.size(0)

        images = images.to(device)
        probs  = model(images)                 # your model already outputs softmax
        pred_i = probs.argmax(1).cpu()
        pred_p = probs.max(1).values.cpu()

        for path, pi, pp, true_i in zip(bpaths, pred_i.numpy(), pred_p.numpy(), labels.numpy()):
            if pi == target_idx and saved_pred < max_save:
                base = os.path.basename(path)
                shutil.copy2(path, os.path.join(pred_out_dir, f"{pp:.2f}__{base}"))
                saved_pred += 1
                if true_i != target_idx and saved_mis < max_save:
                    # also keep a sample of *wrong* "Pasture" predictions
                    shutil.copy2(path, os.path.join(pred_mis_dir, f"{berlin_dataset.classes[true_i]}__{pp:.2f}__{base}"))
                    saved_mis += 1

print(f"Saved {saved_pred} images predicted as {target_class} to: {pred_out_dir}/")
print(f"Saved {saved_mis} misclassified '{target_class}' preds to: {pred_mis_dir}/")

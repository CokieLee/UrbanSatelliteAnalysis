import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torchvision import transforms

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"

# Gather all image file paths
image_dir = 'EuroSAT_RGB'
all_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
all_paths = [s for s in all_paths if "MNIST" not in s]
print(all_paths)

all_files = []
labels = []

for path in all_paths:
  all_files.extend([os.path.join(path, f) for f in os.listdir(path)])
  index = path.find('EuroSAT_RGB/')
  if index != -1:
      start_index = index + len('EuroSAT_RGB/')
      result = path[start_index:]
  else:
      result = "" # Target string not found
  labels.extend([result] * len(os.listdir(path)))
 

# Get labels (assuming class names are part of the file paths)
labels_names = [os.path.basename(os.path.dirname(f)) for f in all_files]
print(len(labels)) 

labels_unique = np.unique(labels_names)

# print(labels_unique)
# label_enum = [x for x in range(len(labels_unique))]
print()

label_numeric = []
for i in range(len(labels)):
    label_numeric.append(np.where(labels_unique == labels[i])[0][0])
   
print("labels:",labels[0],labels[9000])
print("labels numeric:",label_numeric[0],label_numeric[9000])

# Split the file paths, using stratify to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    all_files, label_numeric, test_size=0.2, random_state=42, stratify=labels
) 

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):

        """
        Building blocks of convolutional neural network.

        Parameters:
            * in_channels: Number of channels in the input image (for grayscale images, 1)
            * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
        """
        super(CNN, self).__init__()

        # 1st convolutional layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2nd convolutional layer
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1)
        # Fully connected layer
        self.fc1 = nn.Linear(16 * 16, num_classes)


    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: Input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.fc1(x)            # Apply fully connected layer
        x = torch.nn.functional.softmax(x,dim=1) # apply softmax to x
        # x = torch.nn.functional.max()
        return x

model = CNN(in_channels=3, num_classes=10).to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, label_tensor=None, transform=None):
        """
        Args:
            image_paths: List of file paths to images.
            labels: List of corresponding labels (class indices, not one-hot).
            label_tensor: Optional tensor mapping for labels (if needed).
            transform: PyTorch transforms to apply to images.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.label_tensor = label_tensor
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')  # Adjust mode if needed (e.g., 'L' for grayscale)
        
        # Load label
        label = self.labels[idx]
        if self.label_tensor is not None:
            label = self.label_tensor[label]
        else:
            label = torch.tensor(label, dtype=torch.long)  # Ensure label is a tensor of class index

        # Apply transformations to image
        if self.transform:
            image = self.transform(image)

        return image, label

# Create Dataset
dataset = CustomImageDataset(
    image_paths=X_train,
    labels=y_train,  # Should be class indices (e.g., [7, 2, 5, ...])
    transform=transforms.ToTensor()
)

# Create DataLoader
batch_size = 32  # Adjust based on hardware
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,  # Shuffle for training
    num_workers=4,  # Parallel data loading (adjust based on CPU cores)
    pin_memory=True  # Faster data transfer to GPU (if using GPU)
)

num_epochs=10
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
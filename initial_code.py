# Our primary dataset:
# 
# Eurosat
# 
# [DOI](10.1109/IGARSS.2018.8519248)
# 
# [dataset](https://zenodo.org/records/7711810#.ZAm3k-zMKEA)
# 
# 
# download EuroSAT_MS.zip (the full spectral dataset) and extract it into this directory
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import torch.nn.functional as F

import torchvision
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

# import torchmetrics

import os
from PIL import Image
from sklearn.model_selection import train_test_split


x = torch.rand(5, 3)
print(x)
 
# Gather all image file paths
image_dir = 'EuroSAT_RGB'
os.path.abspath()
os.path.dirname()
all_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
all_paths = all_paths[1:]
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
import numpy as np

labels_unique = np.unique(labels_names)

# print(labels_unique)
# label_enum = [x for x in range(len(labels_unique))]
print()

label_tensor = []
label_index = []
for i in range(len(labels)):
   vect_init = np.zeros((1,len(labels_unique)))
   vect_init[0][np.where(labels_unique == labels[i])[0]] = 1
#    labels
#    print(np.where(labels_unique == labels[i])[0])
   label_tensor.append(vect_init)
   label_index.append(i)
   
print("labels:",labels[0],labels[9000])
print("label index:",label_index[0],label_index[9000])
print("label tensor:",label_tensor[0],label_tensor[9000])
# labels = [label_enum[labels_unique.index(str_label)] for str_label in labels_unique]


# Split the file paths, using stratify to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    all_files, label_index, test_size=0.2, random_state=42, stratify=labels
) 
print("X:",X_train[0]) 
print(len(X_train)) 
print(len(X_test)) 
print("y:",y_train[0]) 
print(len(y_train)) 
print(len(y_test)) 

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
       return x



# batch_size = 60

# train_dataset = datasets.MNIST(root="EuroSAT_RGB/", download=True, train=True, transform=transforms.ToTensor())

# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# test_dataset = datasets.MNIST(root="EuroSAT_RGB/", download=True, train=False, transform=transforms.ToTensor())

# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN(in_channels=3, num_classes=10).to(device)
print(model)
# >>> CNN(
#   (conv1): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (conv2): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (fc1): Linear(in_features=784, out_features=10, bias=True)
# )


# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

torch.backends.nnpack.enabled = False

num_epochs=10
for epoch in range(num_epochs):
  # Iterate over training batches
  print(f"Epoch [{epoch + 1}/{num_epochs}]")

  for image,li in zip(X_train,y_train):
    image = Image.open(image)
    image = ToTensor()(image)
    label = ToTensor()(label_tensor[li])
    # print(image)
    # print(label)
    # data = image.to(device)
    # targets = label.to(device)
    scores = model(image)
    # print("scores:",scores)
    # print("label:",label)
    loss = criterion(scores, label[0])
    optimizer.zero_grad()
    loss.backward()
  optimizer.step()


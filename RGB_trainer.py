import torch
from torch import nn
from torch import optim
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, Subset

from torchvision import transforms

from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"

# Gather all image file paths
image_dir = 'EuroSAT_RGB'

# Load the full dataset
full_dataset = ImageFolder(root=image_dir, transform=transforms.ToTensor())

# Get all indices and their corresponding labels
indices = list(range(len(full_dataset)))
labels = full_dataset.targets  # ImageFolder stores class labels here
class_names = full_dataset.classes

print(full_dataset.classes)
print(indices[0],indices[9000])
print(labels[0],labels[9000])

# Stratified split: 80% train, 20% test
train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=labels,
    random_state=42  # for reproducibility
)

# create subsets
train_dataset = Subset(full_dataset, train_idx)
test_dataset = Subset(full_dataset, test_idx)

from image_models import Basic_CNN

model = Basic_CNN(3,10).to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create DataLoader
batch_size = 32  # Adjust based on hardware
dataloader = DataLoader(
    train_dataset,
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

torch.save(model.state_dict(), 'RGB_model_state.pth')
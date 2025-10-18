import torch
from torch import nn
import torch.nn.functional as F

class Basic_CNN(nn.Module):
    def __init__(self, in_channels, num_classes):

        """
        Building blocks of convolutional neural network.

        Parameters:
            * in_channels: Number of channels in the input image (for grayscale images, 1)
            * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
        """
        super(Basic_CNN, self).__init__()

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

class Deep_CNN(nn.Module):
    def __init__(self, in_channels, num_classes):

        """
        Building blocks of convolutional neural network.

        Parameters:
            * in_channels: Number of channels in the input image (for grayscale images, 1)
            * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
        """
        super(Deep_CNN, self).__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

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
        x = F.relu(self.conv1(x))  
        x = self.pool(x)           
        x = F.relu(self.conv2(x))  
        x = self.pool(x)           
        x = F.relu(self.conv3(x))  
        x = self.pool(x)           
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.fc1(x)            # Apply fully connected layer
        x = torch.nn.functional.softmax(x,dim=1) # apply softmax to x
        # x = torch.nn.functional.max()
        return x

class Deep_CNN_Dropout(nn.Module):
    def __init__(self, in_channels, num_classes):

        """
        Building blocks of convolutional neural network.

        Parameters:
            * in_channels: Number of channels in the input image (for grayscale images, 1)
            * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
        """
        super(Deep_CNN_Dropout, self).__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer
        self.fc1 = nn.Linear(16 * 16, num_classes)

        # Add Dropout
        self.dropout = nn.Dropout(p=0.5)  # 50% dropout rate


    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: Input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x = F.relu(self.conv1(x))  
        x = self.pool(x)           
        x = F.relu(self.conv2(x))  
        x = self.pool(x)           
        x = F.relu(self.conv3(x))  
        x = self.pool(x)           
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.dropout(x)  # Apply dropout before fully connected layer
        x = self.fc1(x)            # Apply fully connected layer
        x = torch.nn.functional.softmax(x,dim=1) # apply softmax to x
        # x = torch.nn.functional.max()
        return x


class Deep_CNN_Dropout_BatchNorm(nn.Module):
    def __init__(self, in_channels, num_classes):

        """
        Building blocks of convolutional neural network.

        Parameters:
            * in_channels: Number of channels in the input image (for grayscale images, 1)
            * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
        """
        super(Deep_CNN_Dropout_BatchNorm, self).__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(8)  # BatchNorm for conv1 output
        self.bn2 = nn.BatchNorm2d(8)  # BatchNorm for conv2 output
        self.bn3 = nn.BatchNorm2d(4)  # BatchNorm for conv3 output
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer
        self.fc1 = nn.Linear(16 * 16, num_classes)

        # Add Dropout
        self.dropout = nn.Dropout(p=0.5)  # 50% dropout rate


    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: Input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.dropout(x)  # Apply dropout before fully connected layer
        x = self.fc1(x)            # Apply fully connected layer
        x = torch.nn.functional.softmax(x,dim=1) # apply softmax to x
        # x = torch.nn.functional.max()
        return x


class Deep_CNN_Dropout_BatchNorm_No_Softmax(nn.Module):
    def __init__(self, in_channels, num_classes):

        """
        Building blocks of convolutional neural network.

        Parameters:
            * in_channels: Number of channels in the input image (for grayscale images, 1)
            * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
        """
        super(Deep_CNN_Dropout_BatchNorm_No_Softmax, self).__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(8)  # BatchNorm for conv1 output
        self.bn2 = nn.BatchNorm2d(8)  # BatchNorm for conv2 output
        self.bn3 = nn.BatchNorm2d(4)  # BatchNorm for conv3 output
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer
        self.fc1 = nn.Linear(16 * 16, num_classes)

        # Add Dropout
        self.dropout = nn.Dropout(p=0.5)  # 50% dropout rate


    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: Input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.dropout(x)  # Apply dropout before fully connected layer
        x = self.fc1(x)            # Apply fully connected layer
        return x

class Deepest_CNN(nn.Module):
    def __init__(self, in_channels, num_classes):

        """
        This model has 2 more layers compared to the original model

        This model also includes batch dropout, batch normalization, 
        and doesn't apply softmax to the final output.

        It seems to perform quite well after 12 epochs.

        It'll be interesting to see how it compares to some other peoples models! 

        Parameters:
            * in_channels: Number of channels in the input image (for grayscale images, 1)
            * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
        """
        super(Deepest_CNN, self).__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(8)  # BatchNorm for conv1 output
        self.bn2 = nn.BatchNorm2d(8)  # BatchNorm for conv2 output
        self.bn3 = nn.BatchNorm2d(16)  # BatchNorm for conv3 output
        self.bn4 = nn.BatchNorm2d(16)  # BatchNorm for conv4 output
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer
        self.fc1 = nn.Linear(16 * 16, num_classes)

        # Add Dropout
        self.dropout = nn.Dropout(p=0.5)  # 50% dropout rate


    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: Input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.dropout(x)  # Apply dropout before fully connected layer
        x = self.fc1(x)            # Apply fully connected layer
        return x

class transformer_CNN(nn.Module):
    def __init__(self):
        super(transformer_CNN, self).__init__()

        # output 32 convolutional features with square kernel of size 3
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)

        # Designed to ensure that adjacent pixels are either all 0s or all active with an input probability
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # Maxpool 2D takes in (Cin, Hin, Win), outputs (C, Hout, Wout). If do 2, 
        #stride=2, then will divide both H, W dimensions by 2
        self.fc1 = nn.Linear(64 * 32 * 32, 1024)
        
        # to fit to the labels
        self.fc2 = nn.Linear(1024, 10)

        #transformer layer
        self.transformer = nn.TransformerEncoderLayer(d_model = 64, nhead = 8, dim_feedforward=256)

    # x represents our data    
    def forward(self, x):

        # F is from the torch.nn.Functional library (don't require parameters)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        # Run max pooling over x
        x = F.max_pool2d(x,2)

        # prevent overfitting
        x = self.dropout1(x)

        B, C, H, W = x.shape
        x = x.view(B, C, H*W)       #resize so all pixels are concatenated
        x = x.permute(2,0,1)        #H*W, B, C
        x = self.transformer(x)       # apply transformer layer
        x = x.permute(1, 2, 0)       # [B, C, H*W]
        x = x.view(B, C, H, W)      # [B, 64, 32, 32]

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)

        #soft-max to the labels
        output = F.log_softmax(x, dim=1)
        return output

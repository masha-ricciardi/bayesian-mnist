"""
Here I'm creating a Convolutional Neural Network (CNN) class that inherits from nn.Module, which is the base class for all neural network modules in PyTorch. 
This class will define the architecture of the CNN model that will serve as the basis for all our MNIST classification models.

"""
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()  # Instantiates nn.Module, our base case.
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)    # Builds 1st convolutional layer, 1 channel from incoming image, 16 pattern detectors, 5x5 size of each pattern detector
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)   # Builds 2nd convolutional layer, 16 channels from previous layer, 32 pattern detectors, 5x5 size of each pattern detector
        self.fc1 = nn.Linear(32 * 4 * 4, 128)            # Builds 1st fully connected layer, 32 * 4 * 4 input features, 128 output features
        self.fc2 = nn.Linear(128, 10)                   # Builds 2nd fully connected layer, 128 input features, 10 output features

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # Applies 1st convolutional layer, ReLU activation, and max pooling
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # Applies 2nd convolutional layer, ReLU activation, and max pooling
        x = x.view(x.size(0), -1)   # Flattens the output of the convolutional layers, size(0) keeps the batch size intact, -1 makes PyTorch infer the correct second dimension based on the size of the first.
        x = F.relu(self.fc1(x))  # Applies 1st fully connected layer and ReLU activation    
        x = self.fc2(x)  # Applies 2nd fully connected layer
        return x  # Returns the output of the model

# Not using softmax activation function in the output layer because PyTorch's CrossEntropyLoss function combines softmax and negative log-likelihood loss in a single function, which is more numerically stable than applying softmax separately.
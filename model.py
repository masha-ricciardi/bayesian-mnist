"""
Here I'm creating a Convolutional Neural Network (CNN) class that inherits from nn.Module, which is the base class for all neural network modules in PyTorch. 
This class will define the architecture of the CNN model that will serve as the basis for all our MNIST classification models.

First two layers are convolutional layers that will learn to detect patterns in the input images. 
They do not make assumptions about position of certain features in the image e.g. a curve corresponding to a nine can be detected anywhere in the image.
The last two layers are fully connected (linear) layers that do not preserve position, which means they can work with the output of the convolutional layers to make predictions about the class of the input image.

Using only linear layers could theoretically work, but it is computationally expensive and would require a lot of training data to learn the same patterns that convolutional layers can learn with fewer parameters and less data.

Here are the 3 steps each of the two convolutional layers will perform on the input images:

Convolutional layers: 
They take a small patch of the input image and apply a filter (a small matrix of weights) to it, producing a single output value. #
This process is repeated across the entire image, producing a feature map that highlights the presence of certain patterns in the image. The filters are learned during training, allowing the model to learn which patterns are important for classification.
E.g. we have a 3x3 patch of the image that looks like this:

4 1 0  which it matches to this filter 1 1 0
2 3 1                                  0 1 0
0 1 2                                  0 0 1

Each value in the patch is multiplied by the corresponding value in the filter, and the results are summed to produce a single output value (in our example it would be 10). 
This process is repeated across the entire image, producing a feature map that highlights the presence of certain patterns in the image. 
The filters are learned during training, allowing the model to learn which patterns are important for the classification.

ReLU (Rectified Linear Unit):
An activation function that introduces non-linearity into the model, allowing it to learn more complex patterns.
It works by setting all negative numbers to zero and leaving positive numbers unchanged (e.g. if our 10 were -4 it would become 0, in our instance it is unchanged).
This is important because without non-linearity, the model would only be able to learn linear relationships between the input and output, which would limit its ability to learn complex patterns in the data.
Intuitively, "this shape looks like the opposite of what I'm looking for" is not any more useful than "it doesn't look like it".

Max Pooling:
Groups the outputs of the convolutional layers into small non-overlapping patches and keeps the largest value from each patch.
This reduces the dimensions and makes the model more robust to small translations and distortions in the input images.

Since the numbers sit in the centre of the image, we can use a 2x2 max pooling layer to reduce the dimensions of the feature maps by half, while still preserving the important features of the input images.
This works for MNIST specifically but might not be applicable to future datasets.
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
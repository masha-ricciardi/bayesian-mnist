"This script is used to load the MNIST dataset and create data loaders for training and testing."

from torchvision import datasets, transforms

# Converting data to tensor of 3-Dimensions (channels, height, width)
transform = transforms.ToTensor()

# Training data downloaded from torchvision.datasets.MNIST
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)

# Test data
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

# Displaying the number of images in training and test datasets
# The 'if' condition below ensures that the code block is only executed when this script is run directly, and not when it is imported as a module in another script.
# Avoids the number of training and test images being printed when this script is imported elsewhere.
if __name__ == "__main__":
    print(f"train: {len(train_data)} images")
    print(f"test: {len(test_data)} images")
    # Output here should be:
    # train: 60000 images
    # test: 10000 images

    image, label = train_data[0]
    print(f"one image: shape={image.shape}, label={label}")
    # Output here should be:
    # one image: shape=torch.Size([1, 28, 28]), label=5


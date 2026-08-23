<h1>MNIST Classification Neaural Network</h1>

This project aims to first, build a standard point-estimate convolutional neural network and then compare it against two Bayesian approaches: MC-Dropout and Bayes-by-Backpropagation.

The MNIST digit classification dataset is used for this, it consists of 60000 training images and 10000 testing images of hand drawn digits 0-9, all labelled with their true value.

It follows the concepts and architecture guidance from
[*Hands-on Bayesian Neural Networks — A Tutorial for Deep Learning Users*](https://arxiv.org/abs/2007.06823)
(Jospin et al., 2022).

<h2> Progress </h2>

- [x] Data Loading
- [x] Basic point-estimate CNN 
- [ ] MC-Dropout Version
- [ ] Bayes-by-Backpropagation version
- [ ] Comparison

<h2> Architecture </h2>

Same baseline CNN will be used across three models. This includes the use of a manual random seed to ensure results are reproducible and allow for more accurate comparisons across the three models.

The structure is as follows:

Input (1×28×28)
→ Conv2d(1→16, 5×5) → ReLU → MaxPool(2)
→ Conv2d(16→32, 5×5) → ReLU → MaxPool(2)
→ Flatten (512)
→ Linear(512→128) → ReLU
→ Linear(128→10)

First two layers are convolutional layers that will learn to detect patterns in the input images. They do not make assumptions about position of certain features in the image e.g. a curve corresponding to a 9 can be detected anywhere in the image.

The last two layers are fully connected (linear) layers that do not preserve position, which means they can work with the output of the convolutional layers to make predictions about the class of the input image.

Using only linear layers could theoretically work, but it is far more computationally expensive and would require a lot of training data to learn the same patterns that convolutional layers can learn with fewer parameters and less data.

Here are the 3 steps each of the two convolutional layers will perform on the input images:

**1. Convolutional layers:**
They take a small patch of the input image and apply a filter (a small matrix of weights) to it, producing a single output value. 

This process is repeated across the entire image, producing a feature map that highlights the presence of certain patterns in the image. The filters are learned during training, allowing the model to learn which patterns are important for classification.

E.g. we have a 3x3 patch of the image that looks like this:
```
4 1 0  which it matches to this filter 1 1 0
2 3 1                                  0 1 0
0 1 2                                  0 0 1
```
Each value in the patch is multiplied by the corresponding value in the filter, and the results are summed to produce a single output value (in our example it would be 10). This process is repeated across the entire image, producing a feature map that highlights the presence of certain patterns in the image. The filters are learned during training, allowing the model to learn which patterns are important for the classification.

**2. ReLU (Rectified Linear Unit):**
An activation function that introduces non-linearity into the model, allowing it to learn more complex patterns.
It works by setting all negative numbers to zero and leaving positive numbers unchanged (e.g. if our 10 were -4 it would become 0, in our instance it is unchanged).

This is important because without non-linearity, the model would only be able to learn linear relationships between the input and output, which would limit its ability to learn complex patterns in the data.

Intuitively, "this shape looks like the opposite of what I'm looking for" is not any more useful than "it doesn't look like it".

**3. Max Pooling:**
Groups the outputs of the convolutional layers into small non-overlapping patches and keeps the largest value from each patch. This reduces the dimensions and makes the model more robust to small translations and distortions in the input images.

Since the numbers sit in the centre of the image, we can use a 2x2 max pooling layer to reduce the dimensions of the feature maps by half, while still preserving the important features of the input images. This works for MNIST specifically but might not be applicable to future datasets.

<h2> Setup and Usage </h2>

**Set up**
Requires Python 3.14 (or similar) and a virtual environment.

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1        
pip install -r requirements.txt
```
**Usage**

```bash
python data.py    # Loads the MNIST
python model.py    # Builds the baseline of the CNN
python train.py    # trains the baseline CNN for 5 epochs
```
<h2> Results (so far) </h2>

With a run time that averages at around 1 minute, these are the results for the basic CNN over 5 epochs (runs of the dataset):

- Epoch: 0  Train accuracy: 93.14%  Test accuracy: 98.10%
- Epoch: 1  Train accuracy: 98.21%  Test accuracy: 98.37%
- Epoch: 2  Train accuracy: 98.77%  Test accuracy: 98.52%
- Epoch: 3  Train accuracy: 98.99%  Test accuracy: 99.07%
- Epoch: 4  Train accuracy: 99.16%  Test accuracy: 99.06%

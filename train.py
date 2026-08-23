"""
This script is used to load the MNIST dataset and create data loaders for training and testing.
The data loaders will group the images into batches and shuffle the training data.
This is setting up the data for a Mini Batch Stochastic Gradient Descent approach.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import train_data, test_data
from model import CNN
import time

start_time = time.time()  # Records the start time of the training process.

#Variables for the training process
epochs = 5 #Number of times the training dataset will be passed through the model.
train_losses = []
train_correct = []
test_losses = []
test_correct = []

torch.manual_seed(42) # Setting a random seed to ensure starting weights are reproducible. 
# This will help later when comparing the performance of different models.

# Using dataloader to group the train images into batches of 100 and shuffles the order to avoid overfitting and the model picking up the order of the images every epoch.
# This is not necessary for the test data.

train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = DataLoader(test_data, batch_size=100, shuffle=False)

if __name__ == "__main__":
    images, labels = next(iter(train_loader)) 
# This line retrieves the first batch of images and labels from the training data loader. 
# The 'next' function is used to get the next item from the iterator returned by 'iter(train_loader)'.
    print("one training batch:", images.shape, labels.shape)
    print("number of training batches:", len(train_loader))
    print("number of test batches:", len(test_loader))

# Output here should be:
#one training batch: torch.Size([100, 1, 28, 28]) torch.Size([100])
#number of training batches: 600
#number of test batches: 100

model = CNN()
criterion = nn.CrossEntropyLoss()  # combines softmax and cross entropy loss into one function
optimizer = optim.Adam(model.parameters(), lr=1e-3) 
# Adam optimizer is the most commonly used optimizer for training deep learning models.
# It combines the advantages of Adaptive Gradient Algorithm (AdaGrad) and Root Mean Square Propagation (RMSProp).
# It keeps a running memory of the behaviour of a weight's gradient in the last few updates and it's consistency/noisiness


for i in range(epochs):
    trn_correct = 0
    tst_correct = 0
    tst_loss_total = 0
    # Training the model
    for b,(X_train, y_train) in enumerate(train_loader):
        b+=1 # Starts batches at 1
        y_pred = model(X_train) # Gets prediced values from the training set, this is 2D since our 1st convolutional layer is looking for two dimensions
        loss = criterion(y_pred, y_train) # Comparing predictions to correct answers
        predicted = torch.max(y_pred.data, 1)[1] # add up the number of correct predictions, indexing from the first point
        batch_correct = (predicted == y_train).sum() # How many are correct from this batch. True = 1, False = 0
        trn_correct += batch_correct # Keeping track as training goes along

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Results
        if b%600 == 0:
            print(f'Epoch: {i} Batch: {b} Loss: {loss.item()}')

    train_losses.append(loss)
    train_correct.append(trn_correct)

    # Test
    with torch.no_grad(): # No gradient so we don't update our weights and biases with test data
        for b,(X_test, y_test) in enumerate(test_loader):
            y_val = model(X_test)
            predicted = torch.max(y_val.data, 1)[1] # Adding up correct predictions
            tst_correct += (predicted == y_test).sum() # True = 1, False = 0 and sum them up
            tst_loss_total += criterion(y_val, y_test).item()

    losses = tst_loss_total / len(test_loader)
    test_losses.append(losses)
    test_correct.append(tst_correct)
    train_acc = trn_correct.item() / len(train_data) * 100
    test_acc = tst_correct.item() / len(test_data) * 100
    print(f'Epoch: {i}  Train accuracy: {train_acc:.2f}%  Test accuracy: {test_acc:.2f}%')

current_time = time.time()  # Records the current time to calculate elapsed time during training.
total_time = current_time - start_time  # Calculates the total time taken for the training process.
print(f"Total time taken for training: {total_time/60:.2f} minutes")  # Total time in minutes to 2 decimal places.
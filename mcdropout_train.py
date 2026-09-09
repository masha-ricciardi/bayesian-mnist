"""
This script trains and tests the MCDropout version of our model. 
The dropout is active during training but switched off in the testing portion of every epoch.
What changed from the CNN model:
 - Added weight decay to the optimizer
 - Made sure the dropout only occurs during training
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import train_data, test_data
from mcdropout_model import MCDropoutCNN
import time

start_time = time.time()  # Records the start time of the training process.

#Variables for the training process
epochs = 5 #Number of times the training dataset will be passed through the model.
train_losses = []
train_correct = []
test_losses = []
test_correct = []

torch.manual_seed(42) 

train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = DataLoader(test_data, batch_size=100, shuffle=False)

if __name__ == "__main__":
    images, labels = next(iter(train_loader)) 
    print("one training batch:", images.shape, labels.shape)
    print("number of training batches:", len(train_loader))
    print("number of test batches:", len(test_loader))

# Output here should be:
#one training batch: torch.Size([100, 1, 28, 28]) torch.Size([100])
#number of training batches: 600
#number of test batches: 100

model = MCDropoutCNN()
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# Added a weight_decay which shrinks the weights slightly every time the optimizer updates
# This is mathematically equivalent to having a Gaussian prior, since we're penalising large weights with the assumption they come from a bell curve.


for i in range(epochs):
    trn_correct = 0
    tst_correct = 0
    tst_loss_total = 0

    # Training the model
    for b,(X_train, y_train) in enumerate(train_loader):
        b+=1 
        y_pred = model(X_train) 
        loss = criterion(y_pred, y_train)
        predicted = torch.max(y_pred.data, 1)[1] 
        batch_correct = (predicted == y_train).sum() 
        trn_correct += batch_correct 

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
    model.eval()    # Turns off dropout for the testing
    with torch.no_grad(): 
        for b,(X_test, y_test) in enumerate(test_loader):
            y_val = model(X_test)
            predicted = torch.max(y_val.data, 1)[1] 
            tst_correct += (predicted == y_test).sum() # True = 1, False = 0 and sum them up
            tst_loss_total += criterion(y_val, y_test).item()
    model.train()   # Turns dropout back in before the next epoch's training

    losses = tst_loss_total / len(test_loader)
    test_losses.append(losses)
    test_correct.append(tst_correct)
    train_acc = trn_correct.item() / len(train_data) * 100
    test_acc = tst_correct.item() / len(test_data) * 100
    print(f'Epoch: {i}  Train accuracy: {train_acc:.2f}%  Test accuracy: {test_acc:.2f}%')

current_time = time.time() 
total_time = current_time - start_time  
torch.save(model.state_dict(), 'mcdropout_weights.pth') 
print(f"Total time taken for training: {total_time/60:.2f} minutes") 



# Expected output:
"""
one training batch: torch.Size([100, 1, 28, 28]) torch.Size([100])
number of training batches: 600
number of test batches: 100
Epoch: 0 Batch: 600 Loss: 0.12527333199977875
Epoch: 0  Train accuracy: 90.59%  Test accuracy: 97.89%
Epoch: 1 Batch: 600 Loss: 0.16651277244091034
Epoch: 1  Train accuracy: 97.01%  Test accuracy: 98.66%
Epoch: 2 Batch: 600 Loss: 0.03300691023468971
Epoch: 2  Train accuracy: 97.71%  Test accuracy: 98.80%
Epoch: 3 Batch: 600 Loss: 0.04862077534198761
Epoch: 3  Train accuracy: 98.03%  Test accuracy: 99.00%
Epoch: 4 Batch: 600 Loss: 0.027153126895427704
Epoch: 4  Train accuracy: 98.31%  Test accuracy: 99.07%
Total time taken for training: 1.34 minutes
"""
# The train accuracy here is lower since the dropout is online active during the training phase.

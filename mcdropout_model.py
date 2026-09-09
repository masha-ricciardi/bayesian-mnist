"This is the MC-Dropout version of the CNN built in model.py. "
"""
Per the paper: dropout defines a variational posterior over the weights
    of each layer (zi,j ~ Bernoulli(pi), Wi = Mi · diag(zi) -- Eq. 41).
    This is NOT the same role dropout normally plays (regularization) --
    here it's standing in for a distribution we're sampling from, which is
    why a separate regularizer (L2 / weight_decay) is still needed alongside it.
    
"""
import torch.nn as nn
import torch.nn.functional as F

class MCDropoutCNN(nn.Module):
    def __init__(self, dropout_p=0.25):  # dropout-p=0.25 is the probability of dropping a given neuron (this value is typically 0.2-0.5)
        super().__init__()  
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)    
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)   
        self.fc1 = nn.Linear(32 * 4 * 4, 128)            
        self.fc2 = nn.Linear(128, 10)                   
        self.dropout = nn.Dropout(p=dropout_p)          # Randomly zeros out 25% of what passes through it

    def forward(self, x):
        x = F.relu(self.conv1(x))   # 1st 3 layers same as before, just spaced out for clarity
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)         # Here we use the self.dropout defined above

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)
        return x 


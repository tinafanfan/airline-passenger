import pandas as pd
import numpy as np
import sklearn

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from dataset import Pessengers
from models import LSTM

num_epochs = 10000
learning_rate = 0.01

input_size = 1
hidden_size = 2
num_layers = 1
num_classes = 1


# config a dataloader
custom_dataset = Pessengers('train', obs_length=4, train_pctg=0.7)
# create a dataloader
data_loader = DataLoader(custom_dataset, batch_size=10, shuffle=False)

# model
lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
# loss function
criterion = torch.nn.MSELoss()
# optimizer
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):

    for i, (X,Y) in enumerate(data_loader):
        outputs = lstm(X)
        optimizer.zero_grad()
        
        # obtain the loss function
        loss = criterion(outputs, Y)
        
        loss.backward()
        
        optimizer.step()
    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
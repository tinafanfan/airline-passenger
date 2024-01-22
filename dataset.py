import pandas as pd
import numpy as np
import sklearn
import matplotlib 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset



def sliding_windows(data, obs_length):
    '''
    goal: splitting the time series into numerous observations with each length = obs_length
    data: a 1-d numpy.ndarray with dimension (len, 1)
    obs_length: the length of observation

    output: 2 numpy.ndarray, predictor and response, with sahpe = (num, obs_length, 1)
    '''
    x = []
    y = []

    for i in range(len(data) - obs_length - 1):
        _x = data[i:(i+obs_length)]
        _y = data[i+obs_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

def normalize_minmax(data):
    '''
    data: a time series
    output: normalize using MinMaxScaler from sklearn.preprocessing
    '''
    nmlz = MinMaxScaler()
    return nmlz.fit_transform(data)

def split_into_subset(X, Y, subset, train_pctg):
    train_size = int(len(Y) * train_pctg)
    test_size = len(Y) - train_size

    if subset == 'train':
        sample = Variable(torch.Tensor(X[0:train_size]))
        target = Variable(torch.Tensor(Y[0:train_size]))            
    elif subset == 'test':
        sample = Variable(torch.Tensor(X[train_size:len(X)]))
        target = Variable(torch.Tensor(Y[train_size:len(Y)]))
    else:
        sample = Variable(torch.Tensor(X))
        target = Variable(torch.Tensor(Y))

    return sample, target



class Pessengers(Dataset):
    def __init__(self, subset, obs_length, train_pctg):

        self.subset = subset
        self.obs_length = obs_length
        self.data = normalize_minmax(pd.read_csv("airline-passengers.csv").iloc[:,1:2].values)
        self.X, self.Y = sliding_windows(self.data, self.obs_length)
        self.sample, self.target = split_into_subset(self.X, self.Y, subset, train_pctg)

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, index):
        return self.sample[index], self.target[index]    
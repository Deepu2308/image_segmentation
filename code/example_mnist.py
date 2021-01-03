# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 18:59:58 2021

@author: deepu
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from upsampling_pytorch import UpsampleTrainer
from torch import optim
from torch import nn
import torch
from random import sample
from copy import deepcopy
import numpy as np

data = load_digits()

X = data['images']
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y.reshape((-1,1)),
                                                    stratify = y)

max_, min_  = X_train.max(), X_train.min()

X_train = torch.Tensor( (X_train - min_)/ max_)
X_train.requires_grad  = False
X_test  = torch.Tensor( (X_test - min_) / max_)
X_test.requires_grad  = False

# =============================================================================
#     create network
# =============================================================================

#setup network instance
net = UpsampleTrainer(n_channels = 20, bias = True)
LR = .01
optimizer = optim.SGD(net.parameters(),
                      lr = LR)
criterion = nn.BCEWithLogitsLoss()

#freeze training for conv layer
net.conv.weight.requires_grad = False

    
        
# =============================================================================
#     train transposedConv layer
# =============================================================================
    
#show layer weights
#print("\nConvolution Weights\n" , net.conv.weight, sep = '')
#print("\nTransposed Convolution Weights\n" ,net.transposedConv.weight, sep = '')

conv_w = deepcopy(net.conv)    
trans_conv_w = deepcopy(net.transposedConv)

for epoch in range(10000):
    
    #set netwrok to train
    net.train()
    
    #create batch using random numbers
    batch_size = 100
    ind = sample( list(range(len(X_train))), batch_size)
    batch = torch.tensor( X_train[ind].reshape(batch_size,
                                               1,
                                               8,
                                               8),
                         requires_grad=False).float()
    
    #reset optimizer gradients to zero
    optimizer.zero_grad()
    
    #get output        
    output = net(batch)
    
    #calculate loss (target is same as input)
    loss = criterion(output, batch)
    
    #use gradients to optimize parameters
    loss.backward()        
    optimizer.step()

    if (epoch +1) % 100 == 0:
        print("Epoch :{} \t Loss {:.2f}".format(epoch + 1, loss.item()))
        
        from matplotlib import pyplot as plt
        
        i = np.random.randint(1,batch_size-1)
        inp = batch[i-1:i].detach().numpy().reshape(8,8)
        out = net(batch[i-1:i]).detach().numpy().reshape(8,8)
        plt.imshow(inp,cmap='gray')
        plt.title(y_train[ind][i][0])
        plt.imshow(out,cmap='gray')

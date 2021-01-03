# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 18:59:58 2021

@author: deepu
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from upsampling_pytorch import UpsampleTrainer, get_cm
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
net = UpsampleTrainer(n_channels = 20, 
                      bias = True,
                      n_classification = 10) #10 digits
LR = .001
optimizer = optim.SGD(net.parameters(),
                      lr = LR)

#head 1
criterion1 = nn.CrossEntropyLoss()

#head 2
criterion2 = nn.BCEWithLogitsLoss()

    
# =============================================================================
# train convoutional layer
# =============================================================================
    
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
    
    #digit
    digits = y_train[ind] 
    
    #reset optimizer gradients to zero
    optimizer.zero_grad()
    
    #get output        
    output = net(batch)
    
    #calculate loss (target is same as input)
    loss = criterion1(output[1], torch.Tensor(digits.ravel())\
                      .type(torch.LongTensor) )
    
    #use gradients to optimize parameters
    loss.backward()        
    optimizer.step()    

            
    # print epoch level statistics
    if (epoch +1) % 100 == 0:
        net.eval()
        with torch.no_grad():
            
           #get predictions
            predictions = net(torch.tensor( X_test.reshape(-1,
                                               1,
                                               8,
                                               8),
                         requires_grad=False).float())[1]
            
            #compute losses
            test_loss  = criterion1(predictions,torch.Tensor(y_test.ravel())\
                      .type(torch.LongTensor) )
            
            
            #create confusion matrix
            cm      = get_cm( torch.Tensor(digits.ravel())\
                      .type(torch.LongTensor),
                      output[1] , 10
                      )
            test_cm = get_cm( torch.Tensor(y_test.ravel())\
                      .type(torch.LongTensor),
                      predictions , 10
                      )
            
            #display message every t steps and log every 10t steps
            msg = "Epoch : {} loss : {:.2f} test_loss: {:.2f} \t acc: {:.2f} test_acc: {:.2f}".format(epoch,
                   loss.item(),
                   test_loss.item(),
                   cm.diag().sum() / cm.sum(),
                   test_cm.diag().sum() / test_cm.sum()
                   )
            
            print(msg)            
            
            
            
            
# =============================================================================
#     train only transposedConv layer
# =============================================================================
    

#freeze training for conv layer
net.conv.weight.requires_grad = False

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
    
    #digit
    digits = y_train[ind] 
    
    #reset optimizer gradients to zero
    optimizer.zero_grad()
    
    #get output        
    output = net(batch)
    
    #calculate loss (target is same as input)
    loss = criterion2(output[0], batch)
    
    #use gradients to optimize parameters
    loss.backward()        
    optimizer.step()

    if (epoch +1) % 100 == 0:
        print("Epoch :{} \t Loss {:.2f}".format(epoch + 1, loss.item()))
                
        if (epoch +1) % 1000 == 0:
            from matplotlib import pyplot as plt
            
            i = np.random.randint(1,batch_size-1)
            inp = batch[i-1:i].detach().numpy().reshape(8,8)
            out = net(batch[i-1:i])[0].detach().numpy().reshape(8,8)
            
            fig, ax = plt.subplots(1,2)
            fig.suptitle( "True Label : " + str( digits[i-1:i][0][0]) +\
                         "\titeration: " + epoch+1)
            ax[0].imshow(inp,cmap='gray')
            ax[0].set_title("Original")
            ax[1].imshow(out,cmap='gray')
            ax[1].set_title("Generated")

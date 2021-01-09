# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 08:59:52 2020

@author: deepu
"""

import torch
from torch import nn
from torch.functional import F
from torch import optim
from copy import deepcopy


def get_cm(actual, predictions,nb_classes = 6):
        
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        
            _, preds = torch.max(predictions, 1)
            for t, p in zip(actual.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
    
    return confusion_matrix.long()

class UpsampleTrainer(nn.Module):
    
    def __init__(self, 
                 n_channels = 2,
                 bias = False,
                 n_classification = None,
                 conv_out_size = 20*7*7):
        super(UpsampleTrainer, self).__init__()
        
        
        
        self.n_classification = n_classification  #number of classes
        if n_classification:          
            self.conv_out_size = conv_out_size
            self.fc1 = nn.Linear( self.conv_out_size, self.n_classification)
            
        
        
        self.conv = nn.Conv2d(
                    in_channels = 1,
                    out_channels = n_channels,
                    kernel_size = (2,2),
                    stride = 1,
                    padding = 0,
                    dilation =  1,
                    groups = 1,
                    bias = bias,
                    padding_mode = 'zeros'
        )
        
        self.transposedConv = nn.ConvTranspose2d(
                    in_channels = n_channels,
                    out_channels = 1,
                    kernel_size = (2,2),
                    stride = 1,
                    padding = 0,
                    dilation =  1,
                    groups = 1,
                    bias = bias,
                    padding_mode = 'zeros'
        )   
        
    def forward(self,x):
        
        
        x = self.conv(x)
        x1 = self.transposedConv(x)
        
        if self.n_classification:
            x2 = F.relu(x.view(-1, self.conv_out_size))
            x2 = F.dropout(x2, training=self.training)
            x2 = self.fc1(x2)     
            
            return x1,x2
        
        return x1
        
        
class UpsampleOutConvTrainer(nn.Module):
    
    def __init__(self, 
                 n_channels = 2,
                 bias = False,
                 n_classification = None,
                 conv_out_size = 20*7*7):
        super(UpsampleOutConvTrainer, self).__init__()
        
        
        
        self.n_classification = n_classification  #number of classes
        if n_classification:          
            self.conv_out_size = conv_out_size
            self.fc1 = nn.Linear( self.conv_out_size, self.n_classification)
            
        
        
        self.conv = nn.Conv2d(
                    in_channels = 1,
                    out_channels = n_channels,
                    kernel_size = (2,2),
                    stride = 1,
                    padding = 0,
                    dilation =  1,
                    groups = 1,
                    bias = bias,
                    padding_mode = 'zeros'
        )
        
        self.transposedConv = nn.ConvTranspose2d(
                    in_channels = n_channels,
                    out_channels = 5,
                    kernel_size = (2,2),
                    stride = 1,
                    padding = 0,
                    dilation =  1,
                    groups = 1,
                    bias = bias,
                    padding_mode = 'zeros'
        )   
        
        self.OutConv = nn.Conv2d(
                    in_channels = 5,
                    out_channels = 1,
                    kernel_size = (3,3),
                    stride = 1,
                    padding = 1,
                    groups = 1,
                    bias = True,
                    padding_mode = 'zeros'
        )
        
    def forward(self,x):
        
        
        
        x = self.conv(x)
        x1 = self.OutConv(F.relu(self.transposedConv(x)))
        
        if self.n_classification:
            x2 = F.relu(x.view(-1, self.conv_out_size))
            x2 = F.dropout(x2, training=self.training)
            x2 = self.fc1(x2)     
            
            return x1,x2
        
        return x1        
        
        
if __name__ == '__main__':
    
# =============================================================================
#     create network
# =============================================================================

    #setup network instance
    net = UpsampleTrainer()
    optimizer = optim.SGD(net.parameters(),
                          lr = .01)
    criterion = nn.MSELoss()
    
    #freeze training for conv layer
    net.conv.weight.requires_grad = False
    
# =============================================================================
#     perform checks 
# =============================================================================
    with torch.no_grad():       
        
        #create random tensor
        m = torch.randn(1,1,4,4)
        
        #create out of conv
        out_conv  = net.conv(m)
        
        #assert (net.conv.weight * m[0,0,:2,:2]).sum().item() == \
        #    out_conv[0,0,0,0].item(), "sample calculation failed"
            
        #create out of transposed conv
        out_tran_conv =   net.transposedConv(out_conv)
        
        assert m.shape == out_tran_conv.shape, "OutSize doesnt match insize"
    
        out = net.forward(m)
        assert out.shape == m.shape, "Input output shape mismatch"

        loss = criterion(m, out)
        assert loss.item() == ((m - out) * (m - out)).mean().item()
    
# =============================================================================
#     train transposedConv layer
# =============================================================================
    
    #show layer weights
    #print("\nConvolution Weights\n" , net.conv.weight, sep = '')
    #print("\nTransposed Convolution Weights\n" ,net.transposedConv.weight, sep = '')
    
    conv_w = deepcopy(net.conv)    
    trans_conv_w = deepcopy(net.transposedConv)
    
    for epoch in range(100000):
        
        #set netwrok to train
        net.train()
        
        #create batch using random numbers
        batch = torch.randn(32,1,1,1)
        batch =  torch.cat([batch,2*batch,3*batch],3)
        batch =  torch.cat([batch,2*batch,3*batch],2)
        batch[:,0,0,0] = 0
        batch[:,0,-1,-1] = 0
        
        #reset optimizer gradients to zero
        optimizer.zero_grad()
        
        #get output        
        output = net(batch)
        
        #calculate loss (target is same as input)
        loss = criterion(output, batch)
        
        #use gradients to optimize parameters
        loss.backward()        
        optimizer.step()
    
        if (epoch +1) % 1000 == 0:
            print("Epoch :{} \t Loss {:.2f}".format(epoch + 1, loss.item()))
    
    #show layer weights
    print("\nConvolution Old Weights\n" , conv_w.weight, sep = '')
    print("\nConvolution New Weights\n" , net.conv.weight, sep = '')
    print("\nTransposed Convolution Old Weights\n" ,trans_conv_w.weight, sep = '')
    print("\nTransposed Convolution New Weights\n" ,net.transposedConv.weight, sep = '')
    
    
    i  = batch[-1:]
    print(i)
    print(net(i))
    print(net(i)-i)
    
    
    
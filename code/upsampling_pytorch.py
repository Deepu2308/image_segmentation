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
class UpsampleTrainer(nn.Module):
    
    def __init__(self, 
                 n_channels = 2,
                 bias = False):
        super(UpsampleTrainer, self).__init__()
        
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
        x = self.transposedConv(x)
        
        return x
        
        
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
    
    
    
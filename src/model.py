# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 20:00:30 2021

@author: deepu
"""

import torch.nn as nn
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split

import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import pandas as pd 
import logging
import src.dataset as dataset

logging.basicConfig(filename='src/logs/nn_log.log',
                    filemode='a', 
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)


n_epochs = 1500
batch_size_train = 8
batch_size_test  = 4
learning_rate = 0.001
momentum = 0.5

random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)


model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
    classes=2,                      # model output channels (number of classes in your dataset)
    encoder_depth = 3,
    decoder_channels = [64, 32, 16]
)
    

model.cuda()

gpu_available = "GPU available?:      " + str(torch.cuda.is_available())
using_cuda    = "Network using cuda?: " + str(next(model.parameters()).is_cuda)

print(gpu_available)
print(using_cuda)
logging.info("\n\n------------------------------------------------------")
logging.info(gpu_available)
logging.info(using_cuda)

logging.info(\
f"""\n\nNetwork Details:
n_epochs         = {n_epochs}
batch_size_train = {batch_size_train}
batch_size_test  = {batch_size_test}
learning_rate    = {learning_rate}
momentum         = {momentum}    
\n\n"""
)

optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=momentum)
criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    
    
    #read train metadata
    df_train = pd.read_csv(r"input/stage_1_train_images.csv")    
    train,validation = train_test_split(df_train,
                                        stratify = df_train.loc[:,'has_pneumo']
                                        )
    
# =============================================================================
#     train prep
# =============================================================================
    #create train loader
    train_dataset = dataset.DataLoaderSegmentation(
        train.new_filename.to_list()
    )
    
    #create train batch generator
    train_loader  = dataset.data.DataLoader(train_dataset, 
                               batch_size= batch_size_train,
                               shuffle=True,
                               num_workers=0)
    train_gen     = iter(train_loader)

# =============================================================================
#     val prep
# =============================================================================
    #create val loader
    val_dataset = dataset.DataLoaderSegmentation(
        validation.new_filename.to_list()
    )

    #create val batch generator
    val_loader  = dataset.data.DataLoader(val_dataset, 
                               batch_size= batch_size_test,
                               shuffle=True,
                               num_workers=0)
    val_gen     = iter(val_loader)

    #print every 't' steps
    t = 5
    
    
    for epoch in range(n_epochs):  # loop over the dataset multiple times
    
        model.train() 
        train_loss = 0.0
        
        for i_batch, batch in enumerate(train_loader):
        
            # get the inputs and labels
            inputs, labels = batch            
        
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            break
    
        # print epoch level statistics
        if epoch % t == 0:
            network.eval()
            with torch.no_grad():
                
                #get test batch
                try:
                        test_batch      = next(test_gen)
                except StopIteration:
                        test_gen        = iter(test_loader)                
                        test_batch      = next(test_gen)                                
                
                #get predictions
                predictions = network(test_batch['inputs'])
                
                #compute losses
                test_loss  = criterion(predictions, test_batch['labels'])
                train_loss /= (i_batch+1)
                
                #create confusion matrix
                cm      = get_cm(labels, outputs)
                test_cm = get_cm(test_batch['labels'], predictions)
                
                #display message every t steps and log every 10t steps
                msg = "Epoch : {} loss : {:.2f} test_loss: {:.2f} \t acc: {:.2f} test_acc: {:.2f}".format(epoch,
                       train_loss,
                       test_loss.item(),
                       cm.diag().sum() / cm.sum(),
                       test_cm.diag().sum() / test_cm.sum()
                       )
                
                print(msg)
                if epoch % (t*20) == 0: logging.info(msg)
    
    #log last available results
    logging.info(msg)
    
    #log performance on full test set
    network.eval()
    full_test_cm    = get_cm(test_dataset.y, network(test_dataset.X))   
    full_test_loss  = criterion(predictions, test_batch['labels']).item()
    msg             = 'Finished Training. Test Accuracy : {:.2f} Mean Loss : {:.2f}'.format( 
                      (full_test_cm.diag().sum() / full_test_cm.sum()).item(),
                      full_test_loss)
    print(msg)
    logging.info(msg)
    
    
    #save model
    if True:
        torch.save({
                'epoch': epoch,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
    
        },
        r"src/saved_models/model2.pkl")
    
        #load example
        checkpoint = torch.load(r"src/saved_models/model2.pkl")
        network.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    
    #analyze results
    activity_labels =pd.read_csv("input/LabelMap.csv", index_col=0)
    test_df = pd.DataFrame(full_test_cm.long().numpy(), 
                          columns = activity_labels.Activity,
                          index = activity_labels.Activity)
    test_df.to_csv('src/ConfusionMatrixTest.csv')
    
    
    
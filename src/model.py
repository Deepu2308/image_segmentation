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

import matplotlib.pyplot as plt
import seaborn as sns

quickhist = lambda x: sns.histplot(x.ravel(), log_scale=(False,True))


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()




n_epochs = 40
batch_size_train = 8
batch_size_test  = 1
learning_rate = 0.001

random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)


model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
    encoder_depth = 3,
    decoder_channels = [32, 16, 8],
    activation= 'sigmoid'
)
    

model.cuda()

# =============================================================================
# check if GPU is available and log model details
# =============================================================================
gpu_available = "GPU available?:      " + str(torch.cuda.is_available())
using_cuda    = "Network using cuda?: " + str(next(model.parameters()).is_cuda)

print(gpu_available)
print(using_cuda)

logging.basicConfig(filename='src/logs/nn_log.log',
                    filemode='a', 
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logging.info("\n\n------------------------------------------------------")
logging.info(gpu_available)
logging.info(using_cuda)

logging.info(\
f"""\n\nNetwork Details:
n_epochs         = {n_epochs}
batch_size_train = {batch_size_train}
batch_size_test  = {batch_size_test}
learning_rate    = {learning_rate}
\n\n"""
)

    
# =============================================================================
# read train metadata
# =============================================================================
df_train = pd.read_csv(r"input/stage_1_train_images.csv")    
train,validation = train_test_split(df_train,
                                    stratify = df_train.loc[:,'has_pneumo']
                                    )

# =============================================================================
#train prep
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
#val prep
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


# =============================================================================
# model loss and optimizer
# =============================================================================
    
loss = smp.utils.losses.DiceLoss()

metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=learning_rate),
])


# =============================================================================
# create epoch runners 
# =============================================================================
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device='cuda',
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device='cuda',
    verbose=True,
)


# =============================================================================
# start training
# =============================================================================
max_score = 0
for i in range(0, n_epochs):
    
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(val_loader)
    
    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')
        
    if i == 10:
        optimizer.param_groups[0]['lr'] *= 1e-1
        print('Decrease decoder learning rate to 10% of current!')
        
        
if __name__ == '__main__':
    
    batch = next(iter(train_loader))
    model.eval()    
    out = model(batch[0]).cpu().detach().numpy()
    quickhist(out)
    quickhist(batch[1].cpu().detach().numpy())
    
    for i in range(5):
        n = np.random.choice(len(val_dataset))
        
        image_vis = val_dataset[n][0]#.astype('uint8')
        image, gt_mask = val_dataset[n]
        
        gt_mask = gt_mask.squeeze()
        
        x_tensor = image.unsqueeze(0)
        pr_mask = model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            
        visualize(
            image=image_vis.cpu().reshape(pr_mask.shape), 
            ground_truth_mask=gt_mask.cpu(), 
            predicted_mask=pr_mask
    )
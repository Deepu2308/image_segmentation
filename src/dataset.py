# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 19:27:35 2021

@author: deepu
"""

import torch
import os
import numpy as np

import torch.utils.data as data
from PIL import Image

class DataLoaderSegmentation(data.Dataset):
    """
    Class to provide data for image segmentation
    refer https://discuss.pytorch.org/t/dataloader-for-semantic-segmentation/48290
    """
    
    def __init__(self, 
                 file_list,
                 resize_shape = (256,256)
                 ):
        super(DataLoaderSegmentation, self).__init__()
        
        #resize shape
        self.resize_shape = resize_shape
        
        #save path variable
        self.path = os.getcwd() + '/input/'
        
        #get filenames
        self.img_files  = [self.path + r'/png_images/' + i \
                           for i in file_list]
        
    def mask_map(self, x):
            #function to get mask name
            return  x.replace('images', 'masks') 
        
    def __getitem__(self, index):
        
            #get filenames for batch
            img_path  = self.img_files[index]
            mask_path = self.mask_map(img_path)
            
            #open images
            image  = Image.open(img_path).resize(self.resize_shape, Image.BOX)
            mask   = Image.open(mask_path).resize(self.resize_shape, Image.BOX) #filter choice needs imprvement
           
                
            #convert to RGB and then numpy array
            #image,mask = image.convert('RGB'),mask.convert('RGB')

            #add channel dimension
            image,mask = np.expand_dims(np.array(image),0),\
                                np.expand_dims(np.array(mask),0)
            mask[mask>0] = 1
             
            #reorder axes
            #image,mask =  self.reorder_axes(image), self.reorder_axes()
            
            #return 
            return torch.from_numpy(image).float().cuda(), torch.from_numpy(mask).float().cuda()

    def __len__(self):
        return len(self.img_files)

#class simple:
#    def __init__(self, simple = 'asdasd'):
#        self.simple = simple

if __name__ == '__main__'    :
    
    
    file_list = os.listdir(r"C:\Users\deepu\Desktop\MyFolders\Projects\PetProjects\image_segmentation/input/png_images") 
    
    #sample loader
    ld = DataLoaderSegmentation(file_list)

    #get sample
    img, msk = ld.__getitem__(0)

    print(img.shape)

    import dill
    import pickle
    s = dill.dumps(ld)
    f = pickle.dumps(ld)

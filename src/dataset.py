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
    
    def __init__(self, folder_path):
        super(DataLoaderSegmentation, self).__init__()
        
        #save path variable
        self.path = folder_path
        
        #get filenames
        self.img_files  = os.listdir(self.path + '/png_images')    
        self.img_files  = [self.path + r'/png_images/' + i \
                           for i in self.img_files]
        
        #function to get mask name
        self.mask_map  = lambda x:  x.replace('images', 'masks') 
        
            
    def __getitem__(self, index):
        
            #get filenames for batch
            img_path  = self.img_files[index]
            mask_path = self.mask_map(img_path)
            
            #open images
            image  = Image.open(img_path)
            mask   = Image.open(mask_path)
            
            #convert to RGB and then numpy array
            image,mask = np.array(image.convert('RGB')),\
                                np.array(mask.convert('RGB'))
            
            #return 
            return torch.from_numpy(image).float().cuda(), torch.from_numpy(mask).float().cuda()

    def __len__(self):
        return len(self.img_files)
    
if __name__ == '__main__'    :
    
    #sample loader
    ld = DataLoaderSegmentation(r"C:\Users\deepu\Desktop\MyFolders\Projects\PetProjects\image_segmentation\input")

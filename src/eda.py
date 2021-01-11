# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 18:26:03 2021

@author: deepu
"""

from os import listdir
l = listdir(r"C:\Users\deepu\Desktop\MyFolders\Projects\PetProjects\image_segmentation\input\png_masks")

import pandas as pd
df = pd.read_csv(r"C:\Users\deepu\Desktop\MyFolders\Projects\PetProjects\image_segmentation\input\stage_1_train_images.csv")
pos = df[df.has_pneumo == 1].new_filename.values